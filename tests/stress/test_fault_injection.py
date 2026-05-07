"""故障注入测试：网络异常、模型异常、工具故障、超时场景。

核心目标：验证 Agent 在各类故障下不会卡住，能正确返回错误状态。
"""

from __future__ import annotations

import asyncio

import pytest

from agent.loop import react_loop, DefaultContextBuilder
from agent.models import StreamEvent, ToolCall, OpenAIAdapter
from tools.orchestration import execute_tool_calls
from tests.conftest import MockModelProvider, MockModelConfig, MockTool, ErrorTool, SlowTool, make_text_events, make_tool_events


# ── 模型层故障 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fault_model_timeout():
    """模型调用超时（网络慢）不应卡住。"""
    model = MockModelProvider(MockModelConfig(delay_ms=5000, responses=[make_text_events("late")]))
    harness = MockHarness()

    # 使用 asyncio.wait_for 模拟上层超时
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            react_loop(
                prompt="test", model=model, harness=harness, tools=[],
                context_builder=DefaultContextBuilder(), max_turns=5,
            ),
            timeout=0.5,
        )


@pytest.mark.asyncio
async def test_fault_model_connection_reset():
    """模型连接重置。"""
    model = MockModelProvider(MockModelConfig(raise_on_call=ConnectionResetError("连接被重置")))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "error"
    assert "连接被重置" in result.error


@pytest.mark.asyncio
async def test_fault_model_http_500():
    """模型返回 HTTP 500（通过 error 事件模拟）。"""
    model = MockModelProvider(MockModelConfig(responses=[
        [StreamEvent(type="error", content="Internal Server Error")],
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "error"


@pytest.mark.asyncio
async def test_fault_model_malformed_stream():
    """模型流返回非预期格式（如无 done、无 tool_use）。"""
    model = MockModelProvider(MockModelConfig(responses=[
        [StreamEvent(type="text", content="hello")],  # 没有 done
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    # 不应卡住，应能正常完成
    assert result.reason == "completed"


@pytest.mark.asyncio
async def test_fault_model_empty_stream():
    """模型返回完全空流。"""
    model = MockModelProvider(MockModelConfig(responses=[[]]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"
    assert result.text == ""


@pytest.mark.asyncio
async def test_fault_model_infinite_stream():
    """模型流永不结束（无 done），abort_signal 应能中断。"""
    abort = asyncio.Event()

    async def infinite_stream():
        while True:
            yield StreamEvent(type="text", content=".")
            await asyncio.sleep(0.1)

    # 使用自定义 provider 模拟在流生成过程中可被 abort 中断
    class AbortAwareModel:
        async def complete(self, messages, system, tools, *, max_tokens=None, temperature=None, abort_event=None):
            for i in range(1000):
                if abort_event is not None and abort_event.is_set():
                    return
                yield StreamEvent(type="text", content=".")
                await asyncio.sleep(0.05)

    model = AbortAwareModel()

    async def _abort_after(delay: float):
        await asyncio.sleep(delay)
        abort.set()

    harness = MockHarness()
    asyncio.create_task(_abort_after(0.15))

    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=100,
        abort_signal=abort,
    )
    assert result.reason == "aborted"


# ── 工具层故障 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fault_tool_crash():
    """工具崩溃（抛异常）应返回错误，不中断循环。"""
    error_tool = ErrorTool()
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="ErrorTool", arguments={"message": "boom"})]),
        make_text_events("工具失败了，但我继续"),
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[error_tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"


@pytest.mark.asyncio
async def test_fault_tool_timeout():
    """工具执行超时不应卡住整个循环。"""
    # 使用 BashTool 的真实超时机制测试
    from tools.builtin.bash import BashTool
    bash = BashTool()
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={"command": "sleep 10", "timeout": 200})]),
        make_text_events("命令已超时"),
    ]))
    harness = MockHarness()

    start = asyncio.get_event_loop().time()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[bash],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert result.reason == "completed"
    # BashTool 超时 0.2s，总时间应接近 0.2s 而不是 10s
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_fault_tool_partial_failure():
    """多个工具中部分失败，其余应正常执行。"""
    read_tool = MockTool(name="Read", readonly=True)
    error_tool = ErrorTool()
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([
            ToolCall(id="tc1", name="Read", arguments={}),
            ToolCall(id="tc2", name="ErrorTool", arguments={}),
            ToolCall(id="tc3", name="Read", arguments={}),
        ]),
        make_text_events("部分失败"),
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[read_tool, error_tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"
    assert len(result.messages) == 6  # user + assistant + 3 tools + assistant(final)


@pytest.mark.asyncio
async def test_fault_tool_invalid_arguments():
    """工具参数非法（JSON 解析失败）。"""
    model = MockModelProvider(MockModelConfig(responses=[
        [StreamEvent(type="tool_use", content={
            "id": "tc1",
            "type": "function",
            "function": {"name": "Read", "arguments": "{broken json"},
        }), StreamEvent(type="done")],
        make_text_events("参数解析失败"),
    ]))
    harness = MockHarness()
    read_tool = MockTool(name="Read", readonly=True)
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[read_tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"


# ── 编排层故障 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fault_orchestration_unknown_tool():
    """调用未注册工具应返回错误。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="GhostTool", arguments={})]),
        make_text_events("未知工具"),
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"


@pytest.mark.asyncio
async def test_fault_orchestration_permission_loop():
    """权限系统被绕过不应导致无限循环。"""
    from tools.permission import PermissionEngine
    perm = PermissionEngine(deny_rules=["Read(*)"], mode="default")
    read_tool = MockTool(name="Read", readonly=True)

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={})]),
        make_tool_events([ToolCall(id="tc2", name="Read", arguments={})]),
        make_text_events("权限被拒绝"),
    ]))

    harness = MockHarnessWithPermission(perm)
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[read_tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"


# ── 循环层故障 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fault_loop_tool_result_too_large():
    """工具返回超大结果（10MB）不应导致后续循环卡住。"""
    read_tool = MockTool(name="Read", readonly=True)
    read_tool._default_result = type("R", (), {
        "output": "X" * (10 * 1024 * 1024),
        "is_error": False, "data": None, "new_messages": None,
    })()

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={})]),
        make_text_events("收到大数据"),
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[read_tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"


@pytest.mark.asyncio
async def test_fault_loop_circular_tool_calls():
    """模型反复调用相同工具（循环依赖），max_turns 应终止。"""
    read_tool = MockTool(name="Read", readonly=True)
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id=f"tc{i}", name="Read", arguments={"path": "a.txt"})])
        for i in range(100)
    ]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[read_tool],
        context_builder=DefaultContextBuilder(), max_turns=10,
    )
    assert result.reason == "max_turns"


# ── Mock 辅助类 ────────────────────────────────────────────


class MockHarness:
    def __init__(self):
        self.last_executed: list[ToolCall] = []

    async def execute_tools(self, tool_calls, context):
        self.last_executed.extend(tool_calls)
        results = []
        for tc in tool_calls:
            if tc.name == "ErrorTool":
                results.append({"tool_call_id": tc.id, "name": tc.name, "content": '{"error": "失败"}'})
            elif tc.name == "SlowTool":
                # 不在这里 sleep，由上层 wait_for 控制
                results.append({"tool_call_id": tc.id, "name": tc.name, "content": "should timeout"})
            elif tc.name == "GhostTool":
                results.append({"tool_call_id": tc.id, "name": tc.name, "content": '{"error": "未知工具"}'})
            else:
                results.append({"tool_call_id": tc.id, "name": tc.name, "content": "mock ok"})
        return results

    def check_permission(self, tool_name, args, context):
        from tools.base import PermissionDecision, PermissionTier
        return PermissionDecision(tier=PermissionTier.ALLOW)


class MockHarnessWithPermission:
    def __init__(self, perm_engine):
        self._perm = perm_engine

    async def execute_tools(self, tool_calls, context):
        results = []
        for tc in tool_calls:
            dec = self._perm.check(tc.name, tc.arguments, context)
            if not self._perm.should_proceed(dec):
                results.append({"tool_call_id": tc.id, "name": tc.name, "content": '{"error": "权限拒绝"}'})
            else:
                results.append({"tool_call_id": tc.id, "name": tc.name, "content": "ok"})
        return results

    def check_permission(self, tool_name, args, context):
        return self._perm.check(tool_name, args, context)
