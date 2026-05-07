"""ReAct 循环单元测试：覆盖终止条件、状态管理、边界场景。"""

from __future__ import annotations

import asyncio

import pytest

from agent.loop import react_loop, DefaultContextBuilder, Message, TurnResult
from agent.models import StreamEvent, ToolCall
from tests.conftest import MockModelProvider, MockModelConfig, MockTool, make_text_events, make_tool_events


# ── 基础流程测试 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_react_loop_simple_completion():
    """最简场景：用户输入 → 模型直接返回文本 → 完成。"""
    model = MockModelProvider(
        MockModelConfig(responses=[make_text_events("你好，世界！")])
    )
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Dummy", readonly=True)

    result = await react_loop(
        prompt="打个招呼",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=5,
    )

    assert result.reason == "completed"
    assert result.text == "你好，世界！"
    assert len(result.messages) == 2  # user + assistant
    assert result.error is None


@pytest.mark.asyncio
async def test_react_loop_single_tool_call():
    """单轮工具调用：模型调用工具，工具返回结果，模型总结。"""
    read_tool = MockTool(name="Read", readonly=True)
    read_tool._default_result = {"output": "文件内容: hello"}

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": "/tmp/test.txt"})], text="我来读取文件"),
        make_text_events("文件内容是 hello"),
    ]))

    harness = MockHarness()
    ctx = DefaultContextBuilder()

    result = await react_loop(
        prompt="读取文件",
        model=model,
        harness=harness,
        tools=[read_tool],
        context_builder=ctx,
        max_turns=5,
    )

    assert result.reason == "completed"
    assert result.text == "文件内容是 hello"
    assert len(result.messages) == 4  # user + assistant(tool) + tool + assistant(final)


@pytest.mark.asyncio
async def test_react_loop_multi_turn_tool_chain():
    """多轮工具调用链：Read → Write → Read 验证。"""
    read_tool = MockTool(name="Read", readonly=True)
    write_tool = MockTool(name="Write", readonly=False)

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": "a.txt"})]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={"path": "b.txt", "content": "copied"})]),
        make_tool_events([ToolCall(id="tc3", name="Read", arguments={"path": "b.txt"})]),
        make_text_events("完成复制并验证"),
    ]))

    harness = MockHarness()
    ctx = DefaultContextBuilder()

    result = await react_loop(
        prompt="复制文件并验证",
        model=model,
        harness=harness,
        tools=[read_tool, write_tool],
        context_builder=ctx,
        max_turns=10,
    )

    assert result.reason == "completed"
    assert result.text == "完成复制并验证"
    assert len(result.messages) == 8  # 4轮 × 2 (assistant + tool)


@pytest.mark.asyncio
async def test_react_loop_max_turns():
    """达到 max_turns 时正确终止，不卡住。"""
    model = MockModelProvider(MockModelConfig(
        responses=[make_tool_events([ToolCall(id=f"tc{i}", name="Read", arguments={})]) for i in range(10)]
    ))
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="循环测试",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=3,
    )

    assert result.reason == "max_turns"
    assert result.error is None
    assert len(result.messages) == 7  # user + 3轮×(assistant+tool)


@pytest.mark.asyncio
async def test_react_loop_aborted_by_signal():
    """abort_signal 被设置时立即终止，不卡住。"""
    # 先设置 abort，再启动循环，确保第一轮就被中断
    abort = asyncio.Event()
    abort.set()  # 立即设置中断

    model = MockModelProvider(MockModelConfig(responses=[make_text_events("x")]))
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="应该被中断",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=100,
        abort_signal=abort,
    )
    assert result.reason == "aborted"


# ── 错误处理测试 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_react_loop_model_error():
    """模型调用抛出异常时返回 error 状态。"""
    model = MockModelProvider(MockModelConfig(raise_on_call=ConnectionError("网络断开")))
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="触发错误",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=5,
    )

    assert result.reason == "error"
    assert "网络断开" in result.error


@pytest.mark.asyncio
async def test_react_loop_empty_prompt():
    """空提示词不应导致卡住或异常。"""
    model = MockModelProvider(MockModelConfig(responses=[make_text_events("收到空输入")]))
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=5,
    )

    assert result.reason == "completed"


@pytest.mark.asyncio
async def test_react_loop_no_tools():
    """没有工具时模型直接返回文本。"""
    model = MockModelProvider(MockModelConfig(responses=[make_text_events("没有工具可用")]))
    harness = MockHarness()
    ctx = DefaultContextBuilder()

    result = await react_loop(
        prompt="测试",
        model=model,
        harness=harness,
        tools=[],
        context_builder=ctx,
        max_turns=5,
    )

    assert result.reason == "completed"
    assert result.text == "没有工具可用"


# ── 边界条件测试 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_react_loop_zero_max_turns():
    """max_turns=0 时应立即返回 max_turns。"""
    model = MockModelProvider(MockModelConfig())
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="测试",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=0,
    )

    assert result.reason == "max_turns"


@pytest.mark.asyncio
async def test_react_loop_huge_prompt():
    """超大提示词不应导致内存溢出或卡住。"""
    huge = "A" * 500_000
    model = MockModelProvider(MockModelConfig(responses=[make_text_events("收到长文本")]))
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt=huge,
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=5,
    )

    assert result.reason == "completed"
    assert model.call_history[0]["messages"][0]["content"] == huge


@pytest.mark.asyncio
async def test_react_loop_model_returns_only_tool_use_no_done():
    """模型返回 tool_use 但没有 done 事件（异常情况）。"""
    events = [
        StreamEvent(type="tool_use", content={
            "id": "tc1",
            "type": "function",
            "function": {"name": "Read", "arguments": "{}"},
        }),
        # 故意缺少 done
    ]
    model = MockModelProvider(MockModelConfig(responses=[events]))
    harness = MockHarness()
    ctx = DefaultContextBuilder()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="测试",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=ctx,
        max_turns=5,
    )

    # 此时 parse_stream_events 会识别到 tool_calls，finish_reason 为 tool_calls
    # 但由于没有 done 事件，循环会继续执行工具
    assert result.reason in ("completed", "max_turns")


# ── Mock Harness 辅助类 ────────────────────────────────────


class MockHarness:
    """用于单元测试的最简 Harness。"""

    async def execute_tools(self, tool_calls, context):
        from tools.base import ToolResult
        results = []
        for tc in tool_calls:
            # 简单模拟：返回工具名和参数
            content = f"执行 {tc.name}，参数: {tc.arguments}"
            results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": content,
            })
        return results

    def check_permission(self, tool_name, args, context):
        from tools.base import PermissionDecision, PermissionTier
        return PermissionDecision(tier=PermissionTier.ALLOW)
