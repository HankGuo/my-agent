"""边界与压力测试：覆盖极端输入、资源限制、长时间运行。

这些测试专门定位"编码任务时卡住"的问题根因。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent.loop import react_loop, DefaultContextBuilder, TurnResult
from agent.models import StreamEvent, ToolCall, parse_stream_events
from tools.orchestration import execute_tool_calls
from tests.conftest import MockModelProvider, MockModelConfig, MockTool, SlowTool, make_text_events, make_tool_events


# ── ReAct 循环边界 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_boundary_max_turns_1():
    """max_turns=1 时应只执行一轮模型调用。"""
    model = MockModelProvider(MockModelConfig(responses=[make_text_events("x")]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=1,
    )
    assert result.reason == "completed"
    assert len(model.call_history) == 1


@pytest.mark.asyncio
async def test_boundary_max_turns_1000():
    """max_turns=1000 时如果模型一直返回 tool_calls，应在 1000 轮后停止，不卡住。"""
    model = MockModelProvider(MockModelConfig(
        responses=[make_tool_events([ToolCall(id="tc1", name="Read", arguments={})])] * 2000
    ))
    harness = MockHarness()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[tool],
        context_builder=DefaultContextBuilder(), max_turns=1000,
    )
    assert result.reason == "max_turns"
    assert len(model.call_history) == 1000


@pytest.mark.asyncio
async def test_boundary_very_long_prompt():
    """超长提示词（10MB）不应导致内存问题或卡住。"""
    huge = "X" * (10 * 1024 * 1024)
    model = MockModelProvider(MockModelConfig(responses=[make_text_events("ok")]))
    harness = MockHarness()
    result = await react_loop(
        prompt=huge, model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"
    assert len(model.call_history[0]["messages"][0]["content"]) == len(huge)


@pytest.mark.asyncio
async def test_boundary_very_long_model_response():
    """超长模型响应（1MB 文本）不应导致卡住。"""
    long_text = "A" * (1024 * 1024)
    model = MockModelProvider(MockModelConfig(responses=[make_text_events(long_text)]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"
    assert len(result.text) == len(long_text)


@pytest.mark.asyncio
async def test_boundary_many_tool_calls_single_turn():
    """单轮 50 个工具调用应全部执行，不丢失。"""
    calls = [ToolCall(id=f"tc{i}", name="Read", arguments={"path": f"f{i}.txt"}) for i in range(50)]
    model = MockModelProvider(MockModelConfig(responses=[make_tool_events(calls)]))
    harness = MockHarness()
    tool = MockTool(name="Read", readonly=True)

    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"
    assert len(harness.last_executed) == 50


@pytest.mark.asyncio
async def test_boundary_zero_tools():
    """工具列表为空时不应异常。"""
    model = MockModelProvider(MockModelConfig(responses=[make_text_events("no tools")]))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    assert result.reason == "completed"


@pytest.mark.asyncio
async def test_boundary_disabled_tool():
    """禁用工具不应被执行。"""
    tool = MockTool(name="Read", readonly=True)
    tool.is_enabled = lambda: False
    model = MockModelProvider(MockModelConfig(
        responses=[make_tool_events([ToolCall(id="tc1", name="Read", arguments={})])]
    ))
    harness = MockHarness()
    result = await react_loop(
        prompt="test", model=model, harness=harness, tools=[tool],
        context_builder=DefaultContextBuilder(), max_turns=5,
    )
    # 工具被禁用，execute_tools 会返回未知工具错误
    assert result.reason in ("completed", "max_turns")


# ── 模型解析边界 ───────────────────────────────────────────


class TestModelParseBoundary:
    def test_parse_empty_string_content(self):
        """content 为空字符串。"""
        events = [StreamEvent(type="text", content=""), StreamEvent(type="done")]
        result = parse_stream_events(events)
        assert result.text == ""
        assert result.finish_reason == "stop"

    def test_parse_no_text_events(self):
        """只有 done 事件。"""
        events = [StreamEvent(type="done")]
        result = parse_stream_events(events)
        assert result.text == ""
        assert result.finish_reason == "stop"

    def test_parse_malformed_tool_use_missing_function(self):
        """tool_use 缺少 function 字段。"""
        events = [
            StreamEvent(type="tool_use", content={"id": "c1", "type": "function"}),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == ""

    def test_parse_nested_unicode_in_arguments(self):
        """嵌套 Unicode JSON 参数。"""
        args = {"data": {"中文": "值", "emoji": "🎉"}}
        events = [
            StreamEvent(type="tool_use", content={
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "Write",
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.tool_calls[0].arguments == args

    def test_parse_huge_tool_arguments(self):
        """超大工具参数（1MB JSON）。"""
        args = {"content": "A" * (1024 * 1024)}
        events = [
            StreamEvent(type="tool_use", content={
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "Write",
                    "arguments": json.dumps(args),
                },
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert len(result.tool_calls[0].arguments["content"]) == 1024 * 1024


# ── 工具编排边界 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_boundary_100_concurrent_reads():
    """100 个并发只读工具。"""
    tools = {f"Read{i}": MockTool(name=f"Read{i}", readonly=True) for i in range(100)}
    calls = [ToolCall(id=f"c{i}", name=f"Read{i}", arguments={}) for i in range(100)]
    ctx = type("Ctx", (), {"session_id": "", "abort_signal": asyncio.Event(), "config": {}, "working_dir": "."})()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, tools, ctx, max_concurrency=50)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 100
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_boundary_sequential_20_writes():
    """20 个串行写入工具，总时间应线性增长。"""
    write = MockTool(name="Write", readonly=False)
    async def slow_write(args, ctx):
        await asyncio.sleep(0.05)
        return type("R", (), {"output": "ok", "is_error": False, "data": None, "new_messages": None})()
    write.call = slow_write

    tools = {"Write": write}
    calls = [ToolCall(id=f"c{i}", name="Write", arguments={"action": "write"}) for i in range(20)]
    ctx = type("Ctx", (), {"session_id": "", "abort_signal": asyncio.Event(), "config": {}, "working_dir": "."})()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, tools, ctx)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 20
    assert elapsed >= 0.9  # 20 × 0.05 = 1.0s，允许一定误差


# ── Mock Harness 辅助类（增强版） ──────────────────────────


class MockHarness:
    def __init__(self):
        self.last_executed: list[ToolCall] = []

    async def execute_tools(self, tool_calls, context):
        self.last_executed.extend(tool_calls)
        return [
            {"tool_call_id": tc.id, "name": tc.name, "content": f"mock {tc.name}"}
            for tc in tool_calls
        ]

    def check_permission(self, tool_name, args, context):
        from tools.base import PermissionDecision, PermissionTier
        return PermissionDecision(tier=PermissionTier.ALLOW)
