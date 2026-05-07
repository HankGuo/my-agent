"""并发压力测试：多会话、高并发工具调用、资源竞争。

验证系统在并发场景下的稳定性和性能。
"""

from __future__ import annotations

import asyncio

import pytest

from agent.loop import react_loop, DefaultContextBuilder
from agent.models import ToolCall
from tools.orchestration import execute_tool_calls
from tests.conftest import MockModelProvider, MockModelConfig, MockTool, SlowTool, ErrorTool, make_text_events, make_tool_events


# ── 单会话高并发工具 ───────────────────────────────────────


@pytest.mark.asyncio
async def test_stress_50_parallel_reads():
    """单轮 50 个并发读取。"""
    tools = {f"R{i}": MockTool(name=f"R{i}", readonly=True) for i in range(50)}
    calls = [ToolCall(id=f"c{i}", name=f"R{i}", arguments={}) for i in range(50)]
    ctx = type("Ctx", (), {"session_id": "s1", "abort_signal": asyncio.Event(), "config": {}, "working_dir": "."})()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, tools, ctx, max_concurrency=50)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 50
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_stress_mixed_read_write_partitioning():
    """大量读写交错的正确分区。"""
    read = MockTool(name="Read", readonly=True)
    write = MockTool(name="Write", readonly=False)
    tools = {"Read": read, "Write": write}

    # 模式: R R W R W R R W
    calls = [
        ToolCall(id="c1", name="Read", arguments={}),
        ToolCall(id="c2", name="Read", arguments={}),
        ToolCall(id="c3", name="Write", arguments={}),
        ToolCall(id="c4", name="Read", arguments={}),
        ToolCall(id="c5", name="Write", arguments={}),
        ToolCall(id="c6", name="Read", arguments={}),
        ToolCall(id="c7", name="Read", arguments={}),
        ToolCall(id="c8", name="Write", arguments={}),
    ]
    ctx = type("Ctx", (), {"session_id": "s1", "abort_signal": asyncio.Event(), "config": {}, "working_dir": "."})()

    results = await execute_tool_calls(calls, tools, ctx)
    assert len(results) == 8


# ── 多会话并发 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stress_10_sessions_parallel():
    """10 个独立 ReAct 会话并发执行。"""
    async def session_task(sid: str):
        model = MockModelProvider(MockModelConfig(responses=[
            make_text_events(f"session {sid} done"),
        ]))
        harness = MockHarness()
        return await react_loop(
            prompt=f"session {sid}",
            model=model,
            harness=harness,
            tools=[],
            context_builder=DefaultContextBuilder(),
            max_turns=5,
        )

    tasks = [session_task(f"s{i}") for i in range(10)]
    start = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    elapsed = asyncio.get_event_loop().time() - start

    assert all(r.reason == "completed" for r in results)
    assert elapsed < 2.0  # 纯 Mock，应很快


@pytest.mark.asyncio
async def test_stress_5_sessions_with_tools():
    """5 个会话同时执行工具调用。"""
    tool = MockTool(name="Read", readonly=True)

    async def session_task(sid: str):
        model = MockModelProvider(MockModelConfig(responses=[
            make_tool_events([ToolCall(id=f"{sid}_tc1", name="Read", arguments={"path": f"{sid}.txt"})]),
            make_text_events(f"{sid} ok"),
        ]))
        harness = MockHarness()
        return await react_loop(
            prompt=f"read {sid}.txt",
            model=model,
            harness=harness,
            tools=[tool],
            context_builder=DefaultContextBuilder(),
            max_turns=5,
        )

    results = await asyncio.gather(*[session_task(f"s{i}") for i in range(5)])
    assert all(r.reason == "completed" for r in results)


# ── Semaphore 压力 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_stress_semaphore_contention():
    """并发度限制下的任务排队。"""
    tools = {f"S{i}": SlowTool(name=f"S{i}", delay_sec=0.2) for i in range(10)}
    calls = [ToolCall(id=f"c{i}", name=f"S{i}", arguments={"delay": 0.2}) for i in range(10)]
    ctx = type("Ctx", (), {"session_id": "s1", "abort_signal": asyncio.Event(), "config": {}, "working_dir": "."})()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, tools, ctx, max_concurrency=2)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 10
    # 10 任务，并发度 2，每个 0.2s，理论最少 1.0s
    assert elapsed >= 0.8


# ── 长时间运行压力 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_stress_long_running_loop():
    """模拟长时间运行的 Agent（50 轮）。"""
    tool = MockTool(name="Read", readonly=True)
    responses = []
    for i in range(50):
        responses.append(make_tool_events([ToolCall(id=f"tc{i}", name="Read", arguments={})]))
    responses.append(make_text_events("完成50轮"))

    model = MockModelProvider(MockModelConfig(responses=responses))
    harness = MockHarness()

    start = asyncio.get_event_loop().time()
    result = await react_loop(
        prompt="long run",
        model=model,
        harness=harness,
        tools=[tool],
        context_builder=DefaultContextBuilder(),
        max_turns=100,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert result.reason == "completed"
    assert len(result.messages) == 102  # user + 50*(assistant+tool) + assistant
    assert elapsed < 5.0


# ── 资源释放测试 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_stress_resource_cleanup_after_error():
    """大量错误后资源正确释放。"""
    error_tool = ErrorTool()
    responses = []
    for i in range(20):
        responses.append(make_tool_events([ToolCall(id=f"tc{i}", name="ErrorTool", arguments={})]))
    responses.append(make_text_events("结束"))

    model = MockModelProvider(MockModelConfig(responses=responses))
    harness = MockHarness()

    result = await react_loop(
        prompt="errors",
        model=model,
        harness=harness,
        tools=[error_tool],
        context_builder=DefaultContextBuilder(),
        max_turns=50,
    )
    assert result.reason == "completed"
    # 验证没有未处理的异常泄漏


# ── Mock Harness ───────────────────────────────────────────


class MockHarness:
    def __init__(self):
        self.last_executed: list[ToolCall] = []

    async def execute_tools(self, tool_calls, context):
        self.last_executed.extend(tool_calls)
        return [
            {"tool_call_id": tc.id, "name": tc.name, "content": "mock ok"}
            for tc in tool_calls
        ]

    def check_permission(self, tool_name, args, context):
        from tools.base import PermissionDecision, PermissionTier
        return PermissionDecision(tier=PermissionTier.ALLOW)
