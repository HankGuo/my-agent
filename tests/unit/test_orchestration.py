"""工具编排测试：覆盖分区逻辑、并发执行、权限、错误处理。"""

from __future__ import annotations

import asyncio

import pytest

from agent.models import ToolCall
from tools.orchestration import partition_tool_calls, execute_tool_calls
from tools.base import ToolResult, ToolUseContext, PermissionDecision, PermissionTier
from tests.conftest import MockTool, SlowTool, ErrorTool


# ── 分区逻辑测试 ───────────────────────────────────────────


class TestPartitionToolCalls:
    def test_empty_calls(self):
        assert partition_tool_calls([], {}) == []

    def test_all_readonly(self):
        """全部只读工具应分为一个批次。"""
        read = MockTool(name="Read", readonly=True)
        calls = [
            ToolCall(id="c1", name="Read", arguments={"path": "a.txt"}),
            ToolCall(id="c2", name="Read", arguments={"path": "b.txt"}),
            ToolCall(id="c3", name="Read", arguments={"path": "c.txt"}),
        ]
        batches = partition_tool_calls(calls, {"Read": read})
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_read_write_interleave(self):
        """读写工具交错时应正确分区。"""
        read = MockTool(name="Read", readonly=True)
        write = MockTool(name="Write", readonly=False)
        calls = [
            ToolCall(id="c1", name="Read", arguments={}),
            ToolCall(id="c2", name="Read", arguments={}),
            ToolCall(id="c3", name="Write", arguments={}),
            ToolCall(id="c4", name="Read", arguments={}),
            ToolCall(id="c5", name="Write", arguments={}),
        ]
        batches = partition_tool_calls(calls, {"Read": read, "Write": write})
        assert len(batches) == 4
        assert len(batches[0]) == 2  # 两个 Read
        assert len(batches[1]) == 1  # Write
        assert len(batches[2]) == 1  # Read
        assert len(batches[3]) == 1  # Write

    def test_write_only(self):
        """全部写入工具应每个单独一批。"""
        write = MockTool(name="Write", readonly=False)
        calls = [
            ToolCall(id="c1", name="Write", arguments={}),
            ToolCall(id="c2", name="Write", arguments={}),
        ]
        batches = partition_tool_calls(calls, {"Write": write})
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    def test_unknown_tool(self):
        """未知工具应单独成批，不影响前后批次。"""
        read = MockTool(name="Read", readonly=True)
        calls = [
            ToolCall(id="c1", name="Read", arguments={}),
            ToolCall(id="c2", name="Unknown", arguments={}),
            ToolCall(id="c3", name="Read", arguments={}),
        ]
        batches = partition_tool_calls(calls, {"Read": read})
        assert len(batches) == 3
        assert batches[1][0].name == "Unknown"


# ── 执行逻辑测试 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_single_tool():
    """单工具正确执行。"""
    read = MockTool(name="Read", readonly=True)
    read._default_result = ToolResult(output="content")
    ctx = ToolUseContext()

    results = await execute_tool_calls(
        [ToolCall(id="c1", name="Read", arguments={"action": "read", "path": "x.txt"})],
        {"Read": read},
        ctx,
    )

    assert len(results) == 1
    assert results[0]["tool_call_id"] == "c1"
    assert results[0]["content"] == "content"


@pytest.mark.asyncio
async def test_execute_concurrent_reads():
    """多个只读工具应并发执行。"""
    delays = [0.2, 0.2, 0.2]
    tools = {
        f"Slow{i}": SlowTool(name=f"Slow{i}", delay_sec=d)
        for i, d in enumerate(delays)
    }
    calls = [ToolCall(id=f"c{i}", name=f"Slow{i}", arguments={"delay": d}) for i, d in enumerate(delays)]
    ctx = ToolUseContext()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, tools, ctx, max_concurrency=10)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 3
    # 并发执行，总时间应接近 0.2s，而不是 0.6s
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_execute_serial_writes():
    """写入工具应串行执行。"""
    delays = [0.15, 0.15]
    write = MockTool(name="Write", readonly=False)
    # 通过 sleep 模拟慢写入
    original_call = write.call

    async def slow_call(args, ctx):
        await asyncio.sleep(0.15)
        return ToolResult(output="written")

    write.call = slow_call

    calls = [
        ToolCall(id="c1", name="Write", arguments={"action": "write", "path": "a.txt"}),
        ToolCall(id="c2", name="Write", arguments={"action": "write", "path": "b.txt"}),
    ]
    ctx = ToolUseContext()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, {"Write": write}, ctx)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 2
    # 串行执行，总时间应 ≥ 0.3s
    assert elapsed >= 0.25


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    """未知工具返回错误信息。"""
    ctx = ToolUseContext()
    results = await execute_tool_calls(
        [ToolCall(id="c1", name="Ghost", arguments={})],
        {},
        ctx,
    )
    assert len(results) == 1
    assert "未知工具" in results[0]["content"]


@pytest.mark.asyncio
async def test_execute_tool_error_handling():
    """工具抛出异常时应返回错误结果，不中断其他工具。"""
    read = MockTool(name="Read", readonly=True)
    error_tool = ErrorTool()
    ctx = ToolUseContext()

    results = await execute_tool_calls(
        [
            ToolCall(id="c1", name="Read", arguments={"action": "read"}),
            ToolCall(id="c2", name="ErrorTool", arguments={"message": "boom"}),
            ToolCall(id="c3", name="Read", arguments={"action": "read"}),
        ],
        {"Read": read, "ErrorTool": error_tool},
        ctx,
    )

    assert len(results) == 3
    assert results[0]["content"] == "mock ok"
    assert "执行失败" in results[1]["content"] or "故意失败" in results[1]["content"]
    assert results[2]["content"] == "mock ok"


@pytest.mark.asyncio
async def test_execute_permission_denied():
    """权限拒绝时应返回权限错误。"""
    from tools.permission import PermissionEngine

    read = MockTool(name="Read", readonly=True)
    ctx = ToolUseContext()
    perm = PermissionEngine(allow_rules=[], deny_rules=["Read(*)"], ask_rules=[], mode="default")

    results = await execute_tool_calls(
        [ToolCall(id="c1", name="Read", arguments={"path": "secret.txt"})],
        {"Read": read},
        ctx,
        permission_engine=perm,
    )

    assert len(results) == 1
    assert "权限拒绝" in results[0]["content"]


@pytest.mark.asyncio
async def test_execute_validation_failure():
    """输入校验失败应返回错误。"""
    read = MockTool(name="Read", readonly=True)
    ctx = ToolUseContext()

    # 覆盖 validate_input 使其失败
    read.validate_input = lambda args: (False, "path 不能为空")

    results = await execute_tool_calls(
        [ToolCall(id="c1", name="Read", arguments={})],
        {"Read": read},
        ctx,
    )

    assert len(results) == 1
    assert "输入校验失败" in results[0]["content"]


@pytest.mark.asyncio
async def test_concurrency_limit():
    """max_concurrency 限制应生效。"""
    delays = [0.3] * 5
    tools = {f"Slow{i}": SlowTool(name=f"Slow{i}", delay_sec=d) for i, d in enumerate(delays)}
    calls = [ToolCall(id=f"c{i}", name=f"Slow{i}", arguments={"delay": d}) for i, d in enumerate(delays)]
    ctx = ToolUseContext()

    start = asyncio.get_event_loop().time()
    results = await execute_tool_calls(calls, tools, ctx, max_concurrency=2)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 5
    # 5个任务，并发度2，每个0.3s，总时间应 ≥ 0.9s（3批 × 0.3s）
    assert elapsed >= 0.6
