"""工具分区并发执行：只读并行，写入串行。来自 claude-code 的 toolOrchestration 模式。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision
from agent.models import ToolCall

logger = logging.getLogger("my-agent.orchestration")


def partition_tool_calls(
    tool_calls: list[ToolCall],
    tools: dict[str, Tool],
) -> list[list[ToolCall]]:
    """
    将工具调用分区：连续的只读工具分为一组（并发执行），写入工具单独一组（串行执行）。

    来自 claude-code 的 partitionToolCalls 模式。
    """
    if not tool_calls:
        return []

    batches: list[list[ToolCall]] = []
    current_batch: list[ToolCall] = []

    for tc in tool_calls:
        tool = tools.get(tc.name)
        if tool is None:
            # 未知工具，单独执行以报告错误
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append([tc])
            continue

        if tool.is_concurrency_safe(tc.arguments):
            current_batch.append(tc)
        else:
            # 写入工具：先提交当前批次，再单独一组
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append([tc])

    if current_batch:
        batches.append(current_batch)

    return batches


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    tools: dict[str, Tool],
    context: ToolUseContext,
    permission_engine: Any = None,
    max_concurrency: int = 10,
) -> list[dict]:
    """
    执行工具调用列表，返回 tool result 消息列表。

    分区并发：同一批内的只读工具并行执行，写入工具串行执行。
    """
    batches = partition_tool_calls(tool_calls, tools)
    all_results: list[dict] = []

    for batch in batches:
        if len(batch) == 1:
            # 单个工具调用（写入工具或独立调用）
            result = await _execute_single(batch[0], tools, context, permission_engine)
            all_results.append(result)
        else:
            # 并发执行
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _guarded(tc: ToolCall) -> dict:
                async with semaphore:
                    return await _execute_single(tc, tools, context, permission_engine)

            results = await asyncio.gather(*[_guarded(tc) for tc in batch])
            all_results.extend(results)

    return all_results


async def _execute_single(
    tc: ToolCall,
    tools: dict[str, Tool],
    context: ToolUseContext,
    permission_engine: Any = None,
) -> dict:
    """执行单个工具调用，返回 tool result 消息。"""
    tool = tools.get(tc.name)

    if tool is None:
        return _tool_error(tc.id, tc.name, f"未知工具: {tc.name}")

    # 权限检查
    if permission_engine is not None:
        decision = permission_engine.check(tc.name, tc.arguments, context)
        if not permission_engine.should_proceed(decision):
            return _tool_error(tc.id, tc.name, f"权限拒绝: {decision.reason}")

    # 输入校验
    valid, err = tool.validate_input(tc.arguments)
    if not valid:
        return _tool_error(tc.id, tc.name, f"输入校验失败: {err}")

    # 执行
    try:
        result = await tool.call(tc.arguments, context)
        content = result.output if result.output else json.dumps(result.data, ensure_ascii=False, default=str)
        return {
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": content,
        }
    except Exception as e:
        logger.error("工具执行失败 %s: %s", tc.name, e)
        return _tool_error(tc.id, tc.name, f"执行失败: {e}")


def _tool_error(tool_call_id: str, name: str, error: str) -> dict:
    """构造工具错误结果消息。"""
    return {
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps({"error": error}, ensure_ascii=False),
    }
