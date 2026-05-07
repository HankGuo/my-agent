"""核心 ReAct 循环。≤500 行硬约束。模型只推理，Harness 管执行。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

from agent.models import ModelProvider, ModelResponse, StreamEvent, ToolCall, parse_stream_events, tool_to_openai_schema
from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.loop")

# 工具结果进入上下文的最大长度，超过则截断
_MAX_TOOL_RESULT_CHARS = 8_000


# ── 数据类型 ──────────────────────────────────────────────


@dataclass
class Message:
    """对话消息。"""
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str | None = None
    reasoning_content: str | None = None  # DeepSeek 思考模式的推理内容
    tool_calls: list[dict] | None = None  # assistant 发起的工具调用
    tool_call_id: str | None = None  # tool 消息对应的调用 ID
    name: str | None = None  # tool 消息对应的工具名


@dataclass
class TurnState:
    """跨轮次状态。"""
    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    transition: str | None = None  # 上一轮继续的原因


@dataclass
class TurnResult:
    """循环返回结果。"""
    reason: str  # "completed" | "max_turns" | "aborted" | "error"
    text: str = ""
    messages: list[Message] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None


# ── Harness 协议 ───────────────────────────────────────────


@runtime_checkable
class ToolExecutor(Protocol):
    """Harness 接口：工具执行、权限判断。"""

    async def execute_tools(
        self,
        tool_calls: list[ToolCall],
        context: ToolUseContext,
    ) -> list[dict]: ...  # 返回 tool result messages

    def check_permission(
        self, tool_name: str, args: dict, context: ToolUseContext
    ) -> PermissionDecision: ...


# ── 上下文构建 ────────────────────────────────────────────


@runtime_checkable
class ContextBuilder(Protocol):
    """构建系统提示和上下文。"""

    def build_system_prompt(self, tools: list[Tool]) -> str: ...
    def build_messages(self, state: TurnState) -> list[dict]: ...


# ── 核心 ReAct 循环 ───────────────────────────────────────


async def react_loop(
    prompt: str,
    model: ModelProvider,
    harness: ToolExecutor,
    tools: list[Tool],
    context_builder: ContextBuilder,
    *,
    max_turns: int = 50,
    session_id: str = "default",
    abort_signal: asyncio.Event | None = None,
) -> TurnResult:
    """
    ReAct 循环。

    流程：用户输入 → 构建上下文 → 模型推理 → 工具调用(如有) → 下一轮 / 返回
    """
    state = TurnState()
    state.messages.append(Message(role="user", content=prompt))
    tool_use_ctx = ToolUseContext(session_id=session_id)
    if abort_signal:
        tool_use_ctx.abort_signal = abort_signal

    # 构建 OpenAI 格式的工具 schema
    tool_schemas = [tool_to_openai_schema(t.name, t.description, t.input_schema) for t in tools if t.is_enabled()]
    tool_map = {t.name: t for t in tools}

    system_prompt = context_builder.build_system_prompt(tools)

    response: ModelResponse | None = None

    for turn in range(max_turns):
        state.turn_count = turn + 1

        # 检查中断信号
        if abort_signal and abort_signal.is_set():
            return TurnResult(reason="aborted", messages=state.messages)

        # 构建模型请求
        api_messages = context_builder.build_messages(state)

        # 调用模型（传递 abort_signal 以便在流式响应中被中断）
        events: list[StreamEvent] = []
        try:
            async for event in model.complete(
                messages=api_messages,
                system=system_prompt,
                tools=tool_schemas,
                abort_event=tool_use_ctx.abort_signal,
            ):
                events.append(event)
        except Exception as e:
            logger.error("模型调用失败: %s", e)
            return TurnResult(reason="error", messages=state.messages, error=str(e))

        response = parse_stream_events(events)

        # 模型调用结束后再次检查中断（流式响应被 abort 后需在此终止）
        if abort_signal and abort_signal.is_set():
            return TurnResult(reason="aborted", messages=state.messages)

        if response.finish_reason == "error":
            return TurnResult(reason="error", messages=state.messages, error="模型返回错误")

        # 记录 assistant 消息
        assistant_msg = Message(
            role="assistant",
            content=response.text or None,
            reasoning_content=response.reasoning_content or None,
        )
        if response.tool_calls:
            assistant_msg.tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
        state.messages.append(assistant_msg)

        # 无工具调用 → 完成
        if not response.tool_calls:
            return TurnResult(
                reason="completed",
                text=response.text,
                messages=state.messages,
                usage=response.usage,
            )

        # 有工具调用 → 执行
        tool_results = await harness.execute_tools(response.tool_calls, tool_use_ctx)

        # 将 tool results 追加到消息（超长结果截断，防止上下文膨胀）
        for result_msg in tool_results:
            content = result_msg.get("content", "")
            if len(content) > _MAX_TOOL_RESULT_CHARS:
                content = content[:_MAX_TOOL_RESULT_CHARS] + "\n...（结果过长，已截断）"
            state.messages.append(Message(
                role="tool",
                content=content,
                tool_call_id=result_msg.get("tool_call_id", ""),
                name=result_msg.get("name", ""),
            ))

        # 继续下一轮

    return TurnResult(
        reason="max_turns",
        text=response.text if response else "",
        messages=state.messages,
    )


# ── 默认上下文构建器 ──────────────────────────────────────


class DefaultContextBuilder:
    """最简上下文构建：系统提示 + OpenAI 格式消息。"""

    def __init__(self, system_prompt: str = ""):
        self._system_prompt = system_prompt

    def build_system_prompt(self, tools: list[Tool]) -> str:
        parts = []
        if self._system_prompt:
            parts.append(self._system_prompt)
        if tools:
            tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools if t.is_enabled())
            parts.append(f"可用工具:\n{tool_list}")
        return "\n\n".join(parts)

    def build_messages(self, state: TurnState) -> list[dict]:
        result = []
        for msg in state.messages:
            d: dict[str, Any] = {"role": msg.role}
            if msg.content is not None:
                d["content"] = msg.content
            elif msg.role == "assistant" and msg.tool_calls:
                # DeepSeek 等模型要求 assistant 消息有 content 字段
                d["content"] = None
            if msg.reasoning_content is not None:
                d["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:
                d["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
            result.append(d)
        return result
