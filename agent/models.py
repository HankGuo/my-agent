"""模型 Provider 层：OpenAI 兼容格式适配器，覆盖 DeepSeek/Qwen/ChatGLM 等。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

import httpx

logger = logging.getLogger("my-agent.models")


# ── 核心类型 ──────────────────────────────────────────────


@dataclass
class StreamEvent:
    """统一的流式事件，跨所有 Provider。"""
    type: str  # "text" | "reasoning" | "tool_use" | "tool_result" | "error" | "done"
    content: str | dict | None = None
    tool_calls: list[dict] | None = None


@dataclass
class ToolCall:
    """模型发起的工具调用。"""
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """模型完整响应（流结束后汇总）。"""
    text: str = ""
    reasoning_content: str = ""  # DeepSeek 思考模式的推理内容
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


# ── Provider 协议 ─────────────────────────────────────────


@runtime_checkable
class ModelProvider(Protocol):
    """模型推理协议。模型只做推理，不执行工具。"""

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]: ...


# ── OpenAI 兼容适配器 ────────────────────────────────────


class OpenAIAdapter:
    """
    OpenAI 兼容格式适配器。

    覆盖：OpenAI、DeepSeek、Qwen（通义千问）、ChatGLM（智谱）、
    以及所有兼容 /v1/chat/completions 接口的 Provider。

    特殊处理：
    - DeepSeek 思考模式（reasoning_content）需回传给 API
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """调用 OpenAI 兼容 API，流式返回 StreamEvent。"""
        request_messages = []
        if system:
            request_messages.append({"role": "system", "content": system})
        request_messages.extend(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        accumulated_tool_calls: dict[int, dict] = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, json=body, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    yield StreamEvent(
                        type="error",
                        content=f"API 请求失败 (HTTP {response.status_code}): {error_body.decode()}",
                    )
                    return

                async for line in response.aiter_lines():
                    # 定期检查中断信号，避免流式响应卡住时无法终止
                    if abort_event is not None and abort_event.is_set():
                        return

                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        for tc in accumulated_tool_calls.values():
                            yield StreamEvent(type="tool_use", content=tc)
                        yield StreamEvent(type="done")
                        return

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # DeepSeek 思考模式：reasoning_content
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield StreamEvent(type="reasoning", content=reasoning)

                    # 文本内容
                    content = delta.get("content")
                    if content:
                        yield StreamEvent(type="text", content=content)

                    # 工具调用（增量）
                    tool_calls_delta = delta.get("tool_calls")
                    if tool_calls_delta:
                        for tc_delta in tool_calls_delta:
                            idx = tc_delta.get("index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            tc_acc = accumulated_tool_calls[idx]
                            if tc_delta.get("id"):
                                tc_acc["id"] = tc_delta["id"]
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                tc_acc["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tc_acc["function"]["arguments"] += fn["arguments"]

                    # 流结束
                    if finish_reason in ("tool_calls", "stop"):
                        for tc in accumulated_tool_calls.values():
                            yield StreamEvent(type="tool_use", content=tc)
                        accumulated_tool_calls.clear()
                        yield StreamEvent(type="done")
                        return


# ── 工具调用解析 ──────────────────────────────────────────


def parse_stream_events(events: list[StreamEvent]) -> ModelResponse:
    """将流式事件列表汇总为 ModelResponse。"""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage: dict = {}

    for event in events:
        if event.type == "text" and isinstance(event.content, str):
            text_parts.append(event.content)
        elif event.type == "reasoning" and isinstance(event.content, str):
            reasoning_parts.append(event.content)
        elif event.type == "tool_use" and isinstance(event.content, dict):
            tc = event.content
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"_raw": args_str}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=args,
            ))
        elif event.type == "error":
            return ModelResponse(text="", tool_calls=[], finish_reason="error")

    return ModelResponse(
        text="".join(text_parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage,
    )


# ── 工具 Schema 转换 ──────────────────────────────────────


def tool_to_openai_schema(name: str, description: str, input_schema: dict) -> dict:
    """将内部 Tool 定义转换为 OpenAI function calling 格式。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
    }


# ── Provider 工厂 ─────────────────────────────────────────


def create_provider(
    provider_name: str,
    provider_config: dict,
    model_name: str | None = None,
) -> OpenAIAdapter:
    """根据配置创建模型 Provider。"""
    base_url = provider_config.get("base_url", "")
    api_key = provider_config.get("api_key", "")
    model = model_name or provider_config.get("models", [""])[0]

    if not base_url:
        defaults = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "chatglm": "https://open.bigmodel.cn/api/paas/v4",
        }
        base_url = defaults.get(provider_name, "")

    if not api_key:
        raise ValueError(f"Provider '{provider_name}' 缺少 api_key 配置")

    return OpenAIAdapter(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
