"""测试基础设施：Mock 组件、Fixtures、辅助函数。"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest

from agent.models import ModelProvider, StreamEvent, ToolCall
from tools.base import Tool, ToolResult, ToolUseContext


# ── Mock 模型 Provider ─────────────────────────────────────


@dataclass
class MockModelConfig:
    """配置 Mock 模型的行为。"""
    responses: list[list[StreamEvent]] = field(default_factory=list)
    response_index: int = 0
    delay_ms: float = 0.0
    raise_on_call: Exception | None = None


class MockModelProvider:
    """可编程的 Mock 模型，用于精确控制 ReAct 循环的输入输出。"""

    def __init__(self, config: MockModelConfig | None = None):
        self.config = config or MockModelConfig()
        self.calls: list[dict] = field(default_factory=list)
        self._calls: list[dict] = []

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
        """返回预设的流式事件序列。"""
        self._calls.append({
            "messages": messages,
            "system": system,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self.config.raise_on_call:
            raise self.config.raise_on_call

        if self.config.delay_ms > 0:
            await asyncio.sleep(self.config.delay_ms / 1000)

        idx = self.config.response_index
        if idx < len(self.config.responses):
            events = self.config.responses[idx]
            self.config.response_index += 1
            for ev in events:
                if abort_event is not None and abort_event.is_set():
                    yield StreamEvent(type="error", content="请求被中断")
                    return
                yield ev
        else:
            # 默认返回空完成
            yield StreamEvent(type="done")

    @property
    def call_history(self) -> list[dict]:
        return self._calls


def make_text_events(text: str, finish: bool = True) -> list[StreamEvent]:
    """构造纯文本响应的流事件序列。"""
    events = [StreamEvent(type="text", content=text)]
    if finish:
        events.append(StreamEvent(type="done"))
    return events


def make_tool_events(tool_calls: list[ToolCall], text: str = "") -> list[StreamEvent]:
    """构造带工具调用的流事件序列。"""
    events: list[StreamEvent] = []
    if text:
        events.append(StreamEvent(type="text", content=text))
    for tc in tool_calls:
        events.append(StreamEvent(type="tool_use", content={
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            },
        }))
    events.append(StreamEvent(type="done"))
    return events


def make_reasoning_events(text: str, reasoning: str, finish: bool = True) -> list[StreamEvent]:
    """构造带 reasoning_content 的流事件序列（DeepSeek 风格）。"""
    events = [
        StreamEvent(type="reasoning", content=reasoning),
        StreamEvent(type="text", content=text),
    ]
    if finish:
        events.append(StreamEvent(type="done"))
    return events


# ── Mock 工具 ──────────────────────────────────────────────


class MockTool:
    """可编程 Mock 工具，用于测试工具调用链。"""

    description = "一个用于测试的 Mock 工具"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(self, name: str = "MockTool", readonly: bool = False, result_map: dict[str, ToolResult] | None = None):
        self.name = name
        self._readonly = readonly
        self._result_map = result_map or {}
        self._default_result = ToolResult(output="mock ok")
        self.call_history: list[dict] = []

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        self.call_history.append({"args": args, "context": context})
        key = args.get("action", "default")
        return self._result_map.get(key, self._default_result)

    def is_readonly(self, args: dict) -> bool:
        return self._readonly

    def is_concurrency_safe(self, args: dict) -> bool:
        return self._readonly

    def is_destructive(self, args: dict) -> bool:
        return False

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> Any:
        from tools.base import PermissionDecision, PermissionTier
        return PermissionDecision(tier=PermissionTier.ALLOW)

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        if "action" not in args:
            return False, "缺少 action"
        return True, None


class SlowTool:
    """模拟慢工具，用于测试超时和并发。"""

    description = "故意慢的工具"
    input_schema = {"type": "object", "properties": {"delay": {"type": "number"}}, "required": ["delay"]}

    def __init__(self, name: str = "SlowTool", delay_sec: float = 1.0):
        self.name = name
        self.delay_sec = delay_sec
        self.call_history: list[dict] = []

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        delay = args.get("delay", self.delay_sec)
        self.call_history.append({"args": args, "delay": delay})
        await asyncio.sleep(delay)
        return ToolResult(output=f"slept {delay}s")

    def is_readonly(self, args: dict) -> bool:
        return True

    def is_concurrency_safe(self, args: dict) -> bool:
        return True

    def is_enabled(self) -> bool:
        return True

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        return True, None


class ErrorTool:
    """模拟失败工具，用于测试错误处理。"""

    description = "总是失败的工具"
    input_schema = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}

    def __init__(self, name: str = "ErrorTool", exception: Exception | None = None):
        self.name = name
        self._exception = exception or RuntimeError("工具故意失败")

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        raise self._exception

    def is_readonly(self, args: dict) -> bool:
        return False

    def is_concurrency_safe(self, args: dict) -> bool:
        return False

    def is_enabled(self) -> bool:
        return True

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        return True, None


# ── Pytest Fixtures ────────────────────────────────────────


@pytest.fixture
def event_loop():
    """提供显式的事件循环，避免 pytest-asyncio 警告。"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """提供临时目录，测试结束后自动清理。"""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def mock_model():
    """提供默认 Mock 模型。"""
    return MockModelProvider()


@pytest.fixture
def mock_tools():
    """提供一组标准 Mock 工具。"""
    return [
        MockTool(name="ReadFile", readonly=True),
        MockTool(name="WriteFile", readonly=False),
    ]


@pytest.fixture
def tool_context(temp_dir):
    """提供标准 ToolUseContext。"""
    return ToolUseContext(session_id="test-session", working_dir=str(temp_dir))
