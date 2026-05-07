"""Tool Protocol 与核心类型定义。所有工具必须满足此接口。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class PermissionTier(Enum):
    """权限层级。"""
    DENY = "deny"
    ASK = "ask"
    ALLOW = "allow"


@dataclass
class PermissionDecision:
    """权限判定结果。"""
    tier: PermissionTier
    reason: str = ""


@dataclass
class ToolResult:
    """工具执行结果。"""
    data: Any = None
    output: str = ""
    is_error: bool = False
    new_messages: list[dict] | None = None


@dataclass
class ToolUseContext:
    """工具执行上下文，由 Harness 注入。"""
    session_id: str = ""
    abort_signal: asyncio.Event = field(default_factory=asyncio.Event)
    config: dict = field(default_factory=dict)
    working_dir: str = "."


@runtime_checkable
class Tool(Protocol):
    """统一工具协议。内置工具和 MCP 工具均需满足此接口。"""

    name: str
    description: str
    input_schema: dict  # JSON Schema

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult: ...

    def is_readonly(self, args: dict) -> bool:
        """是否只读操作（可并发执行）。"""
        return False

    def is_concurrency_safe(self, args: dict) -> bool:
        """是否并发安全（可与其他工具同时执行）。"""
        return self.is_readonly(args)

    def is_destructive(self, args: dict) -> bool:
        """是否破坏性操作（需额外确认）。"""
        return False

    def is_enabled(self) -> bool:
        """当前是否可用。"""
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        """工具自身的权限检查。默认允许。"""
        return PermissionDecision(tier=PermissionTier.ALLOW)

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        """输入校验。返回 (是否合法, 错误信息)。"""
        return True, None
