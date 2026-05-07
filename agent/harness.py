"""Harness：工具调用编排、权限检查、上下文管理。"""

from __future__ import annotations

import logging

from agent.loop import ToolExecutor, Message, TurnState
from agent.models import ToolCall
from tools.base import Tool, ToolUseContext, PermissionDecision
from tools.orchestration import execute_tool_calls
from tools.permission import PermissionEngine
from tools.registry import ToolRegistry

logger = logging.getLogger("my-agent.harness")


class Harness:
    """
    Agent Harness：连接核心循环和工具系统。

    职责：
    - 工具调用编排（分区并发）
    - 权限检查
    - 上下文构建
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_engine: PermissionEngine,
        *,
        max_concurrency: int = 10,
        session_id: str = "default",
        working_dir: str = ".",
    ):
        self._registry = registry
        self._permission = permission_engine
        self._max_concurrency = max_concurrency
        self._session_id = session_id
        self._working_dir = working_dir

    async def execute_tools(
        self,
        tool_calls: list[ToolCall],
        context: ToolUseContext,
    ) -> list[dict]:
        """执行工具调用列表，返回 tool result 消息。"""
        tools = self._registry.tool_map()
        return await execute_tool_calls(
            tool_calls=tool_calls,
            tools=tools,
            context=context,
            permission_engine=self._permission,
            max_concurrency=self._max_concurrency,
        )

    def check_permission(
        self, tool_name: str, args: dict, context: ToolUseContext
    ) -> PermissionDecision:
        """检查工具调用权限。"""
        return self._permission.check(tool_name, args, context)

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def permission_engine(self) -> PermissionEngine:
        return self._permission
