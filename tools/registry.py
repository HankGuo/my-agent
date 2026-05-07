"""工具注册表：统一管理内置工具和 MCP 工具。"""

from __future__ import annotations

import logging
from typing import Any

from .base import Tool

logger = logging.getLogger("my-agent.registry")


class ToolRegistry:
    """工具注册表。内置工具优先，MCP 工具可热插拔。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具。同名工具会被覆盖。"""
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        """批量注册工具。"""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        """移除工具。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """获取工具。"""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """获取所有已启用的工具，按名称排序。"""
        return sorted(
            [t for t in self._tools.values() if t.is_enabled()],
            key=lambda t: t.name,
        )

    def tool_map(self) -> dict[str, Tool]:
        """获取工具名到 Tool 的映射。"""
        return dict(self._tools)

    def merge_mcp_tools(self, mcp_tools: list[Tool]) -> list[str]:
        """
        合并 MCP 工具到注册表。内置工具优先（同名不覆盖）。

        返回被跳过的工具名列表（与内置冲突）。
        """
        skipped: list[str] = []
        for tool in mcp_tools:
            if tool.name in self._tools:
                skipped.append(tool.name)
                logger.info("MCP 工具 %s 与内置工具冲突，保留内置", tool.name)
            else:
                self._tools[tool.name] = tool
        return skipped
