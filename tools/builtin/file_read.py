"""文件读取工具。"""

from __future__ import annotations

import logging
from pathlib import Path

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.tools.file_read")

_FILE_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "文件路径（绝对或相对路径）",
        },
        "offset": {
            "type": "integer",
            "description": "起始行号（从 0 开始），默认 0",
            "default": 0,
        },
        "limit": {
            "type": "integer",
            "description": "最大读取行数，默认 2000",
            "default": 2000,
        },
    },
    "required": ["path"],
}


class FileReadTool:
    """读取文件内容。"""

    name = "Read"
    description = "读取文件内容，支持指定行范围"
    input_schema = _FILE_READ_SCHEMA

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        path = Path(args["path"]).expanduser()
        offset = args.get("offset", 0)
        limit = args.get("limit", 2000)

        try:
            if not path.exists():
                return ToolResult(output=f"文件不存在: {path}", is_error=True)
            if path.is_dir():
                return ToolResult(output=f"路径是目录: {path}", is_error=True)

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            selected = lines[offset : offset + limit]
            numbered = [f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(selected)]
            output = "".join(numbered)

            return ToolResult(
                output=output,
                data={"path": str(path), "total_lines": len(lines), "shown": len(selected)},
            )
        except Exception as e:
            return ToolResult(output=f"读取失败: {e}", is_error=True)

    def is_readonly(self, args: dict) -> bool:
        return True

    def is_concurrency_safe(self, args: dict) -> bool:
        return True

    def is_destructive(self, args: dict) -> bool:
        return False

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        return PermissionDecision(tier=PermissionTier.ALLOW)

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        if not args.get("path"):
            return False, "缺少 path 参数"
        return True, None
