"""文件写入工具。"""

from __future__ import annotations

import logging
from pathlib import Path

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.tools.file_write")

_FILE_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "文件路径",
        },
        "content": {
            "type": "string",
            "description": "写入内容",
        },
        "create_dirs": {
            "type": "boolean",
            "description": "是否自动创建父目录，默认 true",
            "default": True,
        },
    },
    "required": ["path", "content"],
}


class FileWriteTool:
    """写入文件内容。"""

    name = "Write"
    description = "写入文件内容，如文件已存在则覆盖"
    input_schema = _FILE_WRITE_SCHEMA

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        path = Path(args["path"]).expanduser()
        content = args.get("content", "")
        create_dirs = args.get("create_dirs", True)

        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                output=f"已写入 {path}（{len(content)} 字符）",
                data={"path": str(path), "bytes_written": len(content.encode("utf-8"))},
            )
        except Exception as e:
            return ToolResult(output=f"写入失败: {e}", is_error=True)

    def is_readonly(self, args: dict) -> bool:
        return False

    def is_concurrency_safe(self, args: dict) -> bool:
        return False

    def is_destructive(self, args: dict) -> bool:
        return True  # 覆盖写入视为破坏性

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        return PermissionDecision(tier=PermissionTier.ASK, reason="写入文件需要确认")

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        if not args.get("path"):
            return False, "缺少 path 参数"
        if "content" not in args:
            return False, "缺少 content 参数"
        return True, None
