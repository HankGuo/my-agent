"""精确编辑工具：search-and-replace 式文件修改。"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.tools.file_patch")

_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "文件路径",
        },
        "old_str": {
            "type": "string",
            "description": "要匹配的原始文本（必须在文件中唯一出现一次）",
        },
        "new_str": {
            "type": "string",
            "description": "替换后的新文本",
        },
        "dry_run": {
            "type": "boolean",
            "description": "仅预览 diff，不实际写入，默认 false",
            "default": False,
        },
    },
    "required": ["path", "old_str", "new_str"],
}


class FilePatchTool:
    """精确 search-and-replace 编辑，要求 old_str 唯一匹配。"""

    name = "Patch"
    description = "精确替换文件中的指定文本片段（old_str → new_str），要求唯一匹配"
    input_schema = _PATCH_SCHEMA

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        path = Path(args["path"]).expanduser()
        old_str: str = args["old_str"]
        new_str: str = args["new_str"]
        dry_run: bool = args.get("dry_run", False)

        # 读取文件
        if not path.exists():
            return ToolResult(output=f"文件不存在: {path}", is_error=True)
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(output=f"读取失败: {e}", is_error=True)

        # 计算匹配次数
        count = content.count(old_str)

        if count == 0:
            # 提供上下文帮助定位
            lines = content.splitlines()
            # 尝试模糊查找（取 old_str 第一行做模糊匹配）
            first_line = old_str.splitlines()[0] if old_str.strip() else ""
            context_lines: list[str] = []
            for i, line in enumerate(lines):
                if first_line and first_line.strip() in line:
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    context_lines = [f"{j+1}: {lines[j]}" for j in range(start, end)]
                    break
            hint = "\n".join(context_lines) if context_lines else "(未找到近似匹配)"
            return ToolResult(
                output=f"匹配失败：old_str 在文件中出现 0 次。\n文件: {path}\n近似上下文:\n{hint}",
                is_error=True,
            )

        if count > 1:
            return ToolResult(
                output=f"匹配失败：old_str 在文件中出现 {count} 次，请提供更多上下文使其唯一。",
                is_error=True,
            )

        # 执行替换
        new_content = content.replace(old_str, new_str, 1)

        # 生成 unified diff
        diff = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
        diff_text = "".join(diff)

        if dry_run:
            return ToolResult(output=f"[dry_run] diff 预览:\n{diff_text}")

        # 写回文件
        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(output=f"写入失败: {e}", is_error=True)

        return ToolResult(
            output=f"已修改 {path}\n{diff_text}",
            data={"path": str(path), "replacements": 1},
        )

    def is_readonly(self, args: dict) -> bool:
        return args.get("dry_run", False)

    def is_concurrency_safe(self, args: dict) -> bool:
        return args.get("dry_run", False)

    def is_destructive(self, args: dict) -> bool:
        return not args.get("dry_run", False)

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        if args.get("dry_run", False):
            return PermissionDecision(tier=PermissionTier.ALLOW, reason="dry_run 只读")
        return PermissionDecision(tier=PermissionTier.ASK, reason="文件编辑需要确认")

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        if not args.get("path"):
            return False, "缺少 path 参数"
        if "old_str" not in args:
            return False, "缺少 old_str 参数"
        if "new_str" not in args:
            return False, "缺少 new_str 参数"
        return True, None
