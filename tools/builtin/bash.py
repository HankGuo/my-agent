"""Bash 工具：执行 Shell 命令。"""

from __future__ import annotations

import asyncio
import logging

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.tools.bash")

_BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "要执行的 Shell 命令",
        },
        "timeout": {
            "type": "integer",
            "description": "超时时间（毫秒），默认 30000",
            "default": 30000,
        },
    },
    "required": ["command"],
}


class BashTool:
    """执行 Shell 命令。"""

    name = "Bash"
    description = "在 Shell 中执行命令并返回输出"
    input_schema = _BASH_SCHEMA

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        command = args.get("command", "")
        timeout_ms = args.get("timeout", 30000)
        timeout_sec = min(timeout_ms / 1000, 600)  # 最大 10 分钟

        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return ToolResult(
                    output=f"{output}\n[stderr]\n{error}".strip(),
                    is_error=True,
                    data={"returncode": proc.returncode},
                )
            return ToolResult(output=output or "(无输出)")
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    pass
            return ToolResult(output=f"命令超时（{timeout_sec}s）", is_error=True)
        except Exception as e:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    pass
            return ToolResult(output=f"执行失败: {e}", is_error=True)

    def is_readonly(self, args: dict) -> bool:
        # 简单判断：以 ls/find/cat/head/tail/grep/which/echo/printenv 开头的为只读
        # 但包含重定向符号（> 或 >>）的命令不是只读
        cmd = args.get("command", "").strip()
        if ">" in cmd or "|" in cmd:
            return False
        readonly_prefixes = ("ls ", "find ", "cat ", "head ", "tail ", "grep ", "which ", "echo ", "printenv ", "git status", "git log", "git diff", "git branch")
        return any(cmd.startswith(p) for p in readonly_prefixes)

    def is_concurrency_safe(self, args: dict) -> bool:
        return self.is_readonly(args)

    def is_destructive(self, args: dict) -> bool:
        cmd = args.get("command", "").strip()
        destructive = ("rm ", "rm -rf", "mkfs", "dd ", "> ", "truncate")
        return any(cmd.startswith(p) for p in destructive)

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        if self.is_destructive(args):
            return PermissionDecision(tier=PermissionTier.ASK, reason="破坏性命令需要确认")
        return PermissionDecision(tier=PermissionTier.ALLOW)

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        if not args.get("command"):
            return False, "缺少 command 参数"
        return True, None
