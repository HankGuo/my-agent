"""版本控制工具：安全的 Git 操作子集。"""

from __future__ import annotations

import asyncio
import logging
import time

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.tools.git_ops")

_GIT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "Git 操作",
            "enum": [
                "status", "diff", "diff_stat", "commit",
                "stash", "stash_pop", "checkout_file", "log",
            ],
        },
        "message": {
            "type": "string",
            "description": "commit 消息（action=commit 时必填）",
        },
        "file_path": {
            "type": "string",
            "description": "文件路径（action=diff/checkout_file 时可选）",
        },
    },
    "required": ["action"],
}

# 硬拒绝的操作关键词
_BLOCKED_ACTIONS = {"push", "force_push", "reset_hard"}


class GitOpsTool:
    """安全的 Git 操作。硬拒绝 push / force push / reset --hard。"""

    name = "Git"
    description = "执行安全的 Git 操作（status/diff/commit/stash 等），禁止远程推送"
    input_schema = _GIT_SCHEMA

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        action = args.get("action", "")
        message = args.get("message", "")
        file_path = args.get("file_path", "")

        # 硬拒绝危险操作
        if action in _BLOCKED_ACTIONS:
            return ToolResult(
                output=f"拒绝执行: 自举模式不允许远程操作（{action}）",
                is_error=True,
            )

        # 构造命令
        cmd = self._build_command(action, message, file_path)
        if cmd is None:
            return ToolResult(output=f"不支持的 action: {action}", is_error=True)

        # 参数校验
        if action == "commit" and not message:
            return ToolResult(output="commit 需要提供 message 参数", is_error=True)

        return await self._run_git(cmd, context)

    def _build_command(self, action: str, message: str, file_path: str) -> str | None:
        """根据 action 构造 git 命令。"""
        match action:
            case "status":
                return "git status --short"
            case "diff":
                if file_path:
                    return f"git diff -- {file_path}"
                return "git diff"
            case "diff_stat":
                return "git diff --stat"
            case "commit":
                # 使用 -- 防止消息被解析为参数
                safe_msg = message.replace('"', '\\"')
                return f'git add -A && git commit -m "{safe_msg}"'
            case "stash":
                ts = int(time.time())
                return f'git stash push -m "bootstrap-{ts}"'
            case "stash_pop":
                return "git stash pop"
            case "checkout_file":
                if not file_path:
                    return None
                return f"git checkout -- {file_path}"
            case "log":
                return "git log --oneline -10"
            case _:
                return None

    async def _run_git(self, cmd: str, context: ToolUseContext) -> ToolResult:
        """执行 git 命令并返回结果。"""
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                return ToolResult(
                    output=f"{output}\n{error}".strip(),
                    is_error=True,
                    data={"returncode": proc.returncode},
                )
            return ToolResult(output=output.strip() or "(无输出)")
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    pass
            return ToolResult(output="Git 命令超时（30s）", is_error=True)
        except Exception as e:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    pass
            return ToolResult(output=f"执行失败: {e}", is_error=True)

    def is_readonly(self, args: dict) -> bool:
        action = args.get("action", "")
        return action in ("status", "diff", "diff_stat", "log")

    def is_concurrency_safe(self, args: dict) -> bool:
        return self.is_readonly(args)

    def is_destructive(self, args: dict) -> bool:
        action = args.get("action", "")
        return action in ("checkout_file", "stash_pop")

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        return PermissionDecision(tier=PermissionTier.ALLOW, reason="Git 操作默认允许")

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        action = args.get("action")
        if not action:
            return False, "缺少 action 参数"
        valid_actions = {"status", "diff", "diff_stat", "commit", "stash", "stash_pop", "checkout_file", "log"}
        if action not in valid_actions:
            return False, f"不支持的 action: {action}，可选: {', '.join(sorted(valid_actions))}"
        if action == "checkout_file" and not args.get("file_path"):
            return False, "checkout_file 需要提供 file_path 参数"
        return True, None
