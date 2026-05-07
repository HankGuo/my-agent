"""结构化测试执行工具：运行 pytest 并解析结果。"""

from __future__ import annotations

import asyncio
import logging
import re

from tools.base import Tool, ToolResult, ToolUseContext, PermissionDecision, PermissionTier

logger = logging.getLogger("my-agent.tools.test_runner")

_TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "description": "测试目标（文件/目录/模块），为空则运行全部测试",
            "default": "",
        },
        "timeout": {
            "type": "integer",
            "description": "超时时间（秒），默认 60",
            "default": 60,
        },
    },
    "required": [],
}

# 匹配 pytest 摘要行，如 "5 passed, 2 failed, 1 error in 1.23s"
_SUMMARY_RE = re.compile(
    r"(?:(\d+) passed)?"
    r"(?:,?\s*(\d+) failed)?"
    r"(?:,?\s*(\d+) error)?"
    r"(?:,?\s*(\d+) warning)?"
    r"(?:,?\s*(\d+) skipped)?"
)


def _parse_pytest_summary(output: str) -> dict:
    """从 pytest 输出中解析测试结果摘要。"""
    lines = output.strip().splitlines()
    # pytest 摘要通常在最后几行
    for line in reversed(lines[-5:]):
        # 寻找包含 "passed" 或 "failed" 或 "error" 的行
        if "passed" in line or "failed" in line or "error" in line:
            passed = failed = errors = warnings = skipped = 0
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                failed = int(m.group(1))
            m = re.search(r"(\d+) error", line)
            if m:
                errors = int(m.group(1))
            m = re.search(r"(\d+) warning", line)
            if m:
                warnings = int(m.group(1))
            m = re.search(r"(\d+) skipped", line)
            if m:
                skipped = int(m.group(1))
            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "warnings": warnings,
                "skipped": skipped,
                "success": failed == 0 and errors == 0,
            }
    return {"passed": 0, "failed": 0, "errors": 0, "warnings": 0, "skipped": 0, "success": False}


class TestRunnerTool:
    """运行 pytest 并返回结构化结果。"""

    name = "RunTests"
    description = "执行 pytest 测试并返回结构化结果摘要"
    input_schema = _TEST_SCHEMA

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        target = args.get("target", "")
        timeout = min(args.get("timeout", 60), 300)  # 最大 5 分钟

        cmd = f"python -m pytest {target} --tb=short -q".strip()

        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")
            full_output = f"{output}\n{error}".strip()

            # 解析结果
            summary = _parse_pytest_summary(full_output)

            # 构造人类可读摘要
            status = "✅ PASS" if summary["success"] else "❌ FAIL"
            summary_line = (
                f"{status} | passed={summary['passed']} failed={summary['failed']} "
                f"errors={summary['errors']} skipped={summary['skipped']}"
            )

            # 失败时返回完整输出；成功时只返回摘要
            if summary["success"]:
                result_text = f"{summary_line}\n\n{output[-500:]}" if len(output) > 500 else f"{summary_line}\n\n{output}"
            else:
                result_text = f"{summary_line}\n\n{full_output}"

            return ToolResult(
                output=result_text,
                data=summary,
                is_error=not summary["success"],
            )
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    pass
            return ToolResult(output=f"测试超时（{timeout}s）", is_error=True)
        except Exception as e:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    pass
            return ToolResult(output=f"执行失败: {e}", is_error=True)

    def is_readonly(self, args: dict) -> bool:
        return True  # 测试不修改文件

    def is_concurrency_safe(self, args: dict) -> bool:
        return True

    def is_destructive(self, args: dict) -> bool:
        return False

    def is_enabled(self) -> bool:
        return True

    def check_permissions(self, args: dict, context: ToolUseContext) -> PermissionDecision:
        return PermissionDecision(tier=PermissionTier.ALLOW, reason="测试执行默认允许")

    def validate_input(self, args: dict) -> tuple[bool, str | None]:
        return True, None
