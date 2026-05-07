"""自举编排层：原子化 edit → test → commit/rollback 操作。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tools.base import ToolUseContext
from tools.builtin.file_patch import FilePatchTool
from tools.builtin.git_ops import GitOpsTool
from tools.builtin.test_runner import TestRunnerTool

logger = logging.getLogger("my-agent.bootstrap")


@dataclass
class BootstrapResult:
    """自举操作结果。"""
    success: bool
    message: str
    diff: str = ""
    test_output: str = ""
    commit_hash: str = ""


async def bootstrap_edit(
    file_path: str,
    old_str: str,
    new_str: str,
    commit_message: str = "",
    *,
    context: ToolUseContext | None = None,
    test_target: str = "",
) -> BootstrapResult:
    """
    原子自举操作：
    1. git stash（保护现场）
    2. Patch 执行编辑
    3. RunTests 验证
    4. 通过 → git commit
    5. 失败 → git checkout_file（回滚单文件）+ git stash pop
    """
    ctx = context or ToolUseContext()
    patch = FilePatchTool()
    git = GitOpsTool()
    runner = TestRunnerTool()

    # Step 1: stash 保护现场
    stash_result = await git.call({"action": "stash"}, ctx)
    has_stash = not stash_result.is_error and "No local changes" not in stash_result.output

    # Step 2: 执行编辑
    patch_result = await patch.call(
        {"path": file_path, "old_str": old_str, "new_str": new_str},
        ctx,
    )
    if patch_result.is_error:
        # 编辑失败，恢复 stash
        if has_stash:
            await git.call({"action": "stash_pop"}, ctx)
        return BootstrapResult(
            success=False,
            message=f"编辑失败: {patch_result.output}",
        )

    diff_text = patch_result.output

    # Step 3: 运行测试
    test_result = await runner.call({"target": test_target, "timeout": 60}, ctx)
    test_output = test_result.output

    if test_result.is_error:
        # 测试失败，回滚编辑
        logger.warning("测试失败，回滚编辑: %s", file_path)
        await git.call({"action": "checkout_file", "file_path": file_path}, ctx)
        if has_stash:
            await git.call({"action": "stash_pop"}, ctx)
        return BootstrapResult(
            success=False,
            message="测试失败，已回滚",
            diff=diff_text,
            test_output=test_output,
        )

    # Step 4: 测试通过，commit
    msg = commit_message or f"bootstrap: edit {file_path}"
    commit_result = await git.call({"action": "commit", "message": msg}, ctx)

    # 提取 commit hash（从 git commit 输出中）
    commit_hash = ""
    if not commit_result.is_error:
        # 输出格式通常是 "[branch hash] message"
        parts = commit_result.output.split()
        for part in parts:
            if len(part) >= 7 and part.replace("]", "").isalnum():
                commit_hash = part.replace("]", "")
                break

    # 恢复 stash（如果有其他暂存的改动）
    if has_stash:
        await git.call({"action": "stash_pop"}, ctx)

    return BootstrapResult(
        success=True,
        message="编辑成功，测试通过，已提交",
        diff=diff_text,
        test_output=test_output,
        commit_hash=commit_hash,
    )
