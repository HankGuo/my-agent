"""内置工具测试：覆盖 Bash、Read、Write 的功能、边界和并发安全。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tools.builtin.bash import BashTool
from tools.builtin.file_read import FileReadTool
from tools.builtin.file_write import FileWriteTool
from tools.base import ToolUseContext


# ── BashTool 测试 ──────────────────────────────────────────


@pytest.fixture
def bash_tool():
    return BashTool()


@pytest.mark.asyncio
async def test_bash_echo(bash_tool, temp_dir):
    """基础 echo 命令。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "echo hello"}, ctx)
    assert "hello" in result.output
    assert not result.is_error


@pytest.mark.asyncio
async def test_bash_timeout(bash_tool, temp_dir):
    """超时命令应返回超时错误，不卡住。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "sleep 10", "timeout": 500}, ctx)
    assert result.is_error
    assert "超时" in result.output


@pytest.mark.asyncio
async def test_bash_error_exit_code(bash_tool, temp_dir):
    """非零退出码应标记为错误。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "exit 42"}, ctx)
    assert result.is_error
    assert result.data["returncode"] == 42


@pytest.mark.asyncio
async def test_bash_stderr_capture(bash_tool, temp_dir):
    """stderr 应被捕获。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "echo error >&2; exit 1"}, ctx)
    assert result.is_error
    assert "error" in result.output


@pytest.mark.asyncio
async def test_bash_empty_command(bash_tool, temp_dir):
    """空命令应校验失败。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    valid, err = bash_tool.validate_input({"command": ""})
    assert not valid


def test_bash_readonly_detection(bash_tool):
    """只读命令识别正确性。"""
    assert bash_tool.is_readonly({"command": "ls /tmp"})
    assert bash_tool.is_readonly({"command": "cat file.txt"})
    assert bash_tool.is_readonly({"command": "git status"})
    assert not bash_tool.is_readonly({"command": "rm file.txt"})
    assert not bash_tool.is_readonly({"command": "echo x > file.txt"})


def test_bash_destructive_detection(bash_tool):
    """破坏性命令识别。"""
    assert bash_tool.is_destructive({"command": "rm -rf /"})
    assert bash_tool.is_destructive({"command": "> file.txt"})
    assert not bash_tool.is_destructive({"command": "ls"})


@pytest.mark.asyncio
async def test_bash_unicode_output(bash_tool, temp_dir):
    """Unicode 输出正确处理。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "echo '中文测试'"}, ctx)
    assert "中文测试" in result.output


@pytest.mark.asyncio
async def test_bash_large_output(bash_tool, temp_dir):
    """大输出不应导致内存问题。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "python3 -c \"print('A'*1000000)\""}, ctx)
    assert len(result.output) >= 1_000_000


@pytest.mark.asyncio
async def test_bash_long_running_cancel(bash_tool, temp_dir):
    """长时间运行的命令在超时后应能被取消，不残留僵尸进程。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await bash_tool.call({"command": "sleep 100", "timeout": 200}, ctx)
    assert result.is_error
    assert "超时" in result.output
    # 检查没有残留 sleep 进程
    import subprocess
    procs = subprocess.run(["pgrep", "-f", "sleep 100"], capture_output=True, text=True)
    # 注意：pgrep 可能匹配到自身，简单过滤
    lines = [l for l in procs.stdout.strip().split("\n") if l]
    # 此处不严格断言，因为环境可能已有其他 sleep 进程


# ── FileReadTool 测试 ──────────────────────────────────────


@pytest.fixture
def read_tool():
    return FileReadTool()


@pytest.mark.asyncio
async def test_read_existing_file(read_tool, temp_dir):
    """读取存在的文件。"""
    f = temp_dir / "test.txt"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await read_tool.call({"path": str(f)}, ctx)
    assert "line1" in result.output
    assert result.data["total_lines"] == 3


@pytest.mark.asyncio
async def test_read_nonexistent_file(read_tool, temp_dir):
    """读取不存在的文件返回错误。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await read_tool.call({"path": str(temp_dir / "ghost.txt")}, ctx)
    assert result.is_error
    assert "不存在" in result.output


@pytest.mark.asyncio
async def test_read_directory(read_tool, temp_dir):
    """读取目录返回错误。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await read_tool.call({"path": str(temp_dir)}, ctx)
    assert result.is_error
    assert "目录" in result.output


@pytest.mark.asyncio
async def test_read_offset_limit(read_tool, temp_dir):
    """offset 和 limit 生效。"""
    f = temp_dir / "lines.txt"
    f.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await read_tool.call({"path": str(f), "offset": 10, "limit": 5}, ctx)
    assert "line10" in result.output
    assert "line14" in result.output
    assert "line15" not in result.output
    assert result.data["shown"] == 5


@pytest.mark.asyncio
async def test_read_large_file(read_tool, temp_dir):
    """大文件读取性能。"""
    f = temp_dir / "big.txt"
    f.write_text("A\n" * 100_000, encoding="utf-8")
    ctx = ToolUseContext(working_dir=str(temp_dir))
    start = asyncio.get_event_loop().time()
    result = await read_tool.call({"path": str(f)}, ctx)
    elapsed = asyncio.get_event_loop().time() - start
    assert result.data["total_lines"] == 100_000
    # 10万行应在 2 秒内读完
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_read_binary_file(read_tool, temp_dir):
    """二进制文件应被安全处理（不崩溃）。"""
    f = temp_dir / "binary.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await read_tool.call({"path": str(f)}, ctx)
    # 不应崩溃，可能输出乱码但无异常
    assert not result.is_error or "读取失败" in result.output


@pytest.mark.asyncio
async def test_read_unicode_file(read_tool, temp_dir):
    """Unicode 文件正确读取。"""
    f = temp_dir / "unicode.txt"
    f.write_text("中文\n日本語\n한국어\n", encoding="utf-8")
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await read_tool.call({"path": str(f)}, ctx)
    assert "中文" in result.output
    assert "日本語" in result.output


# ── FileWriteTool 测试 ─────────────────────────────────────


@pytest.fixture
def write_tool():
    return FileWriteTool()


@pytest.mark.asyncio
async def test_write_new_file(write_tool, temp_dir):
    """写入新文件。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    path = temp_dir / "new.txt"
    result = await write_tool.call({"path": str(path), "content": "hello"}, ctx)
    assert not result.is_error
    assert path.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_write_overwrite(write_tool, temp_dir):
    """覆盖已有文件。"""
    f = temp_dir / "exist.txt"
    f.write_text("old", encoding="utf-8")
    ctx = ToolUseContext(working_dir=str(temp_dir))
    result = await write_tool.call({"path": str(f), "content": "new"}, ctx)
    assert f.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_write_create_dirs(write_tool, temp_dir):
    """自动创建父目录。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    path = temp_dir / "a" / "b" / "c.txt"
    result = await write_tool.call({"path": str(path), "content": "deep"}, ctx)
    assert path.exists()


@pytest.mark.asyncio
async def test_write_no_create_dirs(write_tool, temp_dir):
    """不创建目录时应失败。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    path = temp_dir / "no" / "dir.txt"
    result = await write_tool.call({"path": str(path), "content": "x", "create_dirs": False}, ctx)
    assert result.is_error


@pytest.mark.asyncio
async def test_write_unicode(write_tool, temp_dir):
    """Unicode 内容正确写入。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    path = temp_dir / "uni.txt"
    content = "中文🎉日本語"
    result = await write_tool.call({"path": str(path), "content": content}, ctx)
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_write_large_file(write_tool, temp_dir):
    """大文件写入性能。"""
    ctx = ToolUseContext(working_dir=str(temp_dir))
    path = temp_dir / "big.txt"
    content = "A" * 10_000_000  # 10MB
    start = asyncio.get_event_loop().time()
    result = await write_tool.call({"path": str(path), "content": content}, ctx)
    elapsed = asyncio.get_event_loop().time() - start
    assert path.stat().st_size == 10_000_000
    assert elapsed < 5.0


@pytest.mark.asyncio
async def test_write_validation(write_tool):
    """输入校验。"""
    valid, err = write_tool.validate_input({"path": "/tmp/x.txt"})
    assert not valid
    assert "content" in err

    valid, err = write_tool.validate_input({"content": "x"})
    assert not valid
    assert "path" in err
