"""AgentBench 风格测试：操作系统交互、文件操作、任务完成能力。

覆盖：文件系统操作、目录管理、批量处理、环境感知。
"""

from __future__ import annotations

import pytest

from agent.loop import react_loop, DefaultContextBuilder
from agent.models import ToolCall
from agent.harness import Harness
from tools.registry import ToolRegistry
from tools.permission import PermissionEngine
from tools.builtin.bash import BashTool
from tools.builtin.file_read import FileReadTool
from tools.builtin.file_write import FileWriteTool
from tests.conftest import MockModelProvider, MockModelConfig, make_tool_events, make_text_events


@pytest.fixture
def agentbench_harness(temp_dir):
    registry = ToolRegistry()
    registry.register_many([BashTool(), FileReadTool(), FileWriteTool()])
    perm = PermissionEngine(allow_rules=["Read(*)", "Bash(*)", "Write(*)"], mode="auto")
    return Harness(registry=registry, permission_engine=perm, working_dir=str(temp_dir))


# ── OS 交互：批量文件创建 ──────────────────────────────────


@pytest.mark.asyncio
async def test_agentbench_batch_create_files(agentbench_harness, temp_dir):
    """批量创建 10 个文件。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": f"cd {temp_dir} && for i in $(seq 1 10); do echo $i > file_$i.txt; done",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"ls {temp_dir}/file_*.txt | wc -l",
        })]),
        make_text_events("创建了 10 个文件"),
    ]))

    result = await react_loop(
        prompt="创建 10 个文件 file_1.txt 到 file_10.txt",
        model=model,
        harness=agentbench_harness,
        tools=agentbench_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    assert (temp_dir / "file_10.txt").exists()


# ── OS 交互：查找替换 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_agentbench_find_and_replace(agentbench_harness, temp_dir):
    """在多个文件中查找并替换文本。"""
    for i in range(3):
        (temp_dir / f"doc{i}.txt").write_text("old brand name\n")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": f"cd {temp_dir} && grep -rl 'old brand' .",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"cd {temp_dir} && sed -i '' 's/old brand/new brand/g' doc*.txt",
        })]),
        make_tool_events([ToolCall(id="tc3", name="Bash", arguments={
            "command": f"cd {temp_dir} && grep -l 'new brand' doc*.txt | wc -l",
        })]),
        make_text_events("替换完成"),
    ]))

    result = await react_loop(
        prompt="把所有文件中的 'old brand' 替换成 'new brand'",
        model=model,
        harness=agentbench_harness,
        tools=agentbench_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    for i in range(3):
        assert "new brand" in (temp_dir / f"doc{i}.txt").read_text()


# ── OS 交互：数据统计 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_agentbench_data_aggregation(agentbench_harness, temp_dir):
    """统计多个文件中的数据。"""
    (temp_dir / "sales.csv").write_text("product,amount\nA,100\nB,200\nA,150\n")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": str(temp_dir / "sales.csv")})]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"cd {temp_dir} && awk -F, '{{sum+=$2}} END {{print sum}}' sales.csv",
        })]),
        make_text_events("总销售额是 450"),
    ]))

    result = await react_loop(
        prompt="计算 sales.csv 中 amount 的总和",
        model=model,
        harness=agentbench_harness,
        tools=agentbench_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"


# ── OS 交互：压缩解压 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_agentbench_archive_operations(agentbench_harness, temp_dir):
    """创建压缩包并验证内容。"""
    (temp_dir / "data.txt").write_text("important data")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": f"cd {temp_dir} && tar czf archive.tar.gz data.txt",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"cd {temp_dir} && tar tzf archive.tar.gz",
        })]),
        make_text_events("压缩完成"),
    ]))

    result = await react_loop(
        prompt="把 data.txt 压缩成 archive.tar.gz",
        model=model,
        harness=agentbench_harness,
        tools=agentbench_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    assert (temp_dir / "archive.tar.gz").exists()


# ── OS 交互：环境检查 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_agentbench_environment_check(agentbench_harness, temp_dir):
    """检查 Python 版本和已安装包。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": "python3 --version",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": "pip list | grep pytest",
        })]),
        make_text_events("环境检查完成"),
    ]))

    result = await react_loop(
        prompt="检查 Python 版本和 pytest 是否安装",
        model=model,
        harness=agentbench_harness,
        tools=agentbench_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"


# ── OS 交互：权限相关 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_agentbench_file_permissions(agentbench_harness, temp_dir):
    """修改文件权限。"""
    script = temp_dir / "run.sh"
    script.write_text("#!/bin/bash\necho hello\n")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": f"chmod +x {script}",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"ls -l {script}",
        })]),
        make_text_events("已添加执行权限"),
    ]))

    result = await react_loop(
        prompt="给 run.sh 添加执行权限",
        model=model,
        harness=agentbench_harness,
        tools=agentbench_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    import stat
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR
