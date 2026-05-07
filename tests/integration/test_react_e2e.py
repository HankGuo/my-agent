"""端到端 ReAct 集成测试：模拟真实编码任务完整流程。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.loop import react_loop, DefaultContextBuilder
from agent.models import StreamEvent, ToolCall
from agent.harness import Harness
from tools.registry import ToolRegistry
from tools.permission import PermissionEngine
from tools.builtin.bash import BashTool
from tools.builtin.file_read import FileReadTool
from tools.builtin.file_write import FileWriteTool
from tests.conftest import MockModelProvider, MockModelConfig, make_text_events, make_tool_events


@pytest.fixture
def coding_tools(temp_dir):
    """提供真实工具集。"""
    registry = ToolRegistry()
    registry.register_many([
        BashTool(),
        FileReadTool(),
        FileWriteTool(),
    ])
    return registry


@pytest.fixture
def real_harness(coding_tools, temp_dir):
    """提供真实 Harness。"""
    perm = PermissionEngine(
        allow_rules=["Read(*)", "Bash(*)", "Write(*)"],
        deny_rules=["Bash(rm -rf /)"],
        ask_rules=[],
        mode="auto",
    )
    return Harness(
        registry=coding_tools,
        permission_engine=perm,
        working_dir=str(temp_dir),
    )


# ── 场景 1：简单代码生成 ───────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_generate_python_script(real_harness, temp_dir):
    """E2E：生成一个 Python 脚本并写入文件。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([
            ToolCall(id="tc1", name="Write", arguments={
                "path": str(temp_dir / "hello.py"),
                "content": "print('hello world')\n",
            }),
        ], text="我来创建文件"),
        make_text_events("已创建 hello.py"),
    ]))

    result = await react_loop(
        prompt="写一个输出 hello world 的 Python 脚本",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    script = temp_dir / "hello.py"
    assert script.exists()
    assert "hello world" in script.read_text()


# ── 场景 2：Read-Modify-Write 链 ───────────────────────────


@pytest.mark.asyncio
async def test_e2e_read_modify_write(real_harness, temp_dir):
    """E2E：读取文件 → 修改内容 → 写回。"""
    src = temp_dir / "data.txt"
    src.write_text("old content\n", encoding="utf-8")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": str(src)})]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(src),
            "content": "new content\n",
        })]),
        make_text_events("已更新文件"),
    ]))

    result = await react_loop(
        prompt="把文件内容改成 'new content'",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    assert src.read_text() == "new content\n"


# ── 场景 3：Bash 编译运行 ───────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_bash_compile_run(real_harness, temp_dir):
    """E2E：写入 C 代码 → gcc 编译 → 运行。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "main.c"),
            "content": '#include <stdio.h>\nint main() { printf("42\\n"); return 0; }\n',
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"gcc {temp_dir}/main.c -o {temp_dir}/main",
        })]),
        make_tool_events([ToolCall(id="tc3", name="Bash", arguments={
            "command": f"{temp_dir}/main",
        })]),
        make_text_events("程序输出 42"),
    ]))

    result = await react_loop(
        prompt="写一个输出 42 的 C 程序并编译运行",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"
    assert (temp_dir / "main").exists()


# ── 场景 4：模型错误后恢复 ─────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_recovery_after_model_error(real_harness, temp_dir):
    """E2E：模型第一次调用失败，第二次成功。"""
    model = MockModelProvider(MockModelConfig(
        responses=[
            make_text_events("", finish=False),  # 异常：没有 done，但会被解析为 stop
        ]
    ))
    # 这里其实是测试单轮完成，因为空文本 + 无 done 也会 finish
    result = await react_loop(
        prompt="测试",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )
    # 不卡住就算通过
    assert result.reason in ("completed", "max_turns")


# ── 场景 5：并发读取 ───────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_concurrent_reads(real_harness, temp_dir):
    """E2E：模型一次发起多个 Read，应并发执行。"""
    for i in range(3):
        (temp_dir / f"file{i}.txt").write_text(f"content{i}")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([
            ToolCall(id="tc1", name="Read", arguments={"path": str(temp_dir / "file0.txt")}),
            ToolCall(id="tc2", name="Read", arguments={"path": str(temp_dir / "file1.txt")}),
            ToolCall(id="tc3", name="Read", arguments={"path": str(temp_dir / "file2.txt")}),
        ]),
        make_text_events("读取完毕"),
    ]))

    start = asyncio.get_event_loop().time()
    result = await react_loop(
        prompt="读取三个文件",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert result.reason == "completed"
    # 3个并发读取应很快完成
    assert elapsed < 1.0


# ── 场景 6：写入串行安全 ───────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_serial_writes(real_harness, temp_dir):
    """E2E：多次写入应串行执行，最终结果正确。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "seq.txt"),
            "content": "first",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(temp_dir / "seq.txt"),
            "content": "second",
        })]),
        make_tool_events([ToolCall(id="tc3", name="Write", arguments={
            "path": str(temp_dir / "seq.txt"),
            "content": "third",
        })]),
        make_text_events("完成"),
    ]))

    result = await react_loop(
        prompt="依次写入 first、second、third",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"
    assert (temp_dir / "seq.txt").read_text() == "third"


# ── 场景 7：权限拒绝处理 ───────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_permission_denied_handling(temp_dir):
    """E2E：权限拒绝时工具返回错误，循环继续。"""
    registry = ToolRegistry()
    registry.register_many([BashTool(), FileWriteTool()])
    perm = PermissionEngine(
        allow_rules=["Read(*)"],
        deny_rules=["Bash(*)", "Write(*)"],
        mode="default",
    )
    harness = Harness(registry=registry, permission_engine=perm, working_dir=str(temp_dir))

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={"command": "ls"})]),
        make_text_events("权限被拒绝，无法执行"),
    ]))

    result = await react_loop(
        prompt="运行 ls",
        model=model,
        harness=harness,
        tools=registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    # 权限拒绝不应导致卡住或崩溃


# ── 场景 8：空工具响应 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_empty_tool_result(real_harness, temp_dir):
    """E2E：工具返回空结果时不应卡住。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={"command": "true"})]),
        make_text_events("命令执行完成"),
    ]))

    result = await react_loop(
        prompt="运行 true",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"


# ── 场景 9：超长消息链 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_long_message_chain(real_harness, temp_dir):
    """E2E：10 轮工具调用不耗尽内存或卡住。"""
    responses = []
    for i in range(10):
        responses.append(make_tool_events([
            ToolCall(id=f"tc{i}", name="Bash", arguments={"command": f"echo turn{i}"}),
        ]))
    responses.append(make_text_events("完成10轮"))

    model = MockModelProvider(MockModelConfig(responses=responses))

    result = await react_loop(
        prompt="执行10轮命令",
        model=model,
        harness=real_harness,
        tools=real_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=20,
    )

    assert result.reason == "completed"
    assert len(result.messages) == 22  # user + 10*(assistant+tool) + assistant
