"""SWE-bench 风格测试：真实代码库修改场景。

模拟：
1. 读取项目结构和文件
2. 定位 bug
3. 修改代码
4. 运行测试验证
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
def swe_harness(temp_dir):
    registry = ToolRegistry()
    registry.register_many([BashTool(), FileReadTool(), FileWriteTool()])
    perm = PermissionEngine(allow_rules=["Read(*)", "Bash(*)", "Write(*)"], mode="auto")
    return Harness(registry=registry, permission_engine=perm, working_dir=str(temp_dir))


# ── 场景 1：简单函数 bug 修复 ──────────────────────────────


@pytest.mark.asyncio
async def test_swe_fix_function_bug(swe_harness, temp_dir):
    """修复单个函数中的逻辑错误。"""
    (temp_dir / "calculator.py").write_text('''
def add(a, b):
    return a - b  # bug: 应该是 +

def sub(a, b):
    return a - b
''')
    (temp_dir / "test_calculator.py").write_text('''
from calculator import add, sub
assert add(1, 2) == 3
assert sub(5, 3) == 2
print("PASS")
''')

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": str(temp_dir / "calculator.py")})]),
        make_tool_events([ToolCall(id="tc2", name="Read", arguments={"path": str(temp_dir / "test_calculator.py")})]),
        make_tool_events([ToolCall(id="tc3", name="Write", arguments={
            "path": str(temp_dir / "calculator.py"),
            "content": '''
def add(a, b):
    return a + b

def sub(a, b):
    return a - b
''',
        })]),
        make_tool_events([ToolCall(id="tc4", name="Bash", arguments={
            "command": f"cd {temp_dir} && python3 test_calculator.py",
        })]),
        make_text_events("Bug 修复完成，测试通过"),
    ]))

    result = await react_loop(
        prompt="修复 calculator.py 中的 bug",
        model=model,
        harness=swe_harness,
        tools=swe_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"


# ── 场景 2：添加缺失的异常处理 ─────────────────────────────


@pytest.mark.asyncio
async def test_swe_add_exception_handling(swe_harness, temp_dir):
    """给函数添加缺失的异常处理。"""
    (temp_dir / "parser.py").write_text('''
def parse_int(s):
    return int(s)  # 可能 ValueError
''')
    (temp_dir / "test_parser.py").write_text('''
from parser import parse_int
assert parse_int("42") == 42
assert parse_int("abc") is None
print("PASS")
''')

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": str(temp_dir / "parser.py")})]),
        make_tool_events([ToolCall(id="tc2", name="Read", arguments={"path": str(temp_dir / "test_parser.py")})]),
        make_tool_events([ToolCall(id="tc3", name="Write", arguments={
            "path": str(temp_dir / "parser.py"),
            "content": '''
def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None
''',
        })]),
        make_tool_events([ToolCall(id="tc4", name="Bash", arguments={
            "command": f"cd {temp_dir} && python3 test_parser.py",
        })]),
        make_text_events("完成"),
    ]))

    result = await react_loop(
        prompt="给 parse_int 添加异常处理",
        model=model,
        harness=swe_harness,
        tools=swe_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"


# ── 场景 3：重构代码结构 ───────────────────────────────────


@pytest.mark.asyncio
async def test_swe_refactor_module(swe_harness, temp_dir):
    """将大文件拆分为模块。"""
    (temp_dir / "monolith.py").write_text('''
def func_a(): return "a"
def func_b(): return "b"
def func_c(): return "c"
''')

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": str(temp_dir / "monolith.py")})]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(temp_dir / "a.py"),
            "content": "def func_a(): return 'a'\n",
        })]),
        make_tool_events([ToolCall(id="tc3", name="Write", arguments={
            "path": str(temp_dir / "b.py"),
            "content": "def func_b(): return 'b'\n",
        })]),
        make_tool_events([ToolCall(id="tc4", name="Write", arguments={
            "path": str(temp_dir / "c.py"),
            "content": "def func_c(): return 'c'\n",
        })]),
        make_tool_events([ToolCall(id="tc5", name="Bash", arguments={
            "command": f"cd {temp_dir} && python3 -c 'from a import func_a; print(func_a())'",
        })]),
        make_text_events("重构完成"),
    ]))

    result = await react_loop(
        prompt="把 monolith.py 拆分成 a.py b.py c.py",
        model=model,
        harness=swe_harness,
        tools=swe_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"
    assert (temp_dir / "a.py").exists()
    assert (temp_dir / "b.py").exists()
    assert (temp_dir / "c.py").exists()


# ── 场景 4：修改配置文件 ───────────────────────────────────


@pytest.mark.asyncio
async def test_swe_update_config(swe_harness, temp_dir):
    """修改 YAML/JSON 配置文件。"""
    (temp_dir / "config.yaml").write_text('''
debug: false
port: 8080
''')

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={"path": str(temp_dir / "config.yaml")})]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(temp_dir / "config.yaml"),
            "content": '''
debug: true
port: 3000
''',
        })]),
        make_text_events("配置更新完成"),
    ]))

    result = await react_loop(
        prompt="把端口改成 3000，debug 改成 true",
        model=model,
        harness=swe_harness,
        tools=swe_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    content = (temp_dir / "config.yaml").read_text()
    assert "port: 3000" in content
    assert "debug: true" in content


# ── 场景 5：多级目录导航 ───────────────────────────────────


@pytest.mark.asyncio
async def test_swe_deep_directory_navigation(swe_harness, temp_dir):
    """在深层目录结构中查找和修改文件。"""
    deep = temp_dir / "src" / "app" / "models"
    deep.mkdir(parents=True)
    (deep / "user.py").write_text("class User:\n    pass\n")

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": f"find {temp_dir} -name '*.py'",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Read", arguments={"path": str(deep / "user.py")})]),
        make_tool_events([ToolCall(id="tc3", name="Write", arguments={
            "path": str(deep / "user.py"),
            "content": "class User:\n    def __init__(self, name):\n        self.name = name\n",
        })]),
        make_text_events("完成"),
    ]))

    result = await react_loop(
        prompt="找到 user.py 并给 User 添加 name 属性",
        model=model,
        harness=swe_harness,
        tools=swe_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"
    assert "self.name = name" in (deep / "user.py").read_text()


# ── 场景 6：Git 操作链 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_swe_git_workflow(swe_harness, temp_dir):
    """Git 初始化 → 提交 → 查看状态。"""
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Bash", arguments={
            "command": f"cd {temp_dir} && git init",
        })]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(temp_dir / "main.py"),
            "content": "print('hello')\n",
        })]),
        make_tool_events([ToolCall(id="tc3", name="Bash", arguments={
            "command": f"cd {temp_dir} && git add . && git commit -m 'init'",
        })]),
        make_tool_events([ToolCall(id="tc4", name="Bash", arguments={
            "command": f"cd {temp_dir} && git log --oneline",
        })]),
        make_text_events("Git 流程完成"),
    ]))

    result = await react_loop(
        prompt="初始化 git 仓库，提交 main.py",
        model=model,
        harness=swe_harness,
        tools=swe_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"
    assert (temp_dir / ".git").is_dir()
