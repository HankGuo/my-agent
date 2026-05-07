"""HumanEval 风格代码生成测试：函数级编程任务。

覆盖：基础算法、字符串处理、数据结构、边界条件。
不依赖外部 API，用 MockModel 验证 ReAct 循环在编码任务中的工具调用链是否正确。
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
def coding_harness(temp_dir):
    registry = ToolRegistry()
    registry.register_many([BashTool(), FileReadTool(), FileWriteTool()])
    perm = PermissionEngine(allow_rules=["Read(*)", "Bash(*)", "Write(*)"], mode="auto")
    return Harness(registry=registry, permission_engine=perm, working_dir=str(temp_dir))


# ── HumanEval #1: 两数之和 ─────────────────────────────────


@pytest.mark.asyncio
async def test_humaneval_two_sum(coding_harness, temp_dir):
    """生成 two_sum 函数并验证。"""
    code = '''def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []

if __name__ == "__main__":
    assert two_sum([2,7,11,15], 9) == [0,1]
    assert two_sum([3,2,4], 6) == [1,2]
    print("PASS")
'''
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "two_sum.py"),
            "content": code,
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"python3 {temp_dir}/two_sum.py",
        })]),
        make_text_events("测试通过"),
    ]))

    result = await react_loop(
        prompt="写 two_sum 函数并通过测试",
        model=model,
        harness=coding_harness,
        tools=coding_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
    assert "PASS" in result.messages[-2].content  # tool result


# ── HumanEval #2: 回文判断 ─────────────────────────────────


@pytest.mark.asyncio
async def test_humaneval_palindrome(coding_harness, temp_dir):
    """生成回文判断函数。"""
    code = '''def is_palindrome(s: str) -> bool:
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

if __name__ == "__main__":
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert not is_palindrome("race a car")
    print("PASS")
'''
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "palindrome.py"),
            "content": code,
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"python3 {temp_dir}/palindrome.py",
        })]),
        make_text_events("PASS"),
    ]))

    result = await react_loop(
        prompt="写回文判断函数",
        model=model,
        harness=coding_harness,
        tools=coding_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"


# ── HumanEval #3: FizzBuzz ─────────────────────────────────


@pytest.mark.asyncio
async def test_humaneval_fizzbuzz(coding_harness, temp_dir):
    """生成 FizzBuzz 函数。"""
    code = '''def fizzbuzz(n: int):
    result = []
    for i in range(1, n+1):
        if i % 15 == 0: result.append("FizzBuzz")
        elif i % 3 == 0: result.append("Fizz")
        elif i % 5 == 0: result.append("Buzz")
        else: result.append(str(i))
    return result

if __name__ == "__main__":
    assert fizzbuzz(5) == ["1","2","Fizz","4","Buzz"]
    print("PASS")
'''
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "fizzbuzz.py"),
            "content": code,
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"python3 {temp_dir}/fizzbuzz.py",
        })]),
        make_text_events("PASS"),
    ]))

    result = await react_loop(
        prompt="写 FizzBuzz 函数",
        model=model,
        harness=coding_harness,
        tools=coding_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"


# ── HumanEval #4: 文件读写链 ───────────────────────────────


@pytest.mark.asyncio
async def test_humaneval_file_chain(coding_harness, temp_dir):
    """多文件读写链：生成模块 + 测试 + 运行。"""
    lib = '''def add(a, b):
    return a + b
'''
    test = '''from lib import add
assert add(1, 2) == 3
assert add(-1, 1) == 0
print("PASS")
'''
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "lib.py"),
            "content": lib,
        })]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(temp_dir / "test_lib.py"),
            "content": test,
        })]),
        make_tool_events([ToolCall(id="tc3", name="Bash", arguments={
            "command": f"cd {temp_dir} && python3 test_lib.py",
        })]),
        make_text_events("测试通过"),
    ]))

    result = await react_loop(
        prompt="写 add 函数和测试并运行",
        model=model,
        harness=coding_harness,
        tools=coding_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"


# ── HumanEval #5: 错误修复 ─────────────────────────────────


@pytest.mark.asyncio
async def test_humaneval_bugfix(coding_harness, temp_dir):
    """SWE-bench 风格：读取有 bug 的文件，修复，验证。"""
    buggy = '''def divide(a, b):
    return a / b  # 没处理 b=0

if __name__ == "__main__":
    print(divide(10, 0))
'''
    fixed = '''def divide(a, b):
    if b == 0:
        return 0
    return a / b

if __name__ == "__main__":
    assert divide(10, 2) == 5.0
    assert divide(10, 0) == 0
    print("PASS")
'''
    (temp_dir / "math.py").write_text(buggy)

    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Read", arguments={
            "path": str(temp_dir / "math.py"),
        })]),
        make_tool_events([ToolCall(id="tc2", name="Write", arguments={
            "path": str(temp_dir / "math.py"),
            "content": fixed,
        })]),
        make_tool_events([ToolCall(id="tc3", name="Bash", arguments={
            "command": f"python3 {temp_dir}/math.py",
        })]),
        make_text_events("修复完成"),
    ]))

    result = await react_loop(
        prompt="修复 math.py 中 divide 的除零 bug",
        model=model,
        harness=coding_harness,
        tools=coding_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=10,
    )

    assert result.reason == "completed"
    assert "PASS" in (temp_dir / "math.py").read_text()


# ── HumanEval #6: 空输入/边界 ──────────────────────────────


@pytest.mark.asyncio
async def test_humaneval_edge_cases(coding_harness, temp_dir):
    """边界条件测试：空列表、大数、特殊字符。"""
    code = '''def safe_max(arr):
    if not arr:
        return None
    return max(arr)

if __name__ == "__main__":
    assert safe_max([]) is None
    assert safe_max([1]) == 1
    assert safe_max([-5, -2, -10]) == -2
    print("PASS")
'''
    model = MockModelProvider(MockModelConfig(responses=[
        make_tool_events([ToolCall(id="tc1", name="Write", arguments={
            "path": str(temp_dir / "safe_max.py"),
            "content": code,
        })]),
        make_tool_events([ToolCall(id="tc2", name="Bash", arguments={
            "command": f"python3 {temp_dir}/safe_max.py",
        })]),
        make_text_events("PASS"),
    ]))

    result = await react_loop(
        prompt="写 safe_max 处理边界",
        model=model,
        harness=coding_harness,
        tools=coding_harness.registry.all_tools(),
        context_builder=DefaultContextBuilder(),
        max_turns=5,
    )

    assert result.reason == "completed"
