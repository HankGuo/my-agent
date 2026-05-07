"""权限系统测试：覆盖规则匹配、模式行为和边界。"""

from __future__ import annotations

import pytest

from tools.permission import PermissionEngine, PermissionDecision, PermissionTier
from tools.base import ToolUseContext


# ── 基础规则匹配 ───────────────────────────────────────────


@pytest.fixture
def default_engine():
    return PermissionEngine(
        allow_rules=["Read(*)", "Bash(ls *)"],
        deny_rules=["Bash(rm -rf /)", "Write(/etc/*)"],
        ask_rules=["Bash(sudo *)"],
        mode="default",
    )


def test_allow_wildcard(default_engine):
    """Read(*) 应允许所有 Read 调用。"""
    ctx = ToolUseContext()
    dec = default_engine.check("Read", {"path": "/any/path"}, ctx)
    assert dec.tier == PermissionTier.ALLOW


def test_allow_specific_pattern(default_engine):
    """Bash(ls *) 只允许 ls 命令。"""
    ctx = ToolUseContext()
    dec = default_engine.check("Bash", {"command": "ls /tmp"}, ctx)
    assert dec.tier == PermissionTier.ALLOW

    dec = default_engine.check("Bash", {"command": "cat /tmp"}, ctx)
    assert dec.tier != PermissionTier.ALLOW


def test_deny_overrides_allow(default_engine):
    """Deny 规则应覆盖 Allow。"""
    ctx = ToolUseContext()
    dec = default_engine.check("Bash", {"command": "rm -rf /"}, ctx)
    assert dec.tier == PermissionTier.DENY


def test_ask_rule(default_engine):
    """Ask 规则应返回 ASK。"""
    ctx = ToolUseContext()
    dec = default_engine.check("Bash", {"command": "sudo apt update"}, ctx)
    assert dec.tier == PermissionTier.ASK


def test_default_ask(default_engine):
    """未匹配任何规则时默认询问（ASK）。"""
    ctx = ToolUseContext()
    dec = default_engine.check("Delete", {"path": "/tmp"}, ctx)
    assert dec.tier == PermissionTier.ASK


# ── 模式行为 ───────────────────────────────────────────────


def test_auto_mode_allows_all():
    """auto 模式允许所有。"""
    engine = PermissionEngine(allow_rules=[], deny_rules=[], ask_rules=[], mode="auto")
    ctx = ToolUseContext()
    dec = engine.check("Bash", {"command": "rm -rf /"}, ctx)
    assert dec.tier == PermissionTier.ALLOW


def test_ask_mode_asks_all():
    """ask 模式对所有请求确认。"""
    engine = PermissionEngine(allow_rules=[], deny_rules=[], ask_rules=[], mode="ask")
    ctx = ToolUseContext()
    dec = engine.check("Read", {"path": "/tmp"}, ctx)
    assert dec.tier == PermissionTier.ASK


# ── 边界测试 ───────────────────────────────────────────────


def test_empty_rules():
    """空规则集默认询问（除 auto 模式）。"""
    engine = PermissionEngine(allow_rules=[], deny_rules=[], ask_rules=[], mode="default")
    ctx = ToolUseContext()
    dec = engine.check("Anything", {}, ctx)
    assert dec.tier == PermissionTier.ASK


def test_malformed_rule_ignored():
    """格式错误的规则应被忽略，不导致崩溃。"""
    engine = PermissionEngine(
        allow_rules=["Read(*)", "MalformedRule", "", "Bash"],
        deny_rules=[],
        ask_rules=[],
        mode="default",
    )
    ctx = ToolUseContext()
    dec = engine.check("Read", {"path": "/tmp"}, ctx)
    assert dec.tier == PermissionTier.ALLOW

    dec = engine.check("MalformedRule", {}, ctx)
    assert dec.tier == PermissionTier.ALLOW


def test_should_proceed():
    """should_proceed 对 ALLOW 和 ASK 返回 True，DENY 返回 False。"""
    engine = PermissionEngine(mode="auto")
    ctx = ToolUseContext()
    allow = engine.check("X", {}, ctx)
    assert engine.should_proceed(allow)

    engine2 = PermissionEngine(ask_rules=["X(*)"], mode="default")
    ask = engine2.check("X", {}, ctx)
    assert engine2.should_proceed(ask)

    engine3 = PermissionEngine(deny_rules=["X(*)"], mode="default")
    deny = engine3.check("X", {}, ctx)
    assert not engine3.should_proceed(deny)


def test_rule_priority_order():
    """Deny > Ask > Allow 的优先级。"""
    engine = PermissionEngine(
        allow_rules=["Bash(*)"],
        deny_rules=["Bash(*)"],
        ask_rules=["Bash(*)"],
        mode="default",
    )
    ctx = ToolUseContext()
    dec = engine.check("Bash", {"command": "ls"}, ctx)
    # 同时匹配三个规则时，Deny 优先级最高
    assert dec.tier == PermissionTier.DENY


# ── 复杂参数匹配 ───────────────────────────────────────────


def test_bash_command_matching():
    """Bash 工具的命令参数匹配（PermissionEngine 对 Bash 有特殊处理）。"""
    engine = PermissionEngine(
        allow_rules=["Bash(ls *)", "Bash(echo *)"],
        deny_rules=["Bash(rm *)"],
        ask_rules=["Bash(sudo *)"],
        mode="default",
    )
    ctx = ToolUseContext()

    dec = engine.check("Bash", {"command": "ls /tmp"}, ctx)
    assert dec.tier == PermissionTier.ALLOW

    dec = engine.check("Bash", {"command": "rm -rf /tmp"}, ctx)
    assert dec.tier == PermissionTier.DENY

    dec = engine.check("Bash", {"command": "sudo apt update"}, ctx)
    assert dec.tier == PermissionTier.ASK

    dec = engine.check("Bash", {"command": "cat /etc/passwd"}, ctx)
    assert dec.tier == PermissionTier.ASK
