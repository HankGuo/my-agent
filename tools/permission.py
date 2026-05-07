"""权限引擎：deny > allow > ask 规则链，工具名模式匹配。"""

from __future__ import annotations

import fnmatch
import logging

from tools.base import PermissionDecision, PermissionTier, ToolUseContext

logger = logging.getLogger("my-agent.permission")


class PermissionEngine:
    """
    权限引擎。

    规则优先级：deny > allow > ask(默认)
    规则格式：ToolName(pattern)，如 Bash(git *)、Read(*)、Skill(review)
    无括号的模式匹配整个工具名，如 Bash 匹配所有 Bash 工具调用。
    """

    def __init__(
        self,
        allow_rules: list[str] | None = None,
        deny_rules: list[str] | None = None,
        ask_rules: list[str] | None = None,
        mode: str = "default",
    ):
        self._allow_rules = allow_rules or []
        self._deny_rules = deny_rules or []
        self._ask_rules = ask_rules or []
        self._mode = mode

    def check(
        self,
        tool_name: str,
        args: dict,
        context: ToolUseContext,
    ) -> PermissionDecision:
        """
        检查工具调用权限。

        返回 PermissionDecision，优先级：deny > allow > ask。
        """
        # 构造匹配键：Bash + 命令参数
        match_key = tool_name
        if tool_name == "Bash" and args.get("command"):
            match_key = f"Bash({args['command']})"

        # 1. deny 规则（最高优先级）
        for rule in self._deny_rules:
            if self._match_rule(rule, tool_name, match_key):
                logger.info("权限拒绝: %s 匹配规则 %s", match_key, rule)
                return PermissionDecision(tier=PermissionTier.DENY, reason=f"匹配拒绝规则: {rule}")

        # 2. allow 规则
        for rule in self._allow_rules:
            if self._match_rule(rule, tool_name, match_key):
                return PermissionDecision(tier=PermissionTier.ALLOW, reason=f"匹配允许规则: {rule}")

        # 3. ask 规则
        for rule in self._ask_rules:
            if self._match_rule(rule, tool_name, match_key):
                return PermissionDecision(tier=PermissionTier.ASK, reason=f"匹配询问规则: {rule}")

        # 4. 默认行为
        if self._mode == "auto":
            return PermissionDecision(tier=PermissionTier.ALLOW, reason="自动模式默认允许")
        return PermissionDecision(tier=PermissionTier.ASK, reason="无匹配规则，默认询问")

    @staticmethod
    def _match_rule(rule: str, tool_name: str, match_key: str) -> bool:
        """
        匹配规则。

        规则格式：
        - "Bash(git *)" → 匹配 Bash 工具且命令以 git 开头
        - "Bash" → 匹配所有 Bash 工具调用
        - "Read(*)" → 匹配所有 Read 工具调用
        - "mcp__server" → 匹配特定 MCP 服务器
        """
        # 带括号的规则：ToolName(pattern)
        if "(" in rule and rule.endswith(")"):
            rule_tool, _, pattern = rule[:-1].partition("(")
            if rule_tool != tool_name:
                return False
            # pattern 匹配命令/参数文本
            return fnmatch.fnmatch(match_key[len(rule_tool) + 1 : -1] if "(" in match_key else "", pattern)

        # 不带括号：直接匹配工具名
        return fnmatch.fnmatch(tool_name, rule)

    def should_proceed(self, decision: PermissionDecision) -> bool:
        """根据权限判定和模式决定是否继续执行。"""
        if decision.tier == PermissionTier.DENY:
            return False
        if decision.tier == PermissionTier.ALLOW:
            return True
        # ASK 模式：个人 Agent 默认继续，记录日志
        logger.info("权限询问（自动继续）: %s", decision.reason)
        return True
