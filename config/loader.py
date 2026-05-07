"""配置加载器：YAML 读取 + 环境变量替换 + Pydantic 校验。"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .schema import AppConfig


# 匹配 ${VAR} 或 ${VAR:-default} 格式
_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _substitute_env_vars(value: str) -> str:
    """替换字符串中的 ${VAR} 和 ${VAR:-default} 环境变量引用。"""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2) or ""
        return os.environ.get(var_name, default_value)

    return _ENV_PATTERN.sub(_replace, value)


def _walk_and_substitute(obj: object) -> object:
    """递归遍历配置对象，替换所有字符串中的环境变量。"""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    加载配置文件，替换环境变量，返回 Pydantic 校验后的 AppConfig。

    查找顺序：
    1. 显式指定的 config_path
    2. 环境变量 HANK_CLAW_CONFIG
    3. 当前目录下的 config.yaml
    4. ~/my-agent/config.yaml
    """
    if config_path is None:
        config_path = os.environ.get("HANK_CLAW_CONFIG")

    if config_path is None:
        candidates = [
            Path.cwd() / "config.yaml",
            Path.home() / "my-agent" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        # 没有配置文件，使用默认配置
        return AppConfig()

    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # 递归替换环境变量
    substituted = _walk_and_substitute(raw_config)

    return AppConfig(**substituted)
