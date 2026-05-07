"""配置回写器：保存模型配置到 config.yaml，保留环境变量引用。"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .schema import ModelConfig

# 匹配环境变量引用格式
_ENV_REF_PATTERN = re.compile(r"^\$\{\w+(?::-[^}]*)?\}$")


def _is_env_ref(value: str) -> bool:
    """判断字符串是否为环境变量引用，如 ${DEEPSEEK_API_KEY}。"""
    return bool(_ENV_REF_PATTERN.match(value))


def _load_raw_config(config_path: Path) -> dict:
    """加载原始 YAML（不替换环境变量）。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_raw_api_keys(config_path: Path) -> dict[str, str]:
    """获取原始配置中各 provider 的 api_key（可能含 ${VAR} 引用）。"""
    raw = _load_raw_config(config_path)
    providers = raw.get("model", {}).get("providers", {})
    return {name: cfg.get("api_key", "") for name, cfg in providers.items()}


def save_model_config(
    config_path: Path,
    model_config: ModelConfig,
    edited_api_keys: dict[str, str],
) -> None:
    """
    保存模型配置到 config.yaml，保留未修改的环境变量引用。

    参数:
        config_path: 配置文件路径
        model_config: 新的模型配置（Pydantic 对象）
        edited_api_keys: 用户在界面上编辑后的 api_key 值
            - 如果值为空字符串或全是 * 号，说明用户没有修改，保留原始引用
            - 如果值以 ${ 开头，直接写入（用户手动输入了环境变量引用）
            - 否则写入用户输入的实际值
    """
    raw = _load_raw_config(config_path)
    original_keys = _get_raw_api_keys(config_path)

    # 更新 model 部分
    if "model" not in raw:
        raw["model"] = {}

    raw["model"]["provider"] = model_config.provider
    raw["model"]["default_model"] = model_config.default_model
    raw["model"]["fallback_model"] = model_config.fallback_model

    # 构建 providers 字典
    new_providers: dict[str, dict] = {}
    for name, provider_cfg in model_config.providers.items():
        # 决定 api_key 的写入值
        edited_key = edited_api_keys.get(name, "")
        original_key = original_keys.get(name, "")

        if not edited_key or edited_key == "********":
            # 用户没改，保留原始值（可能是 ${VAR}）
            api_key_to_save = original_key
        elif _is_env_ref(edited_key):
            # 用户输入了环境变量引用
            api_key_to_save = edited_key
        else:
            # 用户输入了实际 key 值
            api_key_to_save = edited_key

        new_providers[name] = {
            "base_url": provider_cfg.base_url,
            "api_key": api_key_to_save,
            "models": provider_cfg.models,
        }

    raw["model"]["providers"] = new_providers

    # 写入文件
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
