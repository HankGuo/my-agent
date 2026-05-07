"""my-agent 入口：CLI / daemon / gateway 启动。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import load_config
from config.schema import AppConfig
from utils.logging import setup_logging

logger = logging.getLogger("my-agent")


def _build_agent(config: AppConfig):
    """根据配置构建 Agent 组件。"""
    from agent.models import create_provider, OpenAIAdapter
    from agent.loop import DefaultContextBuilder
    from agent.harness import Harness
    from tools.base import Tool
    from tools.registry import ToolRegistry
    from tools.permission import PermissionEngine
    from tools.builtin.bash import BashTool
    from tools.builtin.file_read import FileReadTool
    from tools.builtin.file_write import FileWriteTool

    # 模型 Provider
    provider_name = config.model.provider
    provider_config = config.model.providers.get(provider_name)
    if not provider_config:
        raise ValueError(f"未配置模型 Provider: {provider_name}")

    provider_dict = {
        "base_url": provider_config.base_url,
        "api_key": provider_config.api_key,
        "models": provider_config.models,
    }
    model = create_provider(provider_name, provider_dict, config.model.default_model)

    # 工具注册
    registry = ToolRegistry()
    registry.register_many([BashTool(), FileReadTool(), FileWriteTool()])

    # 权限引擎
    permission = PermissionEngine(
        allow_rules=config.permissions.rules.allow,
        deny_rules=config.permissions.rules.deny,
        ask_rules=config.permissions.rules.ask,
        mode=config.permissions.mode,
    )

    # Harness
    harness = Harness(
        registry=registry,
        permission_engine=permission,
        working_dir=config.agent.working_dir,
    )

    # 上下文构建器
    context_builder = DefaultContextBuilder(
        system_prompt=f"你是 {config.agent.name}，一个智能 AI 助手。请用中文回答。"
    )

    return model, harness, context_builder, registry


async def _run_cli(config: AppConfig, prompt: str | None = None):
    """运行 CLI 模式。"""
    from agent.loop import react_loop
    from channels.cli import CLIAdapter

    model, harness, context_builder, registry = _build_agent(config)

    if prompt:
        # 单次模式
        result = await react_loop(
            prompt=prompt,
            model=model,
            harness=harness,
            tools=registry.all_tools(),
            context_builder=context_builder,
            max_turns=config.agent.max_turns,
        )
        print(result.text if result.text else "(无响应)")
        if result.error:
            print(f"错误: {result.error}", file=sys.stderr)
        return

    # 交互模式
    cli = CLIAdapter()

    async def agent_func(text: str):
        """流式 Agent 响应。"""
        result = await react_loop(
            prompt=text,
            model=model,
            harness=harness,
            tools=registry.all_tools(),
            context_builder=context_builder,
            max_turns=config.agent.max_turns,
        )
        if result.text:
            yield result.text
        if result.error:
            yield f"\n[错误: {result.error}]"

    await cli.run_interactive(agent_func)


def main():
    parser = argparse.ArgumentParser(description="my-agent: 个人 AI Agent")
    parser.add_argument("--cli", action="store_true", help="启动 CLI 交互模式")
    parser.add_argument("--prompt", type=str, help="单次对话提示")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式启动")
    parser.add_argument("--web", action="store_true", help="启动 Web UI 模式")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 初始化日志
    log_level = "DEBUG" if args.debug else config.logging.level
    setup_logging(
        level=log_level,
        log_file=config.logging.file,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )

    logger.info("my-agent 启动，模型: %s/%s", config.model.provider, config.model.default_model)

    # 根据模式运行
    if args.daemon:
        logger.info("守护进程模式（尚未实现 Gateway）")
        print("守护进程模式尚未实现，请使用 --cli 模式")
        return

    # Web UI 模式
    if args.web:
        _run_web(config)
        return

    # 默认或 --cli 模式
    prompt = args.prompt
    asyncio.run(_run_cli(config, prompt))


def _run_web(config: AppConfig):
    """运行 Web UI 模式。"""
    from webui.app import WebUI

    model, harness, context_builder, registry = _build_agent(config)

    webui = WebUI(
        model=model,
        harness=harness,
        context_builder=context_builder,
        registry=registry,
        max_turns=config.agent.max_turns,
        agent_name=config.agent.name,
    )

    logger.info("Web UI 启动: http://127.0.0.1:%d", config.channels.web.gradio_port)
    webui.launch(
        port=config.channels.web.gradio_port,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
