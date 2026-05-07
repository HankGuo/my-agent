"""CLI 通道：本地终端对话适配器。"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, AsyncGenerator

from .base import ChannelAdapter, ChannelMessage, ChannelResponse

logger = logging.getLogger("my-agent.channels.cli")


class CLIAdapter:
    """
    本地 CLI 对话适配器。

    交互流程：读取用户输入 → 包装为 ChannelMessage → 交给 Agent → 输出响应
    """

    name = "cli"

    def __init__(self) -> None:
        self._dispatch: Any = None
        self._running = False

    async def start(self, dispatch: Any) -> None:
        """启动 CLI 交互循环。"""
        self._dispatch = dispatch
        self._running = True
        logger.info("CLI 通道已启动")

    async def stop(self) -> None:
        """停止 CLI。"""
        self._running = False

    async def send(self, response: ChannelResponse) -> None:
        """输出 Agent 响应到终端。"""
        print(f"\n🤖 {response.text}\n")

    async def receive(self) -> AsyncGenerator[ChannelMessage, None]:
        """从终端读取用户输入。"""
        while self._running:
            try:
                # 在事件循环中运行阻塞的 input()
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("你> ").strip()
                )
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "退出"):
                    self._running = False
                    break
                yield ChannelMessage(text=line, sender="user", channel="cli")
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

    def validate_config(self) -> bool:
        """CLI 无需额外配置。"""
        return True

    async def run_interactive(
        self,
        agent_func: Any,
    ) -> None:
        """
        运行交互式对话。

        agent_func: async (text: str) -> str
        接收用户文本，返回 Agent 响应文本。
        """
        print("my-agent CLI 已启动，输入消息开始对话，输入 exit 退出\n")

        while self._running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("你> ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not line:
                continue
            if line.lower() in ("exit", "quit", "退出"):
                print("再见！")
                break

            try:
                # 显示思考提示
                print("🤖 ", end="", flush=True)

                # 流式输出
                response_text = ""
                async for chunk in agent_func(line):
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        response_text += chunk

                if not response_text:
                    print("(无响应)")

                print()  # 换行

            except Exception as e:
                logger.error("Agent 执行出错: %s", e)
                print(f"\n执行出错: {e}")
