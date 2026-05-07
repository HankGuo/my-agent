"""Gradio Web UI：聊天界面 + 配置管理。"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import gradio as gr

from agent.loop import react_loop, DefaultContextBuilder, Message, TurnState
from agent.models import ModelProvider, StreamEvent, parse_stream_events, tool_to_openai_schema
from agent.harness import Harness
from tools.base import ToolUseContext
from tools.registry import ToolRegistry

logger = logging.getLogger("my-agent.webui")


class WebUI:
    """Gradio Web 界面。"""

    def __init__(
        self,
        model: ModelProvider,
        harness: Harness,
        context_builder: DefaultContextBuilder,
        registry: ToolRegistry,
        max_turns: int = 50,
        agent_name: str = "my-agent",
    ):
        self._model = model
        self._harness = harness
        self._context_builder = context_builder
        self._registry = registry
        self._max_turns = max_turns
        self._agent_name = agent_name

    async def _run_agent_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """流式运行 Agent，逐步 yield 累积的响应文本。"""
        tools = self._registry.all_tools()
        tool_schemas = [tool_to_openai_schema(t.name, t.description, t.input_schema) for t in tools if t.is_enabled()]
        system_prompt = self._context_builder.build_system_prompt(tools)
        tool_use_ctx = ToolUseContext(session_id="webui")

        state = TurnState()
        state.messages.append(Message(role="user", content=message))

        accumulated_text = ""

        for turn in range(self._max_turns):
            state.turn_count = turn + 1
            api_messages = self._context_builder.build_messages(state)

            # 调用模型，实时 yield 文本
            events: list[StreamEvent] = []
            try:
                async for event in self._model.complete(
                    messages=api_messages,
                    system=system_prompt,
                    tools=tool_schemas,
                ):
                    events.append(event)
                    if event.type == "text" and isinstance(event.content, str):
                        accumulated_text += event.content
                        yield accumulated_text
            except Exception as e:
                yield f"{accumulated_text}\n\n[模型调用失败: {e}]"
                return

            response = parse_stream_events(events)

            if response.finish_reason == "error":
                yield f"{accumulated_text}\n\n[模型返回错误]"
                return

            # 记录 assistant 消息
            assistant_msg = Message(
                role="assistant",
                content=response.text or None,
                reasoning_content=response.reasoning_content or None,
            )
            if response.tool_calls:
                assistant_msg.tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
            state.messages.append(assistant_msg)

            if not response.tool_calls:
                return

            # 执行工具
            tool_results = await self._harness.execute_tools(response.tool_calls, tool_use_ctx)
            for result_msg in tool_results:
                state.messages.append(Message(
                    role="tool",
                    content=result_msg.get("content", ""),
                    tool_call_id=result_msg.get("tool_call_id", ""),
                    name=result_msg.get("name", ""),
                ))

            # 下一轮继续，重置累积文本
            accumulated_text = ""

        yield accumulated_text + "\n\n[已达最大轮次]"

    def build(self) -> gr.Blocks:
        """构建 Gradio 界面。"""
        with gr.Blocks(
            title=self._agent_name,
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown(f"# 🤖 {self._agent_name}\n个人 AI Agent，支持文件读写、Shell 命令等工具调用")

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=600)
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="输入消息...",
                            scale=5,
                            show_label=False,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                        clear_btn = gr.Button("清空", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### 工具列表")
                    tool_info = "\n".join(f"- **{t.name}**: {t.description}" for t in self._registry.all_tools())
                    gr.Markdown(tool_info)

                    gr.Markdown("### 配置")
                    gr.Textbox(label="当前模型", value=self._model.model, interactive=False)
                    gr.Textbox(label="最大轮次", value=str(self._max_turns), interactive=False)

            async def respond(message: str, history: list):
                """处理用户消息，流式输出。"""
                if not message.strip():
                    yield "", history
                    return

                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": ""})
                yield "", history

                try:
                    async for chunk in self._run_agent_streaming(message):
                        history[-1] = {"role": "assistant", "content": chunk}
                        yield "", history
                except Exception as e:
                    history[-1] = {"role": "assistant", "content": f"执行出错: {e}"}
                    yield "", history

            send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            clear_btn.click(fn=lambda: [], outputs=[chatbot])

        return app

    def launch(self, port: int = 7860, **kwargs):
        """启动 Web UI。"""
        app = self.build()
        app.launch(server_port=port, **kwargs)
