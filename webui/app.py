"""Gradio Web UI：聊天界面 + 配置管理。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncGenerator

import gradio as gr

from agent.loop import react_loop, DefaultContextBuilder, Message, TurnState
from agent.models import (
    ModelProvider,
    StreamEvent,
    parse_stream_events,
    tool_to_openai_schema,
    create_provider,
)
from agent.harness import Harness
from config.schema import AppConfig, ModelConfig, ModelProviderConfig
from config.writer import save_model_config
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
        config: AppConfig | None = None,
        config_path: Path | None = None,
    ):
        self._model = model
        self._harness = harness
        self._context_builder = context_builder
        self._registry = registry
        self._max_turns = max_turns
        self._agent_name = agent_name
        self._config = config
        self._config_path = config_path

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

    def _get_provider_choices(self) -> list[str]:
        """获取当前所有 provider 名称列表。"""
        if self._config:
            return list(self._config.model.providers.keys())
        return []

    def _get_provider_detail(self, provider_name: str) -> tuple[str, str, str, list[str], str]:
        """获取指定 provider 的详情：(base_url, api_key_placeholder, models_str, models_list, default_model)。"""
        if not self._config or provider_name not in self._config.model.providers:
            return "", "", "", [], ""

        cfg = self._config.model.providers[provider_name]
        models_str = ", ".join(cfg.models)
        # API Key 脱敏显示
        api_key_display = "********" if cfg.api_key else ""
        default_model = self._config.model.default_model if self._config.model.provider == provider_name else (cfg.models[0] if cfg.models else "")
        return cfg.base_url, api_key_display, models_str, cfg.models, default_model

    def build(self) -> gr.Blocks:
        """构建 Gradio 界面。"""
        config = self._config

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
                    gr.Markdown("### ⚙️ 模型配置")

                    # Provider 选择
                    provider_choices = self._get_provider_choices()
                    current_provider = config.model.provider if config else ""
                    provider_dropdown = gr.Dropdown(
                        choices=provider_choices,
                        value=current_provider,
                        label="当前 Provider",
                        interactive=True,
                    )

                    # Provider 详情
                    init_detail = self._get_provider_detail(current_provider)
                    base_url_input = gr.Textbox(
                        label="Base URL",
                        value=init_detail[0],
                        interactive=True,
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        value=init_detail[1],
                        type="password",
                        interactive=True,
                    )
                    models_input = gr.Textbox(
                        label="模型列表（逗号分隔）",
                        value=init_detail[2],
                        interactive=True,
                    )

                    # 默认模型选择
                    default_model_dropdown = gr.Dropdown(
                        choices=init_detail[3],
                        value=init_detail[4],
                        label="默认模型",
                        interactive=True,
                    )

                    # Fallback 模型
                    fallback_model_input = gr.Textbox(
                        label="Fallback 模型（可选）",
                        value=config.model.fallback_model or "" if config else "",
                        interactive=True,
                    )

                    # 新增 Provider
                    with gr.Accordion("新增 Provider", open=False):
                        new_provider_name = gr.Textbox(label="Provider 名称")
                        new_provider_url = gr.Textbox(label="Base URL")
                        new_provider_key = gr.Textbox(label="API Key", type="password")
                        new_provider_models = gr.Textbox(label="模型列表（逗号分隔）")
                        add_provider_btn = gr.Button("添加", variant="secondary")

                    save_btn = gr.Button("💾 保存配置", variant="primary")
                    save_status = gr.Markdown("")

                    # ── 工具列表折叠 ──
                    with gr.Accordion("工具列表", open=False):
                        tool_info = "\n".join(
                            f"- **{t.name}**: {t.description}"
                            for t in self._registry.all_tools()
                        )
                        gr.Markdown(tool_info)

            # ── 事件处理 ──────────────────────────────────────────

            def on_provider_change(provider_name: str):
                """切换 Provider 时更新详情字段。"""
                base_url, api_key_display, models_str, models_list, default_model = self._get_provider_detail(provider_name)
                return (
                    gr.update(value=base_url),
                    gr.update(value=api_key_display),
                    gr.update(value=models_str),
                    gr.update(choices=models_list, value=default_model),
                )

            provider_dropdown.change(
                fn=on_provider_change,
                inputs=[provider_dropdown],
                outputs=[base_url_input, api_key_input, models_input, default_model_dropdown],
            )

            def on_models_change(models_str: str):
                """模型列表变更时更新默认模型下拉。"""
                models = [m.strip() for m in models_str.split(",") if m.strip()]
                value = models[0] if models else ""
                return gr.update(choices=models, value=value)

            models_input.change(
                fn=on_models_change,
                inputs=[models_input],
                outputs=[default_model_dropdown],
            )

            def on_add_provider(name: str, url: str, key: str, models_str: str):
                """添加新 Provider。"""
                if not name.strip():
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "⚠️ Provider 名称不能为空",
                    )

                name = name.strip()
                models = [m.strip() for m in models_str.split(",") if m.strip()]

                # 更新内存中的配置
                if self._config:
                    self._config.model.providers[name] = ModelProviderConfig(
                        base_url=url.strip(),
                        api_key=key.strip(),
                        models=models,
                    )

                new_choices = self._get_provider_choices()
                return (
                    gr.update(choices=new_choices, value=name),
                    gr.update(value=""),  # 清空名称
                    gr.update(value=""),  # 清空 URL
                    gr.update(value=""),  # 清空 Key
                    gr.update(value=""),  # 清空 models
                    f"✅ 已添加 Provider: {name}",
                )

            add_provider_btn.click(
                fn=on_add_provider,
                inputs=[new_provider_name, new_provider_url, new_provider_key, new_provider_models],
                outputs=[provider_dropdown, new_provider_name, new_provider_url, new_provider_key, new_provider_models, save_status],
            )

            def on_save(
                provider_name: str,
                base_url: str,
                api_key: str,
                models_str: str,
                default_model: str,
                fallback_model: str,
            ):
                """保存配置到文件并热更新模型 Provider。"""
                if not self._config or not self._config_path:
                    return "⚠️ 配置未加载，无法保存"

                # 解析模型列表
                models = [m.strip() for m in models_str.split(",") if m.strip()]

                # 更新当前选中 provider 的配置（内存）
                if provider_name in self._config.model.providers:
                    pcfg = self._config.model.providers[provider_name]
                    pcfg.base_url = base_url.strip()
                    # 只有用户真正输入了新 key 才更新内存
                    if api_key and api_key != "********":
                        pcfg.api_key = api_key.strip()
                    pcfg.models = models
                else:
                    # 新 provider（通过直接在下拉框输入的方式）
                    actual_key = api_key.strip() if api_key != "********" else ""
                    self._config.model.providers[provider_name] = ModelProviderConfig(
                        base_url=base_url.strip(),
                        api_key=actual_key,
                        models=models,
                    )

                # 更新全局模型配置
                self._config.model.provider = provider_name
                self._config.model.default_model = default_model or (models[0] if models else "")
                self._config.model.fallback_model = fallback_model.strip() or None

                # 收集编辑后的 api_key（用于回写时判断是否保留环境变量引用）
                edited_api_keys: dict[str, str] = {}
                for name in self._config.model.providers:
                    if name == provider_name:
                        edited_api_keys[name] = api_key
                    else:
                        # 未编辑的 provider 保留原样
                        edited_api_keys[name] = "********"

                # 持久化到 config.yaml
                try:
                    save_model_config(
                        config_path=self._config_path,
                        model_config=self._config.model,
                        edited_api_keys=edited_api_keys,
                    )
                except Exception as e:
                    logger.error("保存配置失败: %s", e)
                    return f"❌ 保存失败: {e}"

                # 热更新模型 Provider
                try:
                    pcfg = self._config.model.providers[provider_name]
                    provider_dict = {
                        "base_url": pcfg.base_url,
                        "api_key": pcfg.api_key,
                        "models": pcfg.models,
                    }
                    new_model = create_provider(
                        provider_name, provider_dict, self._config.model.default_model
                    )
                    self._model = new_model
                    logger.info("模型热更新: %s/%s", provider_name, self._config.model.default_model)
                except Exception as e:
                    logger.error("热更新模型失败: %s", e)
                    return f"⚠️ 配置已保存，但热更新失败: {e}"

                return f"✅ 配置已保存并生效（{provider_name}/{self._config.model.default_model}）"

            save_btn.click(
                fn=on_save,
                inputs=[
                    provider_dropdown,
                    base_url_input,
                    api_key_input,
                    models_input,
                    default_model_dropdown,
                    fallback_model_input,
                ],
                outputs=[save_status],
            )

            # ── 聊天事件 ──────────────────────────────────────

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
