"""模型适配层测试：覆盖流式解析、错误处理、reasoning_content。"""

from __future__ import annotations

import json

import pytest

from agent.models import (
    StreamEvent,
    ToolCall,
    ModelResponse,
    parse_stream_events,
    tool_to_openai_schema,
)


# ── parse_stream_events 测试 ───────────────────────────────


class TestParseStreamEvents:
    """流式事件解析的边界和正确性测试。"""

    def test_empty_events(self):
        """空事件列表应返回空响应。"""
        result = parse_stream_events([])
        assert result.text == ""
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    def test_simple_text(self):
        """纯文本事件解析。"""
        events = [
            StreamEvent(type="text", content="Hello"),
            StreamEvent(type="text", content=" World"),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.text == "Hello World"
        assert result.finish_reason == "stop"

    def test_text_with_reasoning(self):
        """DeepSeek 风格 reasoning_content 解析。"""
        events = [
            StreamEvent(type="reasoning", content="让我想想"),
            StreamEvent(type="reasoning", content="..."),
            StreamEvent(type="text", content="答案是 42"),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.text == "答案是 42"
        assert result.reasoning_content == "让我想想..."
        assert result.finish_reason == "stop"

    def test_single_tool_call(self):
        """单工具调用解析。"""
        events = [
            StreamEvent(type="tool_use", content={
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json.dumps({"path": "/tmp/test.txt"}),
                },
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Read"
        assert result.tool_calls[0].arguments == {"path": "/tmp/test.txt"}
        assert result.finish_reason == "tool_calls"

    def test_multiple_tool_calls(self):
        """多工具调用解析。"""
        events = [
            StreamEvent(type="tool_use", content={
                "id": "call_1",
                "type": "function",
                "function": {"name": "Read", "arguments": "{\"path\": \"a.txt\"}"},
            }),
            StreamEvent(type="tool_use", content={
                "id": "call_2",
                "type": "function",
                "function": {"name": "Read", "arguments": "{\"path\": \"b.txt\"}"},
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[1].id == "call_2"

    def test_tool_call_with_invalid_json(self):
        """工具参数 JSON 解析失败时回退到 _raw。"""
        events = [
            StreamEvent(type="tool_use", content={
                "id": "call_1",
                "type": "function",
                "function": {"name": "Read", "arguments": "不是JSON"},
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.tool_calls[0].arguments == {"_raw": "不是JSON"}

    def test_error_event(self):
        """error 事件应返回 finish_reason="error"。"""
        events = [
            StreamEvent(type="text", content="部分文本"),
            StreamEvent(type="error", content="连接超时"),
        ]
        result = parse_stream_events(events)
        assert result.finish_reason == "error"
        assert result.text == ""  # 遇到 error 后应丢弃之前的内容

    def test_no_done_event(self):
        """没有 done 事件时，有 tool_calls 则 finish_reason 为 tool_calls，否则 stop。"""
        events_text = [StreamEvent(type="text", content="hello")]
        result_text = parse_stream_events(events_text)
        assert result_text.finish_reason == "stop"
        assert result_text.text == "hello"

        events_tool = [
            StreamEvent(type="tool_use", content={
                "id": "c1",
                "type": "function",
                "function": {"name": "Read", "arguments": "{}"},
            }),
        ]
        result_tool = parse_stream_events(events_tool)
        assert result_tool.finish_reason == "tool_calls"

    def test_mixed_content_and_tool(self):
        """文本 + 工具调用的混合响应。"""
        events = [
            StreamEvent(type="text", content="我来查一下"),
            StreamEvent(type="tool_use", content={
                "id": "c1",
                "type": "function",
                "function": {"name": "Read", "arguments": "{}"},
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.text == "我来查一下"
        assert len(result.tool_calls) == 1
        assert result.finish_reason == "tool_calls"

    def test_none_content_handling(self):
        """content 为 None 的事件不应导致异常。"""
        events = [
            StreamEvent(type="text", content=None),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.text == ""

    def test_unicode_arguments(self):
        """Unicode 参数正确解析。"""
        args = {"path": "/用户/文档/测试.txt", "内容": "中文"}
        events = [
            StreamEvent(type="tool_use", content={
                "id": "c1",
                "type": "function",
                "function": {"name": "Write", "arguments": json.dumps(args, ensure_ascii=False)},
            }),
            StreamEvent(type="done"),
        ]
        result = parse_stream_events(events)
        assert result.tool_calls[0].arguments == args


# ── tool_to_openai_schema 测试 ─────────────────────────────


class TestToolSchema:
    def test_simple_schema(self):
        schema = tool_to_openai_schema(
            name="Read",
            description="读取文件",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "Read"
        assert schema["function"]["description"] == "读取文件"
        assert schema["function"]["parameters"]["type"] == "object"

    def test_empty_schema(self):
        schema = tool_to_openai_schema("NoArgs", "无参数", {"type": "object", "properties": {}})
        assert schema["function"]["parameters"]["properties"] == {}
