"""通道适配器协议。所有通道（CLI、Web、飞书等）必须满足此接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Protocol, runtime_checkable


@dataclass
class ChannelMessage:
    """通道入站消息。"""
    text: str
    sender: str = ""
    channel: str = ""
    metadata: dict = field(default_factory=dict)
    attachments: list[dict] = field(default_factory=list)


@dataclass
class ChannelResponse:
    """通道出站响应。"""
    text: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ChannelAdapter(Protocol):
    """通道适配器协议。平台差异完全封装在实现内。"""

    name: str

    async def start(self, dispatch: Any) -> None:
        """启动通道，接收消息调度器。"""
        ...

    async def stop(self) -> None:
        """停止通道。"""
        ...

    async def send(self, response: ChannelResponse) -> None:
        """发送响应到通道。"""
        ...

    async def receive(self) -> AsyncGenerator[ChannelMessage, None]:
        """接收来自通道的消息流。"""
        ...

    def validate_config(self) -> bool:
        """校验通道配置是否完整。"""
        ...
