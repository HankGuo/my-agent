"""配置 schema 定义，使用 Pydantic 提供类型安全和默认值。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent 基础配置。"""
    name: str = "hank-claw"
    working_dir: str = "~/my-agent"
    language: str = "zh-CN"
    max_turns: int = 50
    max_budget_usd: float | None = None


class ModelProviderConfig(BaseModel):
    """单个模型 Provider 配置，均兼容 OpenAI 格式。"""
    base_url: str = ""
    api_key: str = ""
    models: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """模型配置总览。"""
    provider: str = "deepseek"
    providers: dict[str, ModelProviderConfig] = Field(default_factory=dict)
    default_model: str = "deepseek-chat"
    fallback_model: str | None = None


class PermissionRulesConfig(BaseModel):
    """权限规则配置。"""
    allow: list[str] = Field(default_factory=lambda: ["Read(*)", "Skill(*)"])
    deny: list[str] = Field(default_factory=lambda: ["Bash(rm -rf /)"])
    ask: list[str] = Field(default_factory=list)


class PermissionConfig(BaseModel):
    """权限系统配置。"""
    mode: Literal["default", "auto", "ask"] = "default"
    rules: PermissionRulesConfig = Field(default_factory=PermissionRulesConfig)


class VectorSearchConfig(BaseModel):
    """向量搜索配置（可选 Qdrant）。"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6333


class DreamingConfig(BaseModel):
    """记忆整合（梦境）配置。"""
    enabled: bool = True
    idle_timeout_seconds: int = 300


class MemoryConfig(BaseModel):
    """记忆系统配置。"""
    enabled: bool = True
    db_path: str = "~/my-agent/data/memory.db"
    memory_md_limit: int = 2200
    user_md_limit: int = 1375
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    dreaming: DreamingConfig = Field(default_factory=DreamingConfig)
    stale_days: int = 30
    archive_days: int = 90


class CuratorConfig(BaseModel):
    """技能 Curator 配置。"""
    enabled: bool = True
    interval_seconds: int = 3600


class SkillsConfig(BaseModel):
    """技能系统配置。"""
    dirs: list[str] = Field(default_factory=lambda: ["~/my-agent/skills"])
    auto_create: bool = True
    curator: CuratorConfig = Field(default_factory=CuratorConfig)


class MCPServerConfig(BaseModel):
    """单个 MCP 服务器配置。"""
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    type: Literal["stdio", "http"] = "stdio"
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    """Gateway 服务器配置。"""
    host: str = "127.0.0.1"
    port: int = 8765
    ws_path: str = "/ws"


class WebChannelConfig(BaseModel):
    """Web 通道配置。"""
    enabled: bool = True
    gradio_port: int = 7860


class CLIChannelConfig(BaseModel):
    """CLI 通道配置。"""
    enabled: bool = True


class ChannelsConfig(BaseModel):
    """通道配置总览。当前仅实现 CLI 和 Web。"""
    cli: CLIChannelConfig = Field(default_factory=CLIChannelConfig)
    web: WebChannelConfig = Field(default_factory=WebChannelConfig)


class DaemonConfig(BaseModel):
    """macOS launchd 守护配置。"""
    enabled: bool = True
    label: str = "com.hankia.my-agent"
    keep_alive: bool = True
    run_at_load: bool = True
    stdout_path: str = "~/my-agent/logs/stdout.log"
    stderr_path: str = "~/my-agent/logs/stderr.log"


class LoggingConfig(BaseModel):
    """日志配置。"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "~/my-agent/logs/agent.log"
    max_bytes: int = 10_485_760  # 10MB
    backup_count: int = 5


class AppConfig(BaseModel):
    """应用总配置。对应 config.yaml 的完整结构。"""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
