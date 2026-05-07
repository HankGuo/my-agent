# my-agent

个人 AI Agent，支持 ReAct 推理循环、工具调用、权限管理和多通道交互。

## 特性

- **ReAct 推理循环**：基于 Thought-Action-Observation 模式的多轮交互
- **工具调用**：内置 Bash、文件读写等工具，支持 MCP 扩展
- **权限管理**：细粒度规则引擎（allow/deny/ask），保护系统安全
- **多模型支持**：OpenAI、DeepSeek、Qwen（通义千问）、ChatGLM（智谱）等兼容 OpenAI 格式的 Provider
- **多通道交互**：CLI 终端交互 + Gradio Web UI
- **流式响应**：支持 reasoning_content（DeepSeek 思考模式）
- **并发安全**：只读工具并行执行，写入工具串行执行

## 架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI/Web   │────▶│  ReAct Loop │────▶│   Model     │
│   Channel   │◀────│             │◀────│  Provider   │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │   Harness   │
                    │  工具编排    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌──────┐    ┌────────┐   ┌──────────┐
         │ Bash │    │  Read  │   │  Write   │
         └──────┘    └────────┘   └──────────┘
```

## 安装

```bash
# 克隆仓库
git clone <repo-url> my-agent
cd my-agent

# 安装依赖
pip install -r requirements.txt
```

### 依赖

- Python >= 3.10
- httpx（API 调用）
- pydantic（配置校验）
- gradio（Web UI，可选）
- pytest + pytest-asyncio（测试）

## 配置

复制 `config.yaml` 并根据需要修改：

```yaml
agent:
  name: "my-agent"
  working_dir: "."
  language: "zh-CN"
  max_turns: 50

model:
  provider: "deepseek"
  providers:
    deepseek:
      base_url: "https://api.deepseek.com/v1"
      api_key: "${DEEPSEEK_API_KEY}"
      models: ["deepseek-chat"]
  default_model: "deepseek-chat"

permissions:
  mode: "default"
  rules:
    allow:
      - "Read(*)"
      - "Bash(ls *)"
    deny:
      - "Bash(rm -rf /)"
    ask:
      - "Write(*)"
```

支持环境变量注入，如 `${DEEPSEEK_API_KEY}`。

## 使用

### CLI 模式

```bash
# 交互模式
python main.py --cli

# 单次对话
python main.py --prompt "写一个 hello world 的 Python 脚本"

# 调试模式
python main.py --cli --debug
```

### Web UI 模式

```bash
python main.py --web
```

启动后访问 `http://127.0.0.1:7860`

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行压力测试
pytest tests/stress/ -v

# 运行基准测试
pytest tests/benchmark/ -v
```

当前测试覆盖 135 个用例，涵盖：
- **单元测试**：ReAct 循环、模型层、工具编排、权限系统、内置工具
- **集成测试**：端到端编码任务、并发读写、权限拒绝、长消息链
- **基准测试**：HumanEval 风格代码生成、SWE-bench 风格 bug 修复、AgentBench 风格 OS 交互
- **压力测试**：边界条件、故障注入、并发压力、资源释放

## 项目结构

```
my-agent/
├── main.py                 # 入口文件
├── config.yaml             # 配置文件
├── config/
│   ├── loader.py           # 配置加载
│   └── schema.py           # Pydantic 配置模型
├── agent/
│   ├── loop.py             # ReAct 核心循环
│   ├── models.py           # 模型 Provider 适配器
│   └── harness.py          # Harness 工具编排
├── tools/
│   ├── base.py             # Tool 协议定义
│   ├── registry.py         # 工具注册表
│   ├── permission.py       # 权限引擎
│   ├── orchestration.py    # 分区并发执行
│   └── builtin/            # 内置工具
│       ├── bash.py
│       ├── file_read.py
│       └── file_write.py
├── channels/               # 交互通道
│   ├── cli.py
│   └── base.py
├── webui/                  # Gradio Web UI
│   └── app.py
├── tests/                  # 测试用例
│   ├── unit/
│   ├── integration/
│   ├── benchmark/
│   └── stress/
└── utils/
    └── logging.py
```

## 设计原则

1. **模型只推理，Harness 管执行**：模型层与工具执行严格分离
2. **权限默认询问**：未匹配规则时默认 ASK，不静默放行危险操作
3. **只读并行，写入串行**：并发安全通过工具分区实现
4. **流式响应可中断**：abort_signal 在模型流式调用中可检查
5. **工具结果截断**：超长结果进入上下文前截断，防止 Token 膨胀

## 许可证

MIT
