# 小奈 · LangGraph 版

用 LangGraph 重新实现兴趣社区 AI 助手「小奈」的核心流程。原版在 Java + Spring AI 里手写了路由、状态、人工确认和输出校验；这个版本用 LangGraph 的原生概念实现同一套东西，对比两种做法的取舍。

数据是内存里的模拟活动，模型用通义千问 qwen-plus（DashScope 的 OpenAI 兼容接口）。

## 架构

```
START → supervisor ─┬→ search_agent ⇄ search_tools ─────────────┐
                    ├→ signup_agent ⇄ signup_tools → confirm ───┼→ END
                    │        └→ signup_done（代码兜底）→ confirm │
                    └→ chat ─────────────────────────→ check ───┘
```

- **supervisor**：先用规则判断高置信意图（查活动 / 报名 / 我的报名），判断不了才问模型；模型回答不合法就退回闲聊。
- **search_agent / signup_agent**：两个子 agent，各自只拿自己需要的工具。
- **confirm**：`interrupt()` 暂停整张图等用户点确认，`Command(resume=...)` 恢复；确认前、执行前各校验一次能不能报。
- **check**：逐句校验回答，拦下「已帮你报名」这类与实际状态不符的说法，标记依据里找不到的《标题》。
- **checkpointer**：`InMemorySaver` 按 `thread_id` 保存会话状态，跨轮记得对话和待确认的提议。

## 设计要点（Harness 思路）

模型只负责理解和表达，确定性的事交给代码：

| 问题 | 做法 |
| --- | --- |
| 模型不调工具就编结果 | 高置信意图由规则路由，直接进对应 agent |
| 工具调用死循环 | 每轮最多 3 次，到上限后不再给工具，逼模型直接回答（`MAX_ROUNDS`） |
| 模型冒充用户 | `user_id` 从 `RunnableConfig` 注入，不是工具参数 |
| 模型自己执行写操作 | 只能 `propose_signup`，真正执行在 `confirm` 节点，用户确认后才做 |
| 确认期间名额被抢 | 执行前再校验一次 |
| 模型说「已帮你报名」但没报 | `check` 节点逐句拦下 |
| 模型自己推算星期几算错 | 工具结果里直接带 `weekday` |

## 和 Java 版的对照

| LangGraph | Java + Spring AI 版 |
| --- | --- |
| `StateGraph` + 条件边 | `IntentRules` 规则路由 + 手写流程 |
| `interrupt()` / `Command(resume=)` | 确认卡 + Redis 里的显式「待确认提议」状态 |
| checkpointer（按 `thread_id`） | Redis 会话历史 + 服务端存储 |
| 节点后的 `check` | `SentenceGate` / `AnswerCheck` 流式逐句校验 |
| `RunnableConfig` 注入身份 | 工具身份由服务端注入 |
| LangSmith tracing（未接） | `AiTrace` + 问答落库 + 监控看板 |

**取舍**：LangGraph 把流程画成图，`interrupt` 和 checkpointer 现成可用，多 agent 和长流程更省事；Java 版和业务服务在同一个 JVM，工具直接调 Service，没有跨语言、跨服务的成本，单 agent 加一个确认状态用不上图编排。

## 真模型跑出来的问题

单测全过以后，用 qwen-plus 跑真实对话又发现 3 个问题，修法都落在代码里：

1. **星期算错**：模型把 2026-11-07（周六）说成周五 → 工具结果直接带 `weekday`，今天的星期也写进系统提示词。
2. **校验误杀**：「你已报名《周末飞盘新手局》」是查询结果，却被「虚报已完成」规则拦下 → 规则收窄为只拦「已帮你 / 为你报名」。
3. **用户说「帮我报一下」，模型却反问「要报名吗？」**：路由判对了，提示词也写了要直接调用 `propose_signup`，模型照样不调 → 新增 `signup_done` 节点：模型没提议、这一轮只查到一个候选时，由代码直接出确认卡。

第 3 条和 Java 版遇到的情况一样：提示词约束不住的，交给代码。

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 单测（不调模型，用脚本化的假模型跑整张图）
.venv/bin/python -m pytest -q tests

# 命令行对话
export DASHSCOPE_API_KEY=你的密钥
.venv/bin/python main.py
```

可以试试：`上海周末有什么活动` → `帮我报名飞盘那个` → `我报了哪些` → `武汉的狼人杀也帮我报一下`。
