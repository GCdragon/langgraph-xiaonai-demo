"""小奈的 LangGraph 版：supervisor + 两个子 agent + 人工确认 + 输出校验。

    START → supervisor ─┬→ search_agent ⇄ search_tools ─┐
                        ├→ signup_agent ⇄ signup_tools ─┼→ confirm（interrupt 等用户确认）→ END
                        │        └→ signup_done（模型没提议时代码兜底）┘
                        └→ chat ────────────────────────┴→ check（逐句校验）→ END
"""
import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from . import data, rules
from .tools import SEARCH_TOOLS, SIGNUP_TOOLS

MAX_ROUNDS = 3  # 每轮对话最多调几次工具；写在提示词里没用，在图里硬性限制

AGENTS = ("search", "signup", "chat")


class State(MessagesState):
    route: str
    rounds: int
    proposal: dict | None
    signed_up: bool
    problems: list[str]


SIGNUP_HINT = SystemMessage(
    "用户已经明确要报名：找到对应活动后直接调用 propose_signup，不要再反问用户要不要报名；"
    "系统会弹确认卡让用户确认。找不到或有多个候选时再问用户是哪一个。"
)


def _system() -> SystemMessage:
    t = data.STORE.today
    return SystemMessage(
        f"你是兴趣社区的助手小奈。今天是 {t.isoformat()}（{data.WEEKDAYS[t.weekday()]}）。"
        "只根据工具返回的数据回答，不要编造活动；提到活动时用《标题》。"
        "你不能直接替用户报名：需要报名时调用 propose_signup，由用户确认后系统才会执行，"
        "所以不要说「已经帮你报名」。回答简短。"
    )


def _last_human(state: State) -> str:
    return next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")


def build_graph(llm: BaseChatModel, checkpointer=None):
    search_llm = llm.bind_tools(SEARCH_TOOLS)
    signup_llm = llm.bind_tools(SIGNUP_TOOLS)

    def supervisor(state: State) -> Command[Literal["search_agent", "signup_agent", "chat"]]:
        text = _last_human(state)
        route = rules.route(text)
        if route is None:
            # 规则判断不了的才问模型；模型乱答就退回闲聊，不让它决定流程走向之外的事
            reply = llm.invoke([
                SystemMessage("判断用户意图，只回一个词：search（查活动/查我的报名）、"
                              "signup（要报名）、chat（其他）。"),
                HumanMessage(text),
            ]).content.strip().lower()
            route = reply if reply in AGENTS else "chat"
        target = "chat" if route == "chat" else f"{route}_agent"
        return Command(goto=target, update={
            "route": route, "rounds": 0, "proposal": None, "signed_up": False, "problems": []})

    def agent(bound, hint=None):
        def run(state: State) -> dict:
            msgs = [_system(), *([hint] if hint else []), *state["messages"]]
            if state["rounds"] >= MAX_ROUNDS:
                resp = llm.invoke(msgs)  # 到上限就不再给工具，逼它直接回答
            else:
                resp = bound.invoke(msgs)
            return {"messages": [resp], "rounds": state["rounds"] + 1}
        return run

    def after_agent(tools_node: str, done: str = "check"):
        def pick(state: State) -> str:
            last = state["messages"][-1]
            return tools_node if getattr(last, "tool_calls", None) else done
        return pick

    def signup_done(state: State) -> Command[Literal["confirm", "check"]]:
        """模型没提议就收尾时兜底：用户明确要报名、这一轮只查到一个候选，就由代码直接出确认卡。"""
        turn = []
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                break
            turn.append(m)
        found = [json.loads(m.content) for m in turn
                 if isinstance(m, ToolMessage) and m.name == "find_activities"]
        if found and len(found[0]) == 1:  # turn 是倒序的，found[0] 是最后一次查询
            return Command(goto="confirm", update={"proposal": found[0][0]})
        return Command(goto="check")

    signup_tool_node = ToolNode(SIGNUP_TOOLS)

    def signup_tools(state: State, config: RunnableConfig) -> Command[Literal["signup_agent", "confirm"]]:
        out = signup_tool_node.invoke(state, config)
        for m in out["messages"]:
            if m.name == "propose_signup":
                body = json.loads(m.content)
                if "proposal" in body:
                    return Command(goto="confirm", update={**out, "proposal": body["proposal"]})
        return Command(goto="signup_agent", update=out)

    def confirm(state: State, config: RunnableConfig) -> dict:
        p = state["proposal"]
        uid = config["configurable"]["user_id"]
        # 先校验再问用户：报不了的不让用户白点确认
        reason = data.STORE.check(uid, p["id"])
        if reason:
            return {"messages": [AIMessage(f"《{p['title']}》现在报不了：{reason}。")]}
        ok = interrupt({"type": "confirm_signup", "activity": p})
        if not ok:
            return {"messages": [AIMessage(f"好的，《{p['title']}》先不报了。")], "proposal": None}
        # 用户确认到执行之间状态可能变了（名额被抢光），执行时再校验一次
        reason = data.STORE.signup(uid, p["id"])
        if reason:
            return {"messages": [AIMessage(f"没报上《{p['title']}》：{reason}。")], "proposal": None}
        return {"messages": [AIMessage(f"已为你报名《{p['title']}》，{p['date']} 见！")],
                "proposal": None, "signed_up": True}

    def chat(state: State) -> dict:
        return {"messages": [llm.invoke([_system(), *state["messages"]])]}

    def check(state: State) -> dict:
        last = state["messages"][-1]
        grounds = "\n".join(m.content for m in state["messages"] if isinstance(m, ToolMessage))
        cleaned, problems = rules.check_answer(last.content, grounds, state.get("signed_up", False))
        if not cleaned:
            cleaned = "这个我暂时答不上来，换个问法试试？"
        # 同 id 的消息会被 add_messages 替换掉，相当于改写最后一条回答
        return {"messages": [AIMessage(cleaned, id=last.id)], "problems": problems}

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("search_agent", agent(search_llm))
    g.add_node("search_tools", ToolNode(SEARCH_TOOLS))
    g.add_node("signup_agent", agent(signup_llm, SIGNUP_HINT))
    g.add_node("signup_tools", signup_tools)
    g.add_node("signup_done", signup_done)
    g.add_node("confirm", confirm)
    g.add_node("chat", chat)
    g.add_node("check", check)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("search_agent", after_agent("search_tools"), ["search_tools", "check"])
    g.add_edge("search_tools", "search_agent")
    g.add_conditional_edges("signup_agent", after_agent("signup_tools", "signup_done"), ["signup_tools", "signup_done"])
    g.add_edge("confirm", END)
    g.add_edge("chat", "check")
    g.add_edge("check", END)
    return g.compile(checkpointer=checkpointer)
