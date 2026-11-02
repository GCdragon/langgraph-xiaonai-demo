"""不调真模型：用脚本化的假模型跑通整张图，验证流程和护栏。"""
import uuid
from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from xiaonai import data, rules
from xiaonai.graph import MAX_ROUNDS, build_graph


class ScriptedModel(GenericFakeChatModel):
    """按顺序吐出预设回复；bind_tools 直接返回自己。"""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, i=0):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


@pytest.fixture(autouse=True)
def fresh_store():
    data.STORE = data.Store(date(2026, 11, 2))  # 周一


def session(replies):
    graph = build_graph(ScriptedModel(messages=iter(replies)), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": uuid.uuid4().hex, "user_id": 7}}

    def send(t):
        payload = Command(resume=t) if isinstance(t, bool) else {"messages": [HumanMessage(t)]}
        return graph.invoke(payload, config)
    return send


def run(replies, *turns):
    send = session(replies)
    return [send(t) for t in turns]


def test_signup_waits_for_confirm_then_executes():
    send = session([call("propose_signup", {"activity_id": 1})])
    paused = send("帮我报名飞盘")
    assert paused["__interrupt__"][0].value["activity"]["id"] == 1
    assert 7 not in data.STORE.acts[1].signed  # 停在确认卡，还没报
    done = send(True)
    assert 7 in data.STORE.acts[1].signed
    assert done["signed_up"] is True


def test_decline_does_not_signup():
    _, done = run([call("propose_signup", {"activity_id": 1})], "帮我报名飞盘", False)
    assert 7 not in data.STORE.acts[1].signed
    assert "先不报" in done["messages"][-1].content


def test_slots_taken_between_propose_and_confirm():
    send = session([call("propose_signup", {"activity_id": 5})])
    assert "__interrupt__" in send("帮我报名狼人杀")
    data.STORE.acts[5].signed.add(99)  # 用户犹豫时，最后一个名额被别人抢走
    done = send(True)
    assert 7 not in data.STORE.acts[5].signed
    assert "名额已满" in done["messages"][-1].content and not done["signed_up"]


def test_signup_falls_back_to_code_when_model_only_asks():
    send = session([call("find_activities", {"city": "武汉", "keyword": "狼人杀"}),
                    AIMessage("武汉有《桌游狼人杀之夜》，要报名吗？")])
    paused = send("武汉的狼人杀也帮我报一下")
    assert paused["__interrupt__"][0].value["activity"]["id"] == 5


def test_signup_with_several_candidates_asks_user():
    (r,) = run([call("find_activities", {"city": "上海"}), AIMessage("上海有两个活动，你想报哪个？")],
               "帮我报名上海的活动")
    assert "__interrupt__" not in r


def test_full_activity_is_rejected_before_asking():
    data.STORE.acts[5].signed.add(99)
    (r,) = run([call("propose_signup", {"activity_id": 5})], "帮我报名狼人杀")
    assert "__interrupt__" not in r
    assert "名额已满" in r["messages"][-1].content


def test_tool_rounds_are_capped():
    loops = [call("find_activities", {"city": "上海"}, i) for i in range(MAX_ROUNDS)]
    (r,) = run([*loops, AIMessage("上海有《周末飞盘新手局》。")], "上海有什么活动")
    assert r["rounds"] == MAX_ROUNDS + 1
    assert r["messages"][-1].content == "上海有《周末飞盘新手局》。"


def test_check_drops_fake_done_claim_and_flags_unknown_title():
    (r,) = run([call("find_activities", {"city": "上海"}),
                AIMessage("上海有《周末飞盘新手局》和《不存在的局》。我已经帮你报名了。")],
               "上海有什么活动")
    assert "已经帮你报名" not in r["messages"][-1].content
    assert any("不存在的局" in p for p in r["problems"])


def test_my_signups_uses_server_side_user():
    data.STORE.acts[3].signed.add(7)
    data.STORE.acts[4].signed.add(8)
    (r,) = run([call("my_signups", {}), AIMessage("你报了《西湖徒步半日游》。")], "我报了哪些")
    tool_out = [m for m in r["messages"] if m.type == "tool"][0].content
    assert "西湖徒步" in tool_out and "篮球" not in tool_out


def test_supervisor_falls_back_to_chat_on_garbage():
    (r,) = run(["我觉得应该是 maybe", AIMessage("你好呀")], "今天天气不错")
    assert r["route"] == "chat"


@pytest.mark.parametrize("text,route", [
    ("上海周末有什么活动", "search"), ("帮我报名飞盘", "signup"),
    ("我报了哪些活动", "search"), ("武汉的狼人杀也帮我报一下", "signup"), ("你是谁", None),
])
def test_rules_route(text, route):
    assert rules.route(text) == route


@pytest.mark.parametrize("text,dropped", [
    ("我已经帮你报名了。", True), ("已为你成功报名《周末飞盘新手局》。", True),
    ("你已报名《周末飞盘新手局》。", False),
])
def test_done_claim_only_blocks_acting_for_user(text, dropped):
    cleaned, _ = rules.check_answer(text, "周末飞盘新手局", signed_up=False)
    assert (cleaned == "") is dropped


def test_search_skips_past_and_filters_weekend():
    titles = [a["title"] for a in data.STORE.search("上海", weekend_only=True)]
    assert titles == ["周末飞盘新手局"]
