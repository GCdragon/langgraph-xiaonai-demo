"""命令行对话：python main.py。遇到报名确认会停下来问 y/n。"""
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from xiaonai.graph import build_graph
from xiaonai.llm import make_llm


def show(result: dict) -> None:
    print("小奈：", result["messages"][-1].content)
    for p in result.get("problems") or []:
        print("  [校验]", p)


def main() -> None:
    graph = build_graph(make_llm(), checkpointer=InMemorySaver())
    # thread_id 区分会话，user_id 由「服务端」注入，模型拿不到也改不了
    config = {"configurable": {"thread_id": uuid.uuid4().hex, "user_id": 1}}
    print("和小奈聊天，输入 q 退出。试试：上海周末有什么活动 / 帮我报名飞盘那个 / 我报了哪些")
    while (text := input("\n你：").strip()) not in ("q", "quit", "exit"):
        if not text:
            continue
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        while "__interrupt__" in result:
            act = result["__interrupt__"][0].value["activity"]
            yes = input(f"  [确认卡] 报名《{act['title']}》（{act['city']} {act['date']}，"
                        f"剩 {act['left']} 个名额）？y/n：").strip().lower() == "y"
            result = graph.invoke(Command(resume=yes), config)
        show(result)


if __name__ == "__main__":
    main()
