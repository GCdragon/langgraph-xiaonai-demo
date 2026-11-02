import os

from langchain_openai import ChatOpenAI


def make_llm() -> ChatOpenAI:
    """通义千问走 DashScope 的 OpenAI 兼容接口，换模型只改这里。"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("请先设置环境变量 DASHSCOPE_API_KEY")
    return ChatOpenAI(
        model=os.environ.get("XIAONAI_MODEL", "qwen-plus"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=key,
        temperature=0.3,
    )
