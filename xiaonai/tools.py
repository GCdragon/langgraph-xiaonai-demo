"""给模型用的工具。只读工具随便调；写操作只能「提议」，真正执行在确认节点里。"""
import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from . import data


@tool
def find_activities(city: str | None = None, keyword: str | None = None,
                    weekend_only: bool = False) -> str:
    """查询即将开始的活动。city 为城市名，keyword 为标题关键词，weekend_only 只看周末。"""
    return json.dumps(data.STORE.search(city, keyword, weekend_only), ensure_ascii=False)


@tool
def my_signups(config: RunnableConfig) -> str:
    """查询当前用户已报名的活动。"""
    # 用户身份来自服务端 config，不是模型参数，模型没法冒充别人
    uid = config["configurable"]["user_id"]
    return json.dumps(data.STORE.mine(uid), ensure_ascii=False)


@tool
def propose_signup(activity_id: int) -> str:
    """提议为用户报名某个活动。只生成确认请求，不会真的报名，需要用户确认。"""
    a = data.STORE.acts.get(activity_id)
    if a is None:
        return json.dumps({"error": "活动不存在"}, ensure_ascii=False)
    return json.dumps({"proposal": a.brief()}, ensure_ascii=False)


SEARCH_TOOLS = [find_activities, my_signups]
SIGNUP_TOOLS = [find_activities, propose_signup]
