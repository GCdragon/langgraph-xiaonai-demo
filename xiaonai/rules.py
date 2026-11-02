"""确定性规则：能用代码判断的，不交给模型。"""
import re

ASK_ACTIVITY = re.compile(r"(有|找|推荐).{0,6}(活动|局)")
ASK_SIGNUP = re.compile(r"(帮我|给我|我要|我想).{0,6}报(名|一下|个)|报名.{0,6}《")
ASK_MINE = re.compile(r"我(报|报名)了(哪些|什么)|我的报名")

# 只拦「我替你办完了」这种说法；「你已报名 X」是查询结果，不拦
DONE_CLAIM = re.compile(r"(已|已经)(成功)?(帮你|为你|给你)(成功)?报(好)?名")
TITLE = re.compile(r"《([^》]{1,40})》")
SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?\n]?")


def route(text: str) -> str | None:
    """高置信意图直接路由；返回 None 表示交给 supervisor 模型判断。"""
    if ASK_MINE.search(text):
        return "search"
    if ASK_SIGNUP.search(text):
        return "signup"
    if ASK_ACTIVITY.search(text):
        return "search"
    return None


def check_answer(answer: str, grounds: str, signed_up: bool) -> tuple[str, list[str]]:
    """逐句检查回答：没真正报名却说「已报名」的句子删掉；引用的《标题》不在依据里的记下来。"""
    kept, problems = [], []
    for s in SENTENCE.findall(answer):
        if not signed_up and DONE_CLAIM.search(s):
            problems.append(f"已拦下：{s.strip()}")
            continue
        kept.append(s)
    for t in TITLE.findall(answer):
        if t not in grounds:
            problems.append(f"标题对不上：《{t}》")
    return "".join(kept).strip(), problems
