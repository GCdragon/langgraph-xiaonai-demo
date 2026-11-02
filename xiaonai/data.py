"""模拟业务数据与业务规则。真实项目里这一层是 Java Service，这里用内存数据代替。"""
from dataclasses import dataclass, field
from datetime import date, timedelta


WEEKDAYS = "周一 周二 周三 周四 周五 周六 周日".split()


@dataclass
class Activity:
    id: int
    title: str
    city: str
    day: date
    slots: int
    signed: set[int] = field(default_factory=set)

    def left(self) -> int:
        return self.slots - len(self.signed)

    def brief(self) -> dict:
        # 星期几由代码给出，不让模型自己推算
        return {"id": self.id, "title": self.title, "city": self.city,
                "date": self.day.isoformat(), "weekday": WEEKDAYS[self.day.weekday()],
                "left": self.left()}


def _next_weekday(today: date, weekday: int) -> date:
    return today + timedelta(days=(weekday - today.weekday()) % 7 or 7)


def seed(today: date | None = None) -> dict[int, Activity]:
    t = today or date.today()
    sat, sun = _next_weekday(t, 5), _next_weekday(t, 6)
    rows = [
        Activity(1, "周末飞盘新手局", "上海", sat, 12),
        Activity(2, "滨江夜跑 5 公里", "上海", t + timedelta(days=2), 20),
        Activity(3, "西湖徒步半日游", "杭州", sun, 15),
        Activity(4, "篮球新人友好局", "武汉", sat, 10),
        Activity(5, "桌游狼人杀之夜", "武汉", sun, 1),
        Activity(6, "已经结束的读书会", "上海", t - timedelta(days=3), 8),
    ]
    return {a.id: a for a in rows}


class Store:
    def __init__(self, today: date | None = None):
        self.today = today or date.today()
        self.acts = seed(self.today)

    def search(self, city: str | None = None, keyword: str | None = None,
               weekend_only: bool = False) -> list[dict]:
        out = []
        for a in self.acts.values():
            if a.day < self.today:
                continue
            if city and a.city != city:
                continue
            if keyword and keyword not in a.title:
                continue
            if weekend_only and a.day.weekday() < 5:
                continue
            out.append(a.brief())
        return out

    def mine(self, user_id: int) -> list[dict]:
        return [a.brief() for a in self.acts.values() if user_id in a.signed]

    def check(self, user_id: int, act_id: int) -> str | None:
        """能不能报名；返回不能报的原因，能报返回 None。执行前必须再查一次，不信任提议时的状态。"""
        a = self.acts.get(act_id)
        if a is None:
            return "活动不存在"
        if a.day < self.today:
            return "活动已结束"
        if user_id in a.signed:
            return "你已经报过名了"
        if a.left() <= 0:
            return "名额已满"
        return None

    def signup(self, user_id: int, act_id: int) -> str | None:
        reason = self.check(user_id, act_id)
        if reason is None:
            self.acts[act_id].signed.add(user_id)
        return reason


STORE = Store()
