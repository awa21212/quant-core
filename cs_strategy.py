"""策略层：把「今天买什么、买多少」抽成纯函数，不碰账户和现金。

三条腿共用这里：
- 上行·ETF 轮动：动量 > 0 且站上 MA20，取动量最前 K 只
- 下行·防御腿：站上自身趋势闸门 + 20 日波动率最低的 N 只
- 上行·个股：金叉 + KER 过滤（真正挑票仍交给 portfolio.pick_candidates）
"""
from __future__ import annotations

import pandas as pd

CORE_WEIGHT = 0.20        # 核心账本占比（V4.6：0.30→0.20，与 core_bear/lt_bear=zero 一起把 MDD 压到 −20.8%）
                          # 依据：五段窗口全胜（16y 夏普 0.71→0.80；25-26 0.63→0.81；
                          # 三子区间 0.33→0.46 / 1.03→1.23 / 0.43→0.58）。核心是纯 beta，拖累了组合。
LT_SHARE = 0.50           # 卫星账本里「长期股」的占比
MAX_WEIGHT = 0.10         # 单票上限（占总资产）
INDUSTRY_MAX = 3
MAX_POSITIONS = 10
BAND = 0.01
COOLDOWN = 5              # 冷却期（交易日）
CONFIRM = 2               # 连续确认天数
MIN_HOLD = 0              # 持有期下限（交易日）
TRANCHES = 1              # 分批建仓批数：1=一次性打满
TRANCHE_GAP = 3           # 相邻两批至少隔几个交易日
KER_MIN = 0.0


def etf_candidates(aligned: dict, d, codes: list[str], held: set[str], bear_set: set[str],
                   last_exit: dict, pos: int, cooldown: int, trend_ma: int,
                   universe: dict, stats: dict) -> list[dict]:
    """上行期 ETF 轮动候选：动量 > 0、站上 MA20、（可选）站上长期均线。"""
    out = []
    for code in codes:
        if code in held or code in bear_set:
            continue
        row = aligned[code].loc[d]
        if trend_ma and not bool(row.get("above_trend", False)):
            continue
        mom = row.get("mom")
        if pd.isna(mom) or float(mom) <= 0 or not bool(row["above_ma20"]):
            continue
        if code in last_exit and (pos - last_exit[code]) < cooldown:
            stats["skipped_cooldown"] += 1
            continue
        out.append({"code": code, "mom": float(mom),
                    "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                    "industry": universe.get(code, ("ETF", "其他"))[1]})
    out.sort(key=lambda c: -c["mom"])
    return out


def defensive_candidates(aligned: dict, d, pool: dict, held: set[str], last_exit: dict,
                         pos: int, cooldown: int, stats: dict) -> list[dict]:
    """下行期防御资产候选：站上自身趋势闸门、按 20 日波动率升序。"""
    out = []
    for code in pool:
        if code in held or code not in aligned:
            continue
        row = aligned[code].loc[d]
        vol = row.get("vol20")
        if vol is None or pd.isna(vol):
            continue
        if not bool(row.get("above_bear", False)):
            continue
        if code in last_exit and (pos - last_exit[code]) < cooldown:
            stats["skipped_cooldown"] += 1
            continue
        out.append({"code": code, "mom": 0.0, "vol20": float(vol),
                    "vol": float(vol), "industry": pool[code][1]})
    out.sort(key=lambda c: c["vol"])
    return out


def stock_candidates(aligned: dict, d, codes: list[str], held: set[str], bear_set: set[str],
                     last_exit: dict, pos: int, cooldown: int, ker_min: float,
                     industry_map: dict, stats: dict) -> list[dict]:
    """上行期个股候选：金叉（已含 confirm）+ KER 门槛。"""
    out = []
    for code in codes:
        if code in held or code in bear_set:
            continue
        row = aligned[code].loc[d]
        if pd.isna(row["ma20"]) or pd.isna(row["ker14"]):
            continue
        if not bool(row.get("golden_n", False)):
            continue
        if float(row["ker14"]) <= ker_min:
            continue
        if code in last_exit and (pos - last_exit[code]) < cooldown:
            stats["skipped_cooldown"] += 1
            continue
        out.append({"code": code, "ker14": float(row["ker14"]),
                    "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                    "industry": industry_map.get(code, "其他")})
    return out


def equal_weights(codes: list[str], max_weight: float = 1.0, exposure_mult: float = 1.0,
                  defensive: bool = False) -> dict[str, float]:
    """等权分配。

    - 防御腿：不乘 `exposure_mult`（熔断是「进攻受限」，防守期本来就是低风险仓位，
      再打折等于不设防）
    - 其他：等权且受单票上限约束，超出部分留现金
    """
    n = max(1, len(codes))
    if defensive:
        return {c: 1.0 / n for c in codes}
    return {c: min(max_weight, 1.0 / n) * exposure_mult for c in codes}
