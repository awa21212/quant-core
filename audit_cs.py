# 审计 v2：用成交流水从期初 100 万复算现金与持仓，与引擎期末净值勾稽
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
import core_satellite as cs

WINDOWS = [("2026YTD", "2026-01-01", "2026-11-06"),
           ("2020-2025", "2020-01-01", "2025-12-31"),
           ("2015", "2015-01-01", "2015-12-31"),
           ("2008", "2008-01-01", "2008-12-31")]

BUY_WORDS = ("建仓", "买入", "加仓", "回补", "补仓")


def is_buy(side: str) -> bool:
    return any(w in side for w in BUY_WORDS)
problems: list[str] = []


def g(t, k):
    return t[k] if isinstance(t, dict) else getattr(t, k)


for tag, a, b in WINDOWS:
    os.environ["CS_AUDIT"] = "1"
    cfg = cs.Config(start=a, end=b)
    res = cs.run(cfg)
    tr, m, es = res["trades"], res["metrics"], res["end_state"]

    cash = es["capital"]
    core = 0.0
    pos: dict[str, float] = defaultdict(float)
    seen: dict[tuple, list] = defaultdict(list)

    for t in tr:
        if "期末平仓" in (g(t, "note") or ""):     # 期末平仓发生在 end_state 快照之后
            continue
        d = pd.Timestamp(g(t, "date")).date()
        code, side, sh, amt, fee = g(t, "code"), g(t, "side"), g(t, "shares"), g(t, "amount"), g(t, "fee")
        seen[(d, code)].append(side)
        if is_buy(side):
            cash -= amt + fee
            if code == cs.CORE_INDEX:
                core += sh
            else:
                pos[code] += sh
        else:                                    # 减仓/卖出/清仓/止损/期末平仓
            cash += amt - fee
            if code == cs.CORE_INDEX:
                core -= sh
            else:
                pos[code] -= sh

    close = es["last_close"]
    lt_val = sum(es["longterm"].get(c, 0) * close.get(c, 0.0) for c in es["longterm"])
    sig_val = sum(es["tactical"].get(c, 0) * close.get(c, 0.0) for c in es["tactical"])
    recap = cash + core * es["index_close"] + lt_val + sig_val

    print(f"[{tag}] 收益 {m['total_return']:+.2f}%｜期末 {m['final_equity']:,.0f}｜"
          f"流水复算 {recap:,.0f}（差 {recap - es['equity']:+,.2f}）")
    if abs(recap - es["equity"]) > 1.0:
        problems.append(f"[{tag}] 流水复算与期末净值差 {recap - es['equity']:+,.2f}")

    if es["reserve"] < -1e-6 or es["sat_cash"] < -1e-6:
        problems.append(f"[{tag}] 期末现金为负 reserve={es['reserve']:.2f} sat={es['sat_cash']:.2f}")

    for c, v in es["longterm"].items():
        if abs(pos.get(c, 0) - v) > 1e-6:
            problems.append(f"[{tag}] 长期 {c} 流水累计 {pos.get(c, 0):.0f} ≠ 期末 {v}")
    for c, v in es["tactical"].items():
        if abs(pos.get(c, 0) - v) > 1e-6:
            problems.append(f"[{tag}] 择时 {c} 流水累计 {pos.get(c, 0):.0f} ≠ 期末 {v}")
    if abs(core - es["core_shares"]) > 1e-6:
        problems.append(f"[{tag}] 核心份数 流水 {core:.2f} ≠ 期末 {es['core_shares']:.2f}")

    for (d, c), sides in seen.items():
        if any(is_buy(s) for s in sides) and any(not is_buy(s) for s in sides):
            problems.append(f"[{tag}] {d} {c} T+1 违规：{sides}")

    ind = Counter(cs.INDUSTRY_MAP.get(c, "其他") for c in es["tactical"])
    for k, v in ind.items():
        if v > cfg.industry_max:
            problems.append(f"[{tag}] 行业 {k} {v} 只 > 上限 {cfg.industry_max}")
    if len(es["tactical"]) > cfg.max_positions:
        problems.append(f"[{tag}] 持仓 {len(es['tactical'])} > 上限 {cfg.max_positions}")

print("\n" + ("❌ 有问题" if problems else "✅ 全部通过（勾稽/股数/T+1/上限）"))
for p in problems:
    print("  -", p)
