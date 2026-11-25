# -*- coding: utf-8 -*-
"""v5_portfolio.py —— P3 组合层（截面选股 + 约束 + 成本）。

规则
----
- 打分：`0.5*低波动 + 0.5*短期反转`（两者都已在截面内标准化为 [-1,1]，越大越好）
- 过滤：上市 ≥60 交易日、价格 ≥1 元、20 日均成交额 ≥ 2000 万、非 ST
- 选股：按分数从高到低贪心，直到 **50 只**；单票 **≤2%**；行业权重不超过「基准行业权重 +5pp」
- 调仓：每 **20 个交易日** 一次；先卖后买
- 成本：佣金万 2.5（最低 5 元）+ 印花 0.05%（仅卖）+ 过户费 0.001% + 滑点 2bp

用法
----
    python v5_portfolio.py --start 2016-01-01 --end 2026-11-10
    python v5_portfolio.py --rebal 40 --n-hold 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v5_config as C  # noqa: E402
import v5_data as D  # noqa: E402
import v5_factors as F  # noqa: E402


def _fee_buy(amount: float, on: bool = True) -> float:
    if not on:
        return 0.0
    return max(amount * C.COMMISSION, C.COMMISSION_MIN) + amount * C.TRANSFER


def _fee_sell(amount: float, on: bool = True) -> float:
    if not on:
        return 0.0
    return (max(amount * C.COMMISSION, C.COMMISSION_MIN)
            + amount * C.STAMP + amount * C.TRANSFER)


def _impact(amount: float, adv: float, coef: float) -> float:
    """冲击成本 = 成交额 × coef × (成交额/20日均额)，adv 无效则 0。"""
    if coef <= 0 or adv is None or not np.isfinite(adv) or adv <= 0:
        return 0.0
    return amount * coef * min(amount / adv, 1.0)


def eligible_mask(close: pd.DataFrame, amount: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    """可交易性过滤（逐日）。"""
    ok = close.notna() & (close >= 1.0)
    if amount is not None:
        adv20 = amount.rolling(20, min_periods=10).mean()
        ok &= adv20.notna() & (adv20 >= 2e7)
    listed = close.notna().cumsum()
    ok &= listed >= C.MIN_LIST_DAYS
    st = {c for c, n in name_map.items() if isinstance(n, str) and ("ST" in n.upper())}
    if st:
        ok.loc[:, [c for c in st if c in ok.columns]] = False
    return ok


def run(start: str, end: str, n_hold: int = C.N_HOLD, rebal: int = C.REBAL_DAYS,
        capital: float = 1e6, costs: bool = True, max_w: float = C.MAX_W,
        ind_dev: float = C.IND_DEV, verbose: bool = True,
        impact_coef: float = 0.0, impact_max_part: float = 0.05, limits: bool = True) -> dict:
    t0 = time.time()
    panel = F.load_panel(start=start, end=end)
    fac = F.build_factors(panel)
    close, adj = panel["close"], panel["adj_close"]
    amount = panel.get("amount")
    volume = panel.get("volume")
    ret = close.pct_change()
    adv20 = amount.rolling(20, min_periods=10).mean() if amount is not None else None
    halted = ((volume <= 0) | close.isna()) if volume is not None else close.isna()
    limit_up = (ret >= 0.099) & close.notna()
    limit_dn = (ret <= -0.099) & close.notna()
    stats = {"skipped_halt": 0, "skipped_limit": 0, "skipped_t1": 0, "impact": 0.0}
    score = 0.5 * fac["f_lowvol"].fillna(0) + 0.5 * fac["f_rev20"].fillna(0)
    score = score.where(fac["f_lowvol"].notna() & fac["f_rev20"].notna())

    if os.environ.get("V5_RANDOM") == "1":      # 对照实验：随机选股
        rng = np.random.default_rng(int(os.environ.get("V5_SEED","7")))
        score = pd.DataFrame(rng.normal(size=score.shape), index=score.index, columns=score.columns)

    uni = D.universe()
    name_map = dict(zip(uni["code"], uni["name"]))
    ind = pd.read_csv(os.path.join(C.META, "industry_coarse.csv"), dtype={"code": str})
    ind_map = dict(zip(ind["code"], ind["industry_coarse"]))

    ok = eligible_mask(close, amount, name_map)
    score = score.where(ok)

    dates = list(close.index)
    # 基准行业权重：用当日可交易股票的等权分布
    codes_all = list(close.columns)

    cash = capital
    pos: dict[str, dict] = {}   # code -> {shares, cost}
    eq_curve, turnover_hist = [], []
    trades = 0
    fees_total = 0.0

    def industry_cap(day: str) -> dict:
        m = ok.loc[day]
        live = [c for c in codes_all if bool(m.get(c, False))]
        if not live:
            return {}
        cnt: dict[str, int] = {}
        for c in live:
            cnt[ind_map.get(c, "其他")] = cnt.get(ind_map.get(c, "其他"), 0) + 1
        total = len(live)
        return {k: v / total + ind_dev for k, v in cnt.items()}

    for i, day in enumerate(dates):
        # 每日估值
        eq = cash + sum(h["shares"] * adj.at[day, c] for c, h in pos.items()
                        if c in adj.columns and not np.isnan(adj.at[day, c]))
        eq_curve.append((day, eq))

        if i % rebal != 0 or i < 60:
            continue

        caps = industry_cap(day)
        sc = score.loc[day].dropna().sort_values(ascending=False)
        if not len(sc):
            continue
        # 缓冲带：老持仓只要还在前 80 名就保留，避免排名抖动导致整仓换手
        keep_n = int(n_hold * 1.6)
        rank_map = {c: r for r, c in enumerate(sc.index)}
        target: dict[str, float] = {}
        ind_w: dict[str, float] = {}
        for c in pos:                                    # 先保住老仓位
            if c in sc.index and rank_map[c] < keep_n:
                k = ind_map.get(c, "其他")
                if ind_w.get(k, 0) + max_w <= caps.get(k, 1.0) + 1e-9:
                    target[c] = max_w
                    ind_w[k] = ind_w.get(k, 0) + max_w
        for c in sc.index:                               # 再补新面孔
            if len(target) >= n_hold:
                break
            if c in target:
                continue
            k = ind_map.get(c, "其他")
            if ind_w.get(k, 0) + max_w > caps.get(k, 1.0) + 1e-9:
                continue
            target[c] = max_w
            ind_w[k] = ind_w.get(k, 0) + max_w
        if len(target) < n_hold and target:               # 名额没用满就按比例放大
            scale = min(1.0, (n_hold * max_w) / (len(target) * max_w))
            if scale < 1.0:
                target = {c: w for c, w in target.items()}

        # 先卖
        for c in list(pos):
            if c in target:
                continue
            px = adj.at[day, c]
            if np.isnan(px) or px <= 0:
                continue
            if limits and bool(halted.at[day, c]):
                stats["skipped_halt"] += 1
                continue
            if limits and bool(limit_dn.at[day, c]):
                stats["skipped_limit"] += 1
                continue
            if pos[c].get("buy_day") == day:
                stats["skipped_t1"] += 1
                continue
            amt = pos[c]["shares"] * px * (1 - C.SLIPPAGE)
            imp = 0.0
            if impact_coef > 0 and adv20 is not None:
                imp = _impact(amt, adv20.at[day, c], impact_coef)
            stats["impact"] += imp
            cash += amt - _fee_sell(amt, costs) - imp
            fees_total += _fee_sell(amt, costs) + imp
            trades += 1
            del pos[c]
        # 再买/加
        for c, w in target.items():
            px = adj.at[day, c]
            if np.isnan(px) or px <= 0:
                continue
            cur_val = pos[c]["shares"] * px if c in pos else 0.0
            want_val = eq * w
            gap = want_val - cur_val
            # 只在「新建仓」或「偏离超过 40% 目标」时才动手，压住换手
            if c in pos and abs(gap) < want_val * 0.4:
                continue
            if gap <= eq * 0.001:
                continue
            if limits and bool(halted.at[day, c]):
                stats["skipped_halt"] += 1
                continue
            if limits and bool(limit_up.at[day, c]):
                stats["skipped_limit"] += 1
                continue
            shares = int(gap / px / C.LOT) * C.LOT
            if shares <= 0:
                continue
            if impact_coef > 0 and adv20 is not None:      # 单笔不超过 ADV 的 x%
                advv = adv20.at[day, c]
                if np.isfinite(advv) and advv > 0:
                    cap_sh = int(advv * impact_max_part / px / C.LOT) * C.LOT
                    shares = min(shares, cap_sh)
                    if shares <= 0:
                        continue
            amt = shares * px * (1 + C.SLIPPAGE)
            fee = _fee_buy(amt, costs)
            if amt + fee > cash:
                shares = int((cash - fee) / (px * (1 + C.SLIPPAGE)) / C.LOT) * C.LOT
                if shares <= 0:
                    continue
                amt = shares * px * (1 + C.SLIPPAGE)
                fee = _fee_buy(amt, costs)
            imp = 0.0
            if impact_coef > 0 and adv20 is not None:
                imp = _impact(amt, adv20.at[day, c], impact_coef)
            stats["impact"] += imp
            cash -= amt + fee + imp
            fees_total += fee + imp
            trades += 1
            if c in pos:
                pos[c]["shares"] += shares
            else:
                pos[c] = {"shares": shares, "cost": px, "buy_day": day}

    eq_s = pd.Series(dict(eq_curve)).sort_index().astype(float)
    ret = eq_s.pct_change().dropna()
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    total = eq_s.iloc[-1] / capital - 1
    cagr = (eq_s.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (eq_s / eq_s.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(243) if ret.std() else np.nan
    calmar = cagr / abs(mdd) if mdd else np.nan

    # 基准：全体可交易股票的等权买入持有
    # 基准：期初等权买入并持有（不日频再平衡，避免「再平衡红利」虚高）
    adj0 = adj.ffill()
    first_ok = ok.ffill().iloc[60] if len(ok) > 60 else ok.iloc[0]
    cols = [c for c in adj0.columns if bool(first_ok.get(c, False))]
    if cols:
        norm = adj0[cols].div(adj0[cols].iloc[60] if len(adj0) > 60 else adj0[cols].iloc[0])
        bench = norm.mean(axis=1)
    else:
        bench = pd.Series(1.0, index=adj.index)

    res = {
        "start": start, "end": end, "n_days": len(eq_s), "trades": trades,
        "final": eq_s.iloc[-1], "total_return": total * 100, "cagr": cagr * 100,
        "mdd": mdd * 100, "sharpe": sharpe, "calmar": calmar,
        "fees": fees_total, "fee_pct": fees_total / capital * 100,
        "bench_total": (bench.iloc[-1] - 1) * 100, "eq": eq_s, "bench": bench,
        "skipped_halt": stats["skipped_halt"], "skipped_limit": stats["skipped_limit"],
        "skipped_t1": stats["skipped_t1"], "impact_cost": stats["impact"],
    }
    if verbose:
        print(f"[pf] {start}~{end}  {len(eq_s)} 交易日  {time.time()-t0:.0f}s", flush=True)
        print(f"[pf] 期末 {res['final']:,.2f}｜收益 {res['total_return']:+.2f}%｜"
              f"年化 {res['cagr']:+.2f}%｜回撤 {res['mdd']:.2f}%｜夏普 {res['sharpe']:.2f}｜"
              f"卡玛 {res['calmar']:.2f}｜交易 {trades} 笔｜费用 {fees_total:,.0f}（{res['fee_pct']:.2f}%）", flush=True)
        print(f"[pf] 拦截：停牌 {stats['skipped_halt']}｜涨跌停 {stats['skipped_limit']}｜T+1 {stats['skipped_t1']}｜"
              f"冲击成本 {stats['impact']:,.0f} 元", flush=True)
        print(f"[pf] 等权基准 {res['bench_total']:+.2f}%｜超额 {res['total_return']-res['bench_total']:+.2f}pp", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-11-10")
    ap.add_argument("--n-hold", type=int, default=C.N_HOLD)
    ap.add_argument("--rebal", type=int, default=C.REBAL_DAYS)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--no-costs", action="store_true")
    ap.add_argument("--max-w", type=float, default=C.MAX_W)
    ap.add_argument("--ind-dev", type=float, default=C.IND_DEV)
    ap.add_argument("--impact-coef", type=float, default=0.0, help="冲击成本系数（0=关）")
    ap.add_argument("--impact-max-part", type=float, default=0.05, help="单笔最多占 20 日均额比例")
    ap.add_argument("--no-limits", action="store_true", help="关闭涨跌停/停牌/T+1 拦截")
    a = ap.parse_args(argv)
    run(a.start, a.end, a.n_hold, a.rebal, a.capital, not a.no_costs, a.max_w, a.ind_dev,
        impact_coef=a.impact_coef, impact_max_part=a.impact_max_part, limits=not a.no_limits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
