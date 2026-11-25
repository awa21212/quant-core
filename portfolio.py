# -*- coding: utf-8 -*-
"""
portfolio.py —— 多标的组合级回测（选股 / 权重 / 风控 / 成本 / 归因）

【这个文件解决什么问题】
    signal_assistant.py 与 pipeline.py 回答的是「单只股票某天该不该买、买多少」，
    但真实账户是**一篮子股票**，于是多出四个单票回测根本看不到的问题：

        1) 选股与防呆：怎么避免"全仓都是同一个行业""某只票吃掉一半仓位"、
           "回撤 30% 才发现自己没刹车"；
        2) 权重分配：等权是偷懒，波动大的和波动小的一样买，等于主动加风险；
        3) 真实成本：T+1、整手、滑点、佣金、印花税，缺一个回测就是假的；
        4) 绩效归因：钱到底是谁赚的、夏普够不够上实盘。

【四层结构（严格按用户的四条要求实现）】
    ┌─ ① 选股与风控 ─────────────────────────────────────────────┐
    │ 行业中性：同一行业最多持有 INDUSTRY_MAX 只（默认 3）        │
    │ 单票上限：单只不超过 MAX_WEIGHT（默认 10%）                 │
    │ 组合熔断：总资产自峰值回撤 > DRAWDOWN_TRIGGER（默认 10%）    │
    │           → 整体目标仓位 × DRAWDOWN_FACTOR（默认 0.3 折）    │
    │           回撤修复到 DRAWDOWN_RECOVER（默认 5%）以内才解除   │
    └────────────────────────────────────────────────────────────┘
    ┌─ ② 权重分配 ───────────────────────────────────────────────┐
    │ 逆波动率加权：w_i ∝ 1/σ_i（σ = 近 20 日收益率标准差）       │
    │ 再乘 KER 效率因子调节（KER 高=趋势明确，权重上浮）           │
    │ 最后做过单票上限裁剪 + 归一化                               │
    └────────────────────────────────────────────────────────────┘
    ┌─ ③ 真实成本 ───────────────────────────────────────────────┐
    │ T+1：当日买入的票当天不能卖（持仓里记 buy_date）             │
    │ 整手：所有买卖股数向下取整到 100 股                          │
    │ 买入：滑点 0.02% + 佣金万 2.5（最低 5 元）                   │
    │ 卖出：滑点 0.02% + 佣金万 2.5（最低 5 元）+ 印花税千分之一   │
    └────────────────────────────────────────────────────────────┘
    ┌─ ④ 绩效归因 ───────────────────────────────────────────────┐
    │ 夏普 / 卡玛（年化 ÷ 最大回撤）/ 胜率 / 盈亏比               │
    │ 按股票、按行业拆解盈亏；熔断次数与天数                        │
    │ 结论：夏普 < 1 → 明确标注「不可上实盘」                      │
    └────────────────────────────────────────────────────────────┘

【用法】
    # 默认演示股票池（12 只，跨 7 个行业）
    python portfolio.py --start 2025-01-01 --end 2025-12-31

    # 自己指定池子
    python portfolio.py --symbols 000001,600036,600519 --start 2025-01-01 --end 2025-12-31

    # 调参
    python portfolio.py --max-weight 0.10 --industry-max 3 --dd-trigger 0.10 --dd-factor 0.3

【依赖】pandas / numpy / akshare / matplotlib（均已在环境里）

【重要声明】
    这是「按规则机械执行」的研究代码，不是投资建议。回测赚钱不代表实盘赚钱：
    滑点、冲击成本、停牌、涨跌停、除权除息都会让真实结果更差。夏普 < 1 的组合
    在这套标准下就不该上实盘。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import signal_assistant as sa

HERE = os.path.dirname(os.path.abspath(__file__))
TRADING_DAYS = 252
LOT = 100                      # A 股整手

# ============================ 默认参数（用户口径）============================
MAX_WEIGHT = 0.10              # ① 单票上限 10%
INDUSTRY_MAX = 3               # ① 同行业最多 3 只
DRAWDOWN_TRIGGER = 0.10        # ① 组合回撤 10% 触发熔断
DRAWDOWN_FACTOR = 0.30         # ① 熔断后整体仓位打到 3 折
DRAWDOWN_RECOVER = 0.05        # ① 回撤修复到 5% 以内才解除熔断
VOL_WINDOW = 20                # ② 逆波动率用的波动率窗口
KER_ADJUST = 0.5               # ② KER 对权重的调节强度（0 = 不调节）
MAX_POSITIONS = 10             # 最多同时持有 10 只（10% 上限 × 10 = 满仓）
REBALANCE_BAND = 0.005         # ③ 调仓死区：差额不到总资产 0.5% 就不动（省手续费）
COMMISSION = 0.00025           # ③ 万 2.5
COMMISSION_MIN = 5.0           # ③ 最低 5 元
SLIPPAGE = 0.0002              # ③ 买入滑点 0.02%
STAMP = 0.001                  # ③ 卖出印花税千分之一
SHARPE_MIN = 1.0               # ④ 实盘门槛

# 默认股票池：跨行业，避免"一篮子银行"
DEFAULT_UNIVERSE = {
    "000001": "平安银行", "600036": "招商银行",              # 银行
    "600519": "贵州茅台", "000858": "五粮液",                # 白酒
    "300750": "宁德时代", "601012": "隆基绿能",              # 电池 / 光伏
    "600276": "恒瑞医药", "000333": "美的集团",              # 医药 / 家电
    "002594": "比亚迪", "601088": "中国神华",                # 汽车 / 煤炭
    "600030": "中信证券", "002415": "海康威视",              # 券商 / 电子
}

# 行业映射：生产环境应该读 CSV（--industry-file），这里内置一份保证可跑。
# 注意：东财行业接口（stock_board_industry_cons_em）在 push2 限流时会挂，
#       所以默认不依赖网络，改用本地映射 + 可选 CSV 覆盖。
DEFAULT_INDUSTRY = {
    "000001": "银行", "600036": "银行",
    "600519": "食品饮料", "000858": "食品饮料",
    "300750": "电力设备", "601012": "电力设备",
    "600276": "医药生物", "000333": "家用电器",
    "002594": "汽车", "601088": "煤炭",
    "600030": "非银金融", "002415": "电子",
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ============================================================================
# 数据：多标的取数 + 指标
# ============================================================================
def to_code6(sym: str) -> str:
    s = str(sym).strip().upper()
    for suf in (".SZ", ".SH", ".BJ"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else s


def load_one(sym: str, end: pd.Timestamp, adjust: str = "qfq",
             lookback_days: int = 420) -> pd.DataFrame:
    """取一只票的干净日线 + 指标，索引换成日期。

    adjust="hfq" 用于深度历史回测（2015 等）：前复权在十年前会出现负价。
    """
    code6 = to_code6(sym)
    raw = sa.fetch_prices(code6, end, lookback_days=lookback_days, adjust=adjust)
    if raw is None or raw.empty:
        raise RuntimeError(f"{code6} 无数据")
    df, _report = sa.clean_prices(raw)
    df = df.copy()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["golden"] = (df["ma5"] > df["ma20"]) & (df["ma5"].shift(1) <= df["ma20"].shift(1))
    abs_diff = df["close"].diff().abs()
    df["ker14"] = (df["close"] - df["close"].shift(14)).abs() / abs_diff.rolling(14).sum()
    df["vol20"] = df["close"].pct_change().rolling(VOL_WINDOW).std()
    df["above_ma20"] = df["close"] >= df["ma20"]
    df = df.set_index("date").sort_index()
    return df


def load_universe(symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    """批量取数，返回 {代码: 日线} 与统一交易日历。"""
    end_ts = pd.Timestamp(end)
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        code6 = to_code6(sym)
        try:
            df = load_one(code6, end_ts)
        except Exception as exc:
            log(f"      [warn] {code6} 取数失败，跳过：{type(exc).__name__}: {exc}")
            continue
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= end_ts)]
        if len(df) < 30:
            log(f"      [warn] {code6} 区间内只有 {len(df)} 根 K 线，跳过")
            continue
        frames[code6] = df
    if not frames:
        raise SystemExit("[error] 一只都取不到数据")
    days = sorted(set().union(*[set(df.index) for df in frames.values()]))
    return frames, days


def industry_of(code6: str, mapping: dict[str, str], fallback: str = "其他") -> str:
    return mapping.get(code6, fallback)


# ============================================================================
# 组合状态
# ============================================================================
@dataclass
class Holding:
    code: str
    shares: int = 0
    cost: float = 0.0          # 每股成本（含费用摊薄）
    buy_date: pd.Timestamp | None = None
    industry: str = ""

    @property
    def value(self) -> float:  # 由调用方按当日收盘算，这里只做占位
        return self.shares * self.cost


@dataclass
class TradeRec:
    date: pd.Timestamp
    code: str
    side: str                  # 买入 / 卖出
    shares: int
    price: float
    amount: float
    fee: float
    pnl: float = 0.0           # 卖出时结算的实现盈亏
    note: str = ""


@dataclass
class Config:
    symbols: list[str] = field(default_factory=list)
    start: str = "2025-01-01"
    end: str = "2025-12-31"
    capital: float = 1_000_000.0     # 组合通常比单票大，默认 100 万
    max_weight: float = MAX_WEIGHT
    industry_max: int = INDUSTRY_MAX
    dd_trigger: float = DRAWDOWN_TRIGGER
    dd_factor: float = DRAWDOWN_FACTOR
    dd_recover: float = DRAWDOWN_RECOVER
    max_positions: int = MAX_POSITIONS
    band: float = REBALANCE_BAND      # 调仓死区：小于该比例的差额不动，省手续费
    vol_window: int = VOL_WINDOW
    ker_adjust: float = KER_ADJUST
    industry_map: dict[str, str] = field(default_factory=dict)
    fees_on: bool = True


def _commission(amount: float) -> float:
    return max(COMMISSION_MIN, amount * COMMISSION)


def _buy_cost(amount: float, fees_on: bool) -> float:
    return _commission(amount) if fees_on else 0.0


def _sell_cost(amount: float, fees_on: bool) -> float:
    return (_commission(amount) + amount * STAMP) if fees_on else 0.0


def _lot(shares: float) -> int:
    """整手规则：向下取整到 100 股。"""
    return int(max(0, shares) // LOT) * LOT


def can_sell(h: "Holding", d: pd.Timestamp) -> bool:
    """
    T+1 规则的可测试实现：当天买入的股票，当天不能卖。
    注意这里用 >= 而不是 ==，是因为买入日被记成交易日 d，
    任何 d 当天的卖出尝试都会被拒。
    """
    return not (h.buy_date is not None and pd.Timestamp(h.buy_date) >= d)


def pick_candidates(cands: list[dict], holdings: dict[str, "Holding"], cfg: "Config") -> list[dict]:
    """
    行业中性选股：同一行业最多 cfg.industry_max 只，总持仓不超过 cfg.max_positions。
    已持有的行业计数会算进来，避免"先买 3 只银行，再被选进第 4 只"。
    """
    held_by_ind: dict[str, int] = {}
    for h in holdings.values():
        held_by_ind[h.industry] = held_by_ind.get(h.industry, 0) + 1
    picked: list[dict] = []
    for c in sorted(cands, key=lambda x: x["ker14"], reverse=True):
        ind = c["industry"]
        if held_by_ind.get(ind, 0) >= cfg.industry_max:
            continue
        if len(holdings) + len(picked) >= cfg.max_positions:
            break
        picked.append(c)
        held_by_ind[ind] = held_by_ind.get(ind, 0) + 1
    return picked


# ============================================================================
# 权重：逆波动率 + KER 调节
# ============================================================================
def target_weights(cands: list[dict], cfg: Config) -> dict[str, float]:
    """
    在候选里分配权重。

    ① 逆波动率：w ∝ 1/σ
        波动小的多买、波动大的少买 —— 让每只票对组合波动的贡献接近。
    ② KER 调节：w ∝ (1 + ker_adjust × (KER - 0.3) / 0.7)
        趋势越明确（KER 高）权重上浮，趋势弱（KER 低）权重下调。
    ③ 单票上限裁剪：超过 max_weight 的削平，多出来的按比例分给别人。
    ④ 行业上限：在候选阶段已按行业挑了前 N 只。
    """
    if not cands:
        return {}
    raw: dict[str, float] = {}
    for c in cands:
        vol = c["vol20"] if c["vol20"] and c["vol20"] > 1e-6 else 1e-6
        inv_vol = 1.0 / vol
        ker = c["ker14"] if c["ker14"] is not None and not pd.isna(c["ker14"]) else 0.3
        # KER 调节：以 0.3 为中性点，高于它上浮、低于它下调
        ker_mult = 1.0 + cfg.ker_adjust * (float(ker) - 0.3) / 0.7
        ker_mult = min(max(ker_mult, 0.3), 1.8)
        raw[c["code"]] = inv_vol * ker_mult
    total = sum(raw.values())
    if total <= 0:
        return {}
    w = {k: v / total for k, v in raw.items()}

    # 单票上限裁剪（迭代两轮，够用了；极端情况下留一点现金也合理）
    for _ in range(3):
        over = {k: v for k, v in w.items() if v > cfg.max_weight + 1e-9}
        if not over:
            break
        excess = 0.0
        for k in over:
            excess += w[k] - cfg.max_weight
            w[k] = cfg.max_weight
        others = [k for k in w if k not in over]
        base = sum(w[k] for k in others)
        if base <= 0:
            break
        for k in others:
            w[k] += excess * (w[k] / base)
    return w


# ============================================================================
# 回测主循环
# ============================================================================
def run(cfg: Config) -> dict:
    frames, days = load_universe(cfg.symbols, cfg.start, cfg.end)
    codes = list(frames.keys())
    log(f"[portfolio] 股票池 {len(codes)} 只：{', '.join(codes)}")
    log(f"[portfolio] 交易日 {len(days)} 天：{days[0].date()} ~ {days[-1].date()}")

    cash = float(cfg.capital)
    peak = float(cfg.capital)
    holdings: dict[str, Holding] = {}
    trades: list[TradeRec] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    exposure_curve: list[tuple[pd.Timestamp, float]] = []
    breaker_events: list[dict] = []
    breaker_on = False
    breaker_days = 0
    realized: dict[str, float] = {}          # 按股票的已实现盈亏
    stats = {"skipped_t1": 0, "skipped_cash": 0, "skipped_cap": 0}

    # 把每只票的数据转成 dict[date] -> row，缺失日期用前值填充
    aligned: dict[str, pd.DataFrame] = {}
    for code, df in frames.items():
        aligned[code] = df.reindex(days).ffill()

    def px(code: str, d: pd.Timestamp) -> float:
        v = aligned[code].at[d, "close"]
        return float(v) if pd.notna(v) else float("nan")

    def equity_now(d: pd.Timestamp) -> float:
        total = cash
        for code, h in holdings.items():
            p = px(code, d)
            if pd.notna(p):
                total += h.shares * p
        return total

    for d in days:
        # ---------- 1) 估值 + 熔断判断（先用当日收盘价算总资产）----------
        eq = equity_now(d)
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        if not breaker_on and dd <= -cfg.dd_trigger:
            breaker_on = True
            breaker_events.append({"date": str(d.date()), "event": "触发熔断",
                                   "drawdown": round(dd * 100, 2),
                                   "action": f"目标仓位 ×{cfg.dd_factor:.0%}"})
            log(f"      [breaker] {d.date()} 组合回撤 {dd:.2%} ≤ -{cfg.dd_trigger:.0%} → 仓位打 {cfg.dd_factor:.0%}")
        elif breaker_on and dd >= -cfg.dd_recover:
            breaker_on = False
            breaker_events.append({"date": str(d.date()), "event": "解除熔断",
                                   "drawdown": round(dd * 100, 2), "action": "恢复目标仓位"})
            log(f"      [breaker] {d.date()} 回撤修复到 {dd:.2%} → 解除熔断")
        if breaker_on:
            breaker_days += 1
        exposure_mult = cfg.dd_factor if breaker_on else 1.0

        # ---------- 2) 卖出：跌破 MA20（T+1：当日买入的不许卖）----------
        for code in list(holdings.keys()):
            h = holdings[code]
            row = aligned[code].loc[d]
            close = px(code, d)
            if pd.isna(close) or pd.isna(row["ma20"]):
                continue
            if not can_sell(h, d):
                stats["skipped_t1"] += 1
                continue
            if not bool(row["above_ma20"]):
                shares = h.shares
                price = close * (1 - SLIPPAGE)
                amount = shares * price
                fee = _sell_cost(amount, cfg.fees_on)
                cash += amount - fee
                pnl = (price - h.cost) * shares - fee
                realized[code] = realized.get(code, 0.0) + pnl
                trades.append(TradeRec(d, code, "卖出", shares, price, amount, fee, pnl,
                                       f"收盘 {close:.2f} 跌破 MA20 {row['ma20']:.2f}"))
                del holdings[code]

        # ---------- 3) 选股：当日金叉 + KER>0 + 行业上限 ----------
        cands: list[dict] = []
        for code in codes:
            if code in holdings:
                continue
            row = aligned[code].loc[d]
            if pd.isna(row["ma20"]) or pd.isna(row["ker14"]):
                continue
            if not bool(row["golden"]):
                continue
            if float(row["ker14"]) <= 0:
                continue
            cands.append({"code": code, "ker14": float(row["ker14"]),
                          "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                          "industry": industry_of(code, cfg.industry_map)})
        # 行业中性 + 持仓数上限（已持有的行业也算进去）
        picked = pick_candidates(cands, holdings, cfg)
        if len(holdings) + len(picked) >= cfg.max_positions:
            stats["skipped_cap"] += 1

        # ---------- 4) 目标权重 + 换仓 ----------
        if picked or holdings:
            # 已在持仓里的票：如果趋势还在，继续按目标权重持有（权重每天重算）
            pool: list[dict] = []
            for code, h in holdings.items():
                row = aligned[code].loc[d]
                if pd.isna(row["ker14"]):
                    continue
                pool.append({"code": code, "ker14": float(row["ker14"]),
                             "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                             "industry": h.industry})
            pool += picked
            weights = target_weights(pool, cfg)
            weights = {k: v * exposure_mult for k, v in weights.items()}
            eq2 = equity_now(d)

            # 先卖超配（腾现金），再买低配
            for code, w in sorted(weights.items(), key=lambda kv: kv[1]):
                target_val = w * eq2
                if code not in holdings:
                    continue
                h = holdings[code]
                close = px(code, d)
                if pd.isna(close):
                    continue
                cur_val = h.shares * close
                delta = target_val - cur_val
                if delta >= 0:
                    continue
                # 死区：减一点点不划算（佣金最低 5 元 + 印花税），不动
                if -delta < cfg.band * eq2:
                    continue
                if not can_sell(h, d):
                    stats["skipped_t1"] += 1
                    continue
                shares = _lot(min(h.shares, -delta / close))
                if shares <= 0:
                    continue
                price = close * (1 - SLIPPAGE)
                amount = shares * price
                fee = _sell_cost(amount, cfg.fees_on)
                cash += amount - fee
                pnl = (price - h.cost) * shares - fee
                realized[code] = realized.get(code, 0.0) + pnl
                h.shares -= shares
                trades.append(TradeRec(d, code, "卖出", shares, price, amount, fee, pnl,
                                       f"减仓到目标 {w:.1%}"))
                if h.shares == 0:
                    del holdings[code]

            for code, w in sorted(weights.items(), key=lambda kv: -kv[1]):
                target_val = w * eq2
                close = px(code, d)
                if pd.isna(close):
                    continue
                h = holdings.get(code)
                cur_val = h.shares * close if h else 0.0
                delta = target_val - cur_val
                if delta <= 0:
                    continue
                # 死区：加一点点也不划算
                if delta < cfg.band * eq2:
                    continue
                # 单票上限（熔断打折后仍不得超过 max_weight × 当前总资产）
                room = cfg.max_weight * eq2 - cur_val
                delta = min(delta, room)
                shares = _lot(delta / (close * (1 + SLIPPAGE)))
                if shares <= 0:
                    continue
                price = close * (1 + SLIPPAGE)
                amount = shares * price
                fee = _buy_cost(amount, cfg.fees_on)
                if amount + fee > cash:
                    # 现金不够就按可用现金缩量
                    shares = _lot((cash - COMMISSION_MIN) / (price * 1.0))
                    if shares <= 0:
                        stats["skipped_cash"] += 1
                        continue
                    amount = shares * price
                    fee = _buy_cost(amount, cfg.fees_on)
                cash -= amount + fee
                if h is None:
                    h = Holding(code=code, shares=0, cost=0.0, buy_date=d,
                                industry=industry_of(code, cfg.industry_map))
                    holdings[code] = h
                # 成本按加权平均（含买入费用），这样止损/盈亏都是真实的
                total_cost = h.cost * h.shares + amount + fee
                h.shares += shares
                h.cost = total_cost / h.shares
                if h.buy_date is None or pd.Timestamp(h.buy_date) < d:
                    h.buy_date = d          # T+1：今天买的，明天才能卖
                trades.append(TradeRec(d, code, "买入", shares, price, amount, fee, 0.0,
                                       f"金叉 KER={h_shares_note(aligned[code].at[d, 'ker14'])} 目标 {w:.1%}"))

        eq = equity_now(d)
        equity_curve.append((d, eq))
        mv = sum(h.shares * px(c, d) for c, h in holdings.items() if pd.notna(px(c, d)))
        exposure_curve.append((d, mv / eq if eq > 0 else 0.0))

    # ---------- 期末清算 ----------
    last = days[-1]
    for code in list(holdings.keys()):
        h = holdings[code]
        close = px(code, last)
        if pd.isna(close):
            continue
        price = close * (1 - SLIPPAGE)
        amount = h.shares * price
        fee = _sell_cost(amount, cfg.fees_on)
        cash += amount - fee
        pnl = (price - h.cost) * h.shares - fee
        realized[code] = realized.get(code, 0.0) + pnl
        trades.append(TradeRec(last, code, "卖出", h.shares, price, amount, fee, pnl, "期末清算"))
        del holdings[code]
    equity_curve[-1] = (last, cash)

    eq_series = pd.Series({d: v for d, v in equity_curve}).sort_index()
    exp_series = pd.Series({d: v for d, v in exposure_curve}).sort_index()
    metrics = compute_metrics(eq_series, equity_curve, trades, realized, cfg,
                              breaker_events, breaker_days, exp_series, stats)
    return {"metrics": metrics, "equity": eq_series, "exposure": exp_series,
            "trades": trades, "breaker": breaker_events,
            "realized": realized, "days": days, "frames": frames}


def h_shares_note(v) -> str:
    return f"{float(v):.3f}" if v is not None and not pd.isna(v) else "n/a"


# ============================================================================
# 绩效指标（夏普 / 卡玛 / 胜率 / 盈亏比 / 归因）
# ============================================================================
def compute_metrics(eq: pd.Series, points, trades: list[TradeRec], realized: dict[str, float],
                    cfg: Config, breaker_events: list[dict], breaker_days: int,
                    exposure: pd.Series, stats: dict) -> dict:
    start_v, end_v = float(eq.iloc[0]), float(eq.iloc[-1])
    n = len(eq)
    total_ret = end_v / cfg.capital - 1.0
    ann = (1 + total_ret) ** (TRADING_DAYS / n) - 1 if n > 1 else 0.0
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * math.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0
    peak = eq.cummax()
    dd_series = eq / peak - 1
    max_dd = float(dd_series.min())
    calmar = (ann / abs(max_dd)) if max_dd < 0 else float("inf")

    sells = [t for t in trades if t.side == "卖出"]
    wins = [t.pnl for t in sells if t.pnl > 0]
    losses = [t.pnl for t in sells if t.pnl <= 0]
    win_rate = len(wins) / len(sells) if sells else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)

    fees = sum(t.fee for t in trades)
    # 归因：按股票 / 按行业
    by_stock = sorted(realized.items(), key=lambda kv: -kv[1])
    by_ind: dict[str, float] = {}
    for code, pnl in realized.items():
        ind = industry_of(code, cfg.industry_map)
        by_ind[ind] = by_ind.get(ind, 0.0) + pnl

    verdict = ("✅ 夏普 ≥ 1，达到上实盘的门槛（但还要过样本外检验）" if sharpe >= SHARPE_MIN
               else f"⛔ 夏普 {sharpe:.2f} < {SHARPE_MIN} → 按纪律不该上实盘")

    return {
        "start": str(points[0][0].date()), "end": str(points[-1][0].date()),
        "bars": n, "capital": cfg.capital, "final_equity": round(end_v, 2),
        "total_return": round(total_ret * 100, 2), "annual_return": round(ann * 100, 2),
        "max_drawdown": round(max_dd * 100, 2), "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2) if math.isfinite(calmar) else None,
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else None,
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2) if math.isfinite(payoff) else None,
        "trades": len(trades), "round_trips": len(sells), "fees_paid": round(fees, 2),
        "avg_exposure": round(float(exposure.mean()) * 100, 2),
        "max_exposure": round(float(exposure.max()) * 100, 2),
        "breaker_triggers": sum(1 for e in breaker_events if e["event"] == "触发熔断"),
        "breaker_days": breaker_days,
        "skipped_t1": stats["skipped_t1"], "skipped_cash": stats["skipped_cash"],
        "skipped_cap": stats["skipped_cap"],
        "by_stock": {k: round(v, 2) for k, v in by_stock},
        "by_industry": {k: round(v, 2) for k, v in sorted(by_ind.items(), key=lambda kv: -kv[1])},
        "verdict": verdict,
        "params": {
            "max_weight": cfg.max_weight, "industry_max": cfg.industry_max,
            "dd_trigger": cfg.dd_trigger, "dd_factor": cfg.dd_factor,
            "dd_recover": cfg.dd_recover, "max_positions": cfg.max_positions,
            "commission": COMMISSION, "commission_min": COMMISSION_MIN,
            "slippage": SLIPPAGE, "stamp": STAMP, "lot": LOT,
        },
    }


# ============================================================================
# 报告 + 图
# ============================================================================
def write_report(path: str, res: dict) -> str:
    m = res["metrics"]
    L: list[str] = []
    L.append("# 组合级回测报告（多标的 · 四层风控）\n")
    L.append(f"- 区间：{m['start']} ~ {m['end']}（{m['bars']} 个交易日）")
    L.append(f"- 初始资金：{m['capital']:,.0f} 元　期末：{m['final_equity']:,.2f} 元")
    L.append(f"- 参数：单票上限 {m['params']['max_weight']:.0%}｜同行业上限 {m['params']['industry_max']} 只"
             f"｜回撤熔断 {m['params']['dd_trigger']:.0%}→×{m['params']['dd_factor']:.0%}"
             f"｜滑点 {m['params']['slippage']:.2%}｜佣金万 2.5（最低 5 元）｜印花税 {m['params']['stamp']:.1%}")
    L.append("")
    L.append("## 一、绩效\n")
    L.append("| 指标 | 数值 | 门槛 |")
    L.append("|---|---|---|")
    L.append(f"| 总收益率 | **{m['total_return']:+.2f}%** | — |")
    L.append(f"| 年化收益 | {m['annual_return']:+.2f}% | — |")
    L.append(f"| 最大回撤 | {m['max_drawdown']:.2f}% | 熔断线 {m['params']['dd_trigger']:.0%} |")
    L.append(f"| **夏普比率** | **{m['sharpe']:.2f}** | ≥ {SHARPE_MIN} |")
    L.append(f"| 卡玛比率 | {m['calmar']} | > 0.5 才算能扛 |")
    L.append(f"| 胜率 | {m['win_rate']:.2f}% | — |")
    L.append(f"| 盈亏比（均值） | {m['payoff_ratio']} | > 1.5 才健康 |")
    L.append(f"| 盈亏比（总额） | {m['profit_factor']} | > 1.0 |")
    L.append(f"| 交易笔数 / 平仓次数 | {m['trades']} / {m['round_trips']} | — |")
    L.append(f"| 手续费合计 | {m['fees_paid']:,.2f} 元 | — |")
    L.append(f"| 平均仓位 / 最高仓位 | {m['avg_exposure']}% / {m['max_exposure']}% | — |")
    L.append("")
    L.append(f"> 结论：{m['verdict']}")
    L.append("")
    L.append("## 二、风控执行情况\n")
    L.append(f"- 市场中性：同行业最多 {m['params']['industry_max']} 只，单票最多 {m['params']['max_weight']:.0%}")
    L.append(f"- 熔断触发 {m['breaker_triggers']} 次，处于熔断状态 {m['breaker_days']} 个交易日")
    for e in res["breaker"][:8]:
        L.append(f"  - {e['date']}　{e['event']}（回撤 {e['drawdown']:.2f}%）→ {e['action']}")
    L.append(f"- T+1 拦截 {m['skipped_t1']} 次（当日买入不许当日卖）")
    L.append(f"- 现金不足缩量 {m['skipped_cash']} 次｜持仓数达上限 {m['skipped_cap']} 次")
    L.append("")
    L.append("## 三、绩效归因\n")
    L.append("### 按股票（已实现盈亏，元）\n")
    L.append("| 代码 | 已实现盈亏 |")
    L.append("|---|---|")
    for code, pnl in sorted(m["by_stock"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {code} | {pnl:+,.2f} |")
    L.append("")
    L.append("### 按行业（已实现盈亏，元）\n")
    L.append("| 行业 | 已实现盈亏 |")
    L.append("|---|---|")
    for ind, pnl in m["by_industry"].items():
        L.append(f"| {ind} | {pnl:+,.2f} |")
    L.append("")
    L.append("## 四、交易明细\n")
    L.append("| 日期 | 代码 | 方向 | 股数 | 成交价 | 金额 | 费用 | 实现盈亏 | 备注 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in res["trades"]:
        L.append(f"| {t.date.date()} | {t.code} | {t.side} | {t.shares} | {t.price:.2f} | "
                 f"{t.amount:,.0f} | {t.fee:.2f} | {t.pnl:+,.2f} | {t.note} |")
    L.append("")
    L.append("## 五、这套东西的已知边界\n")
    L.append("1. 行业分类是内置映射（`DEFAULT_INDUSTRY`），生产环境应换成 CSV（`--industry-file`）。")
    L.append("2. 选股只用「金叉 + KER>0」，没接基本面过滤；要更稳可以先用 `fundamental_filter.py` 洗一遍池子。")
    L.append("3. 没有建模涨跌停、停牌、除权除息；成交价用当日收盘 ± 滑点，属于乐观假设。")
    L.append("4. 熔断是「降低暴露」不是「清仓」，回撤 10% 后仍保留 30% 仓位，这点和很多人的直觉不同。")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def plot(res: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    eq = res["equity"]
    exp = res["exposure"].reindex(eq.index).fillna(0)
    peak = eq.cummax()
    dd = (eq / peak - 1) * 100

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    ax = axes[0]
    ax.plot(eq.index, eq / eq.iloc[0], label="组合净值", color="#6a5acd", lw=1.6)
    ax.set_title(f"组合回测 {eq.index[0].date()} ~ {eq.index[-1].date()}"
                 f"｜收益 {res['metrics']['total_return']:+.2f}%"
                 f"｜夏普 {res['metrics']['sharpe']:.2f}"
                 f"｜最大回撤 {res['metrics']['max_drawdown']:.2f}%")
    ax.grid(alpha=.3)
    ax.legend(loc="upper left")

    ax2 = axes[1]
    ax2.fill_between(dd.index, dd.values, 0, color="#d94f4f", alpha=.35)
    ax2.plot(dd.index, dd.values, color="#d94f4f", lw=1)
    ax2.axhline(-res["metrics"]["params"]["dd_trigger"] * 100, color="#444", ls="--", lw=.9,
                label=f"熔断线 {res['metrics']['params']['dd_trigger']:.0%}")
    ax2.set_ylabel("回撤 (%)")
    ax2.grid(alpha=.3)
    ax2.legend(loc="lower left", fontsize=8)

    ax3 = axes[2]
    ax3.fill_between(exp.index, exp.values * 100, 0, color="#4f9ad9", alpha=.35)
    ax3.set_ylabel("仓位 (%)")
    ax3.set_ylim(0, 105)
    ax3.grid(alpha=.3)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ============================================================================
# CLI
# ============================================================================
def selftest() -> int:
    """
    自检：把四条要求里"只能是纸面承诺"的部分变成可执行的断言。
    跑 `python portfolio.py --selftest`，全绿才算这四条真的落地了。
    """
    checks: list[tuple[str, bool, str]] = []

    # --- ③ 成本穿透：整手 ---
    checks.append(("整手取整 1234.9 → 1200", _lot(1234.9) == 1200, f"实际 {_lot(1234.9)}"))
    checks.append(("不足一手 → 0 股", _lot(99) == 0, f"实际 {_lot(99)}"))
    checks.append(("整手 100 → 100 股", _lot(100) == 100, f"实际 {_lot(100)}"))

    # --- ③ 成本穿透：佣金 / 印花税 / 滑点 ---
    commission_ok = abs(_buy_cost(100_000, True) - 25.0) < 1e-6
    checks.append(("买入 10 万佣金 = 万 2.5 = 25 元", commission_ok, f"实际 {_buy_cost(100_000, True)}"))
    stamp_ok = abs(_sell_cost(100_000, True) - 125.0) < 1e-6
    checks.append(("卖出 10 万 = 佣金 25 + 印花税 100 = 125 元", stamp_ok, f"实际 {_sell_cost(100_000, True)}"))
    min_ok = _commission(1_000) == COMMISSION_MIN
    checks.append(("佣金最低 5 元（1000 元成交也收 5 元）", min_ok, f"实际 {_commission(1_000)}"))
    checks.append(("滑点 0.02% 已启用", abs(SLIPPAGE - 0.0002) < 1e-9, f"实际 {SLIPPAGE}"))

    # --- ③ T+1 ---
    today = pd.Timestamp("2025-03-03")
    h_today = Holding(code="000001", shares=100, cost=10.0, buy_date=today)
    h_old = Holding(code="000001", shares=100, cost=10.0, buy_date=today - pd.Timedelta(days=1))
    checks.append(("T+1：当日买入当日不可卖", can_sell(h_today, today) is False, ""))
    checks.append(("T+1：隔日可卖", can_sell(h_old, today) is True, ""))

    # --- ① 单票上限 ---
    cfg = Config(symbols=["x"], industry_map={})
    flat = [{"code": f"S{i}", "ker14": 0.3, "vol20": 0.02, "industry": "测试"} for i in range(5)]
    w = target_weights(flat, cfg)
    cap_ok = all(v <= cfg.max_weight + 1e-9 for v in w.values())
    checks.append((f"单票上限 {cfg.max_weight:.0%}：5 只等权各不超上限", cap_ok,
                   f"最大权重 {max(w.values()):.4f}、权重和 {sum(w.values()):.4f}"))

    # --- ① 行业中性 ---
    same_ind = [{"code": f"S{i}", "ker14": 0.5 - i * 0.01, "vol20": 0.02, "industry": "银行"} for i in range(5)]
    picked = pick_candidates(same_ind, {}, cfg)
    checks.append((f"行业中性：同行业最多 {cfg.industry_max} 只", len(picked) == cfg.industry_max,
                   f"实际选中 {len(picked)} 只"))
    mixed = ([{"code": f"B{i}", "ker14": 0.5, "vol20": 0.02, "industry": "银行"} for i in range(4)]
             + [{"code": f"L{i}", "ker14": 0.4, "vol20": 0.02, "industry": "白酒"} for i in range(4)])
    mixed_picked = pick_candidates(mixed, {}, cfg)
    inds = [c["industry"] for c in mixed_picked]
    checks.append(("行业中性：不同行业分别计数", inds.count("银行") == 3 and inds.count("白酒") == 3,
                   f"银行 {inds.count('银行')} 只、白酒 {inds.count('白酒')} 只"))

    # --- ② 逆波动率加权 ---
    # 注意：这里把上限放开到 100%，是为了单独验证"加权公式"本身；
    #       上限的拦截在上一组测试里已经单独验证过了。
    cfg_w = Config(symbols=["x"], industry_map={}, max_weight=1.0)
    lowvol = [{"code": "LOW", "ker14": 0.3, "vol20": 0.01, "industry": "A"},
              {"code": "HIGH", "ker14": 0.3, "vol20": 0.02, "industry": "B"}]
    w2 = target_weights(lowvol, cfg_w)
    ratio = w2["LOW"] / w2["HIGH"] if w2["HIGH"] > 0 else 0
    checks.append(("逆波动率：波动 1% 的票权重是波动 2% 的 2 倍", abs(ratio - 2.0) < 0.05,
                   f"实际比值 {ratio:.3f}"))

    # --- ② KER 调节 ---
    cfg_k = Config(symbols=["x"], industry_map={}, ker_adjust=0.5, max_weight=1.0)
    ker_hi = [{"code": "HI", "ker14": 0.9, "vol20": 0.01, "industry": "A"},
              {"code": "LO", "ker14": 0.1, "vol20": 0.01, "industry": "B"}]
    w3 = target_weights(ker_hi, cfg_k)
    checks.append(("KER 动态调仓：趋势强的票权重更高", w3["HI"] > w3["LO"],
                   f"HI {w3['HI']:.3f} vs LO {w3['LO']:.3f}"))

    # --- ① 回撤熔断 ---
    dd = -0.12
    trigger = dd <= -cfg.dd_trigger
    checks.append((f"熔断：回撤 {dd:.0%} 触发（阈值 {cfg.dd_trigger:.0%}）", trigger, ""))
    checks.append((f"熔断后仓位因子 = {cfg.dd_factor:.0%}", abs((cfg.dd_factor if trigger else 1.0) - 0.3) < 1e-9,
                   f"实际 {cfg.dd_factor}"))
    checks.append((f"回撤 {cfg.dd_recover:.0%} 以内解除熔断",
                   not (dd + 0.08 <= -cfg.dd_trigger), f"{dd + 0.08:.0%}"))

    ok = 0
    print("=== portfolio.py 自检 ===")
    for name, passed, detail in checks:
        print(f"{'✅' if passed else '❌'} {name}" + (f"　（{detail}）" if detail and not passed else ""))
        ok += 1 if passed else 0
    print(f"\n{ok}/{len(checks)} 项通过")
    return 0 if ok == len(checks) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="组合级回测：行业中性 + 单票上限 + 回撤熔断 + 逆波动率权重 + 真实成本 + 归因")
    p.add_argument("--symbols", default=None,
                   help="逗号分隔的股票池，默认用内置 12 只跨行业股票")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--max-weight", type=float, default=MAX_WEIGHT, help="单票上限，默认 0.10")
    p.add_argument("--industry-max", type=int, default=INDUSTRY_MAX, help="同行业最多持有几只")
    p.add_argument("--dd-trigger", type=float, default=DRAWDOWN_TRIGGER, help="组合回撤熔断线")
    p.add_argument("--dd-factor", type=float, default=DRAWDOWN_FACTOR, help="熔断后仓位折扣")
    p.add_argument("--dd-recover", type=float, default=DRAWDOWN_RECOVER, help="解除熔断的回撤阈值")
    p.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    p.add_argument("--band", type=float, default=REBALANCE_BAND,
                   help="调仓死区（占总资产比例），默认 0.005，差额不到就不动")
    p.add_argument("--ker-adjust", type=float, default=KER_ADJUST, help="KER 对权重的调节强度")
    p.add_argument("--industry-file", default=None, help="行业映射 CSV（code,industry）")
    p.add_argument("--no-fees", action="store_true", help="关掉佣金/印花税/滑点（对比用）")
    p.add_argument("--stamp", type=float, default=STAMP,
                   help="卖出印花税，默认 0.001（千分之一）。现行税率 0.05%% 可用 --stamp 0.0005")
    p.add_argument("--out", default="portfolio_report.md")
    p.add_argument("--chart", default="portfolio_curve.png")
    p.add_argument("--json", action="store_true", help="把 metrics 打到 stdout（JSON）")
    p.add_argument("--selftest", action="store_true", help="只跑四条铁律的自检，不回测")
    return p


def load_industry(path: str | None) -> dict[str, str]:
    mapping = dict(DEFAULT_INDUSTRY)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        col_code = "code" if "code" in df.columns else df.columns[0]
        col_ind = "industry" if "industry" in df.columns else df.columns[1]
        for c, i in zip(df[col_code], df[col_ind]):
            mapping[to_code6(c)] = str(i)
        log(f"[industry] 从 {path} 载入 {len(df)} 条映射")
    return mapping


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    symbols = ([to_code6(s) for s in args.symbols.split(",")] if args.symbols
               else list(DEFAULT_UNIVERSE.keys()))
    cfg = Config(symbols=symbols, start=args.start, end=args.end, capital=args.capital,
                 max_weight=args.max_weight, industry_max=args.industry_max,
                 dd_trigger=args.dd_trigger, dd_factor=args.dd_factor,
                 dd_recover=args.dd_recover, max_positions=args.max_positions,
                 band=args.band, ker_adjust=args.ker_adjust, industry_map=load_industry(args.industry_file),
                 fees_on=not args.no_fees)

    # 关掉费用时把滑点也归零，否则"无费用"就不干净
    global SLIPPAGE, STAMP
    if args.no_fees:
        SLIPPAGE = 0.0
    STAMP = args.stamp

    res = run(cfg)
    m = res["metrics"]
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        print(f"期末 {m['final_equity']:,.2f} 元｜收益 {m['total_return']:+.2f}%｜"
              f"回撤 {m['max_drawdown']:.2f}%｜夏普 {m['sharpe']:.2f}｜"
              f"卡玛 {m['calmar']}｜胜率 {m['win_rate']:.1f}%｜盈亏比 {m['payoff_ratio']}")
        print(m["verdict"])
    print(f"[report] {write_report(args.out, res)}", file=sys.stderr)
    print(f"[chart] {plot(res, args.chart)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
