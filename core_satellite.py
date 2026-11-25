#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core_satellite.py —— 核心-卫星组合引擎（V3.3 编排层）
====================================================

这个文件现在只做四件事：读配置 → 逐日编排 → 出报告 → 画图。

具体代码按职责拆成了四个模块：

    cs_data.py      数据层：取数、缓存、清洗、折算修正、指标、对齐
    cs_regime.py    裁判层：MA200 方向 + 年线斜率 + 确认天数 + 缓冲带
    cs_strategy.py  策略层：ETF 动量 / 防御低波 / 个股金叉 的候选与权重
    cs_risk.py      风控层：费用、整手、回撤熔断、宏观减仓目标

三条腿的架构（V3.2 起）：

    总资产 ─┬─ 核心（默认 50%）  沪深300，常态死拿
            │       ↓ 沪深300 跌破 MA200 且年线拐头向下
            │       └─→ 按 --core-bear {hold,half,zero} 减仓，钱进 reserve
            └─ 卫星（默认 50%）
                    ├─ 长期股 25%：5 只买入持有（同样受 --lt-bear 管）
                    └─ 择时腿 25%：上行 = ETF 动量轮动；下行 = 国债/黄金防守

用法
----
    python core_satellite.py --start 2024-01-01 --end 2024-12-31
    python core_satellite.py --start 2008-01-01 --end 2008-12-31 --core-bear zero
    python core_satellite.py --start 2024-01-01 --end 2024-12-31 --regime-slope-min 0
    python core_satellite.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import portfolio as pf
from portfolio import Holding, TradeRec

# ---- 四个模块 ----
from cs_data import (CORE_INDEX, CORE_NAME, ETF_UNIVERSE, BROAD_ETF, DEFAULT_REGULAR, DEFAULT_LONGTERM,
                     INDUSTRY_MAP, NAME_MAP, ETF_TOP, ETF_MOM,
                     load_index, load_stocks, load_etf, prepare)
from cs_regime import (BEAR_POOL, BEAR_POOL_ALL, BEAR_TOP, BEAR_MA, REGIME_CONFIRM,
                       bench_frame, build_states)
from cs_strategy import (CORE_WEIGHT, LT_SHARE, MAX_WEIGHT, INDUSTRY_MAX, MAX_POSITIONS, BAND, COOLDOWN, CONFIRM,
                         MIN_HOLD, TRANCHES, TRANCHE_GAP, KER_MIN,
                         etf_candidates, defensive_candidates, stock_candidates, equal_weights)
import cs_hmm
from cs_risk import (COMMISSION, COMMISSION_MIN, STAMP, SLIPPAGE, LOT, DD_TRIGGER, DD_FACTOR,
                     DD_RECOVER, buy_fee, sell_fee, lot, _STAMP,
                     set_fees, slip_buy, slip_sell, TRANSFER,
                     breaker_step, cut_ratio)

# ---- 实盘门槛（2026-11-18 决定：仍用 1.00）----
# 期间试过下调到 0.80（理由：回测 0.80 ≈ 实盘 0.5~0.6），当天决定改回 1.00：
#   当前主要目标是压回撤，不是尽早开仓；门槛放宽只会把风险提前放到实盘上。
# 保留 0.80 的教训写在 rag_finance/cards/34-夏普门槛.md：
#   门槛高低本身不防过拟合，配「0.50~1.00 区间不许调参」这条锁才防。
SHARPE_LIVE = 1.00        # ≥ 此值：可上小仓验证（≤10% 资金）+ 跑满 12 个月实盘对照
SHARPE_WATCH = 0.50       # < 此值：结构有问题，回滚上一版

# ---- 引擎版本（必须跟着 core_bear/lt_bear/CORE_WEIGHT 的默认值一起改）----
# 教训（2026-11-24）：core_satellite.py 与 core_satellite_v45.py 同时在目录里，
# 读错文件后基于 V4.5 的假设去解释 V4.6 的 -100%，白查一小时。
# 所以版本号要印在报告头和 CLI 里，一眼能看出这次跑的是哪一版。
ENGINE_VERSION = "V4.6"
VERSION_NOTES = {
    "V4.6": "转防守时核心/长期股清零（core_bear=zero, lt_bear=zero），核心占比 20%",
    "V4.5": "转防守时核心/长期股减半（core_bear=half, lt_bear=half）",
    "V4.4": "封版口径：波动目标 10%、现金计息 2%、核心死区 5%、HMM 开",
}


def version_banner(cfg: "Config") -> str:
    """一行版本+关键参数，报告头与 CLI 共用。"""
    cb = getattr(cfg, "core_bear", "-")
    lb = getattr(cfg, "lt_bear", "-")
    cy = getattr(cfg, "cash_yield", 0.0)
    cw = getattr(cfg, "core_weight", 0.0)
    return (f"引擎 {ENGINE_VERSION}｜core_bear={cb}｜lt_bear={lb}"
            f"｜cash_yield={cy:.1%}｜核心占比 {cw:.0%}")

LOG = lambda m: print(m, file=sys.stderr)


# ============================================================================
# 二、配置
# ============================================================================
@dataclass
class Config:
    start: str = "2024-01-01"
    end: str = "2024-12-31"
    capital: float = 1_000_000.0
    core_weight: float = CORE_WEIGHT
    lt_share: float = LT_SHARE
    max_weight: float = MAX_WEIGHT
    industry_max: int = INDUSTRY_MAX
    max_positions: int = MAX_POSITIONS
    band: float = BAND
    cooldown: int = COOLDOWN
    confirm: int = CONFIRM
    exit_gate_market: bool = False
    sleeve: str = "etf"            # stock=个股择时；etf=行业 ETF 动量轮动（默认，实测更优）
    etf_top: int = ETF_TOP
    etf_mom: int = ETF_MOM
    etf: dict = field(default_factory=lambda: dict(ETF_UNIVERSE))
    etf_pool: str = "20"            # 12=扩容前的原始池；17=剔除宽基；20=扩容后（默认）
    trend_ma: int = 0               # >0 时 ETF 入场需站上该长期均线（250=年线）
    min_hold: int = MIN_HOLD
    tranches: int = TRANCHES
    tranche_gap: int = TRANCHE_GAP
    dd_trigger: float = DD_TRIGGER
    dd_factor: float = DD_FACTOR
    dd_recover: float = DD_RECOVER
    ker_min: float = KER_MIN
    ker_adjust: float = 0.5
    regime: str = "ma200"           # ma200=沪深300 站上 MA200 才进攻（默认）；off=永续进攻（V2 行为）否则切防守腿
    bear_top: int = BEAR_TOP        # 防守腿最多持有几只防御资产
    bear_ma: int = BEAR_MA          # 防守资产的趋势闸门均线
    regime_band: float = 0.03       # 状态缓冲带：跌破 MA200×(1-band) 才转下行、站上 ×(1+band) 才转上行
    regime_confirm: int = REGIME_CONFIRM   # 连续 N 日越线才切换状态（1=当日就切）
    regime_slope: int = 0           # 年线斜率窗口（0=不看斜率）
    regime_slope_min: float = 0.0   # 斜率低于此值才算「年线拐头」
    bear_pool: str = "bond_gold"    # bond_gold=国债+黄金（V3.1 默认）；all=再加红利（V3.0 旧口径）
    hmm: bool = True                # V4.0：用 HMM 牛市概率强行满仓（2026-11-09 起默认开启）
    hmm_threshold: float = 0.85     # 概率 ≥ 该值视为「确认牛市」
    hmm_ma200_guard: bool = True
    hmm_exit: float = 0.70          # HMM 退出阈值（滞后：进 0.85，出 0.70）
    hmm_confirm: int = 1            # 进/出各需连续确认几天（进快出慢：靠 exit 阈值滞后，不靠延迟进场）
    breaker_max_days: int = 120     # 熔断最长天数，到点强制解锁（0 = 不限）    # 再加一道「价在年线上方」的确认
    core_bear: str = "zero"         # V4.6：转防守时核心底仓清零（V4.5 是 half；降回撤实验见 V4.6_降回撤_2026-11-18.md）
    lt_bear: str = "zero"           # V4.6：转防守时长期股清零（V4.5 是 half；这是 −27.2%→−20.8% 的主因）
    fees_on: bool = True
    stamp: float = STAMP
    commission: float = COMMISSION        # 佣金费率
    commission_min: float = COMMISSION_MIN  # 单笔最低佣金
    transfer: float = TRANSFER            # 过户费（A 股双向）
    impact: float = 0.0                   # 固定冲击成本（单边附加 bps）
    impact_coef: float = 0.0              # 按成交量占比的冲击系数：cost = coef × (下单额/20日均额)
    impact_max_part: float = 0.05         # 下单额超过 20 日均额的该比例 → 直接拒单
    stamp_old: float = 0.001              # 2023-08-28 之前的印花税率（历史分段用）
    stamp_cut: str = "2023-08-28"
    stamp_history: bool = True            # True=按日期分段用 stamp_old/stamp
    cash_yield: float = 0.02        # 闲置现金年化（V4.4 默认 2%）；0=不计息
    vol_target: float = 0.10        # 目标年化波动（V4.4 默认 10%）；0=关闭
    vol_win: int = 20
    vol_floor: float = 0.30         # 仓位缩放的下限（防止极端波动时清仓）
    vol_band: float = 0.05          # 核心调仓死区（占原始核心份额的比例），防日频微调磨损        # 闲置现金年化收益率（货币基金），0=不计
    regular: dict = field(default_factory=lambda: dict(DEFAULT_REGULAR))
    longterm: dict = field(default_factory=lambda: dict(DEFAULT_LONGTERM))


# ============================================================================
# 五、回测主循环
# ============================================================================
def run(cfg: Config) -> dict:
    _STAMP[0] = cfg.stamp
    set_fees(cfg.commission, cfg.commission_min, cfg.transfer, cfg.impact, cfg.stamp)
    idx, dates, aligned = prepare(cfg)
    if len(dates) < cfg.confirm + 20:
        raise SystemExit("交易日太少，至少需要 30 个交易日")

    etf_mode = cfg.sleeve == "etf"
    sleeve_codes = [c for c in (cfg.etf if etf_mode else cfg.regular) if c in aligned]
    longterm = [c for c in cfg.longterm if c in aligned]
    LOG(f"[universe] {'行业ETF' if etf_mode else '普通股'} {len(sleeve_codes)} 只，"
        f"长期股 {len(longterm)} 只，交易日 {len(dates)}")

    # ---------- V3.0 状态裁判 ----------
    # 每个交易日先问一句「今天是上行还是下行」，上行用动量池，下行换防御池。
    # 核心账本（50% 沪深300）与长期股不参与切换——它们是压舱石，不是战术仓位。
    bear_on = cfg.regime != "off"
    bear_pool = BEAR_POOL_ALL if getattr(cfg, "bear_pool", "bond_gold") == "all" else BEAR_POOL
    bear_set = (set(bear_pool) & set(aligned)) if bear_on else set()
    bench: pd.DataFrame | None = None
    bull_state: bool | None = None
    bull_run = bear_run = 0        # 连续越线计数（配合 --regime-confirm）
    regime_flips = 0
    states: pd.Series | None = None
    if bear_on:
        try:
            # 年线要用「完整历史」算：只喂回测区间的话，前 200 天 MA200 全是 NaN，
            # 裁判会一路返回 True（等于没有裁判）。
            close_full = load_index("1990-01-01", cfg.end)
            slope_win = max(0, int(getattr(cfg, "regime_slope", 0)))
            # 状态机从回测首日起算（不在 2005 年预热），复现旧口径的衔接日行为
            bench = bench_frame(close_full, 200, slope_win).reindex(dates).ffill()
            states = build_states(bench,
                                  confirm=max(1, int(getattr(cfg, "regime_confirm", REGIME_CONFIRM))),
                                  band=float(cfg.regime_band),
                                  slope_win=slope_win,
                                  slope_min=float(getattr(cfg, "regime_slope_min", 0.0)))
            n_raw = int((bench["close"] > bench["ma"]).fillna(True).sum())
            n_bull = int(states.sum())
            LOG(f"[regime] 沪深300 vs MA200：原始上行 {n_raw} 天，裁判判上行 {n_bull} 天 / 共 {len(dates)} 天"
                f"（缓冲带 ±{cfg.regime_band:.1%}，确认 {cfg.regime_confirm} 日，"
                f"斜率窗口 {slope_win} 日、阈值 {float(getattr(cfg, 'regime_slope_min', 0.0)):.2%}）")
        except Exception as exc:
            LOG(f"[warn] 状态裁判不可用，本次回退为「全程进攻」：{exc}")
            bear_on, bear_set, bench, states = False, set(), None, None

    core_cash = cfg.capital * cfg.core_weight
    sat_cash = cfg.capital * (1.0 - cfg.core_weight)

    # ---------- 期初建仓 ----------
    d0 = dates[0]
    idx0 = float(idx.loc[d0])
    # 核心用「指数点位」模拟持有沪深300 ETF，允许小数份额（ETF 份额本来就是整数份，
    # 但指数点位本身不能整手买，这里按金额等分，避免 100 手取整把 33% 的钱剩成现金）
    core_shares = core_cash / (idx0 * (1 + slip_buy()))
    core_amount = core_shares * idx0 * (1 + slip_buy())
    core_fee = buy_fee(core_amount, cfg.fees_on)
    sat_cash -= core_fee
    core_cash = 0.0
    core_shares_orig = core_shares      # V3.2：极值减仓后用来回补
    reserve = 0.0                       # 宏观风控卖出现的现金（不进卫星预算，避免变相加仓）

    trades: list[TradeRec] = []
    # 期初核心建仓也要进成交流水，否则交易明细里看不到底仓、对账也无法复算
    trades.append(TradeRec(d0, CORE_INDEX, "核心建仓", core_shares, idx0 * (1 + slip_buy()),
                           core_amount, core_fee, 0.0, "核心底仓：买入沪深300 指数份额"))
    lt_hold: dict[str, Holding] = {}
    sig_hold: dict[str, Holding] = {}
    last_exit: dict[str, int] = {}
    realized: dict[str, float] = {}
    # 分批建仓状态：code -> 已建批数 / 上一批的交易日序号
    entry_fill: dict[str, int] = {}
    entry_last: dict[str, int] = {}
    # 持有期下限用：code -> 建仓当日的交易日序号
    buy_pos: dict[str, int] = {}

    lt_budget = sat_cash * cfg.lt_share
    if longterm:
        # 先算每只的实际买入价，再按「预算能不能买到一手」分配：
        # 原来的 each = 总预算/只数，遇到高价股（后复权后可能 300+ 元/股）会直接买不足一手被丢掉，
        # 结果「长期股 5 只」实际只建了 4 只、甚至 1 只。
        cand: dict[str, float] = {}
        for code in longterm:
            px = float(aligned[code].loc[d0, "close"])
            if pd.isna(px) or px <= 0:
                LOG(f"[init] 长期股 {code} 首日无行情，跳过")
                continue
            cand[code] = px * (1 + slip_buy())
        per = lt_budget / max(1, len(cand))
        plan: dict[str, int] = {}
        for code, price in cand.items():
            sh = lot(per / price)
            if sh <= 0:
                LOG(f"[init] 长期股 {code} 一手要 {LOT * price:,.0f} 元 > 单只预算 {per:,.0f} 元，跳过")
                continue
            plan[code] = sh
        # 有余钱再均分补一轮，别让预算闲着（也避免每只都卡在刚好一手的边缘）
        left = lt_budget - sum(sh * cand[c] for c, sh in plan.items())
        if plan and left > 0:
            per2 = left / len(plan)
            for code in plan:
                add = lot(per2 / cand[code])
                if add > 0 and add * cand[code] <= left:
                    plan[code] += add
                    left -= add * cand[code]
        for code, sh in plan.items():                    # 按原顺序成交，保持可复现
            price = cand[code]
            amount = sh * price
            fee = buy_fee(amount, cfg.fees_on)
            if amount + fee > sat_cash:
                LOG(f"[init] 长期股 {code} 现金不足，跳过")
                continue
            sat_cash -= amount + fee
            lt_hold[code] = Holding(code, sh, price, d0, INDUSTRY_MAP.get(code, "其他"))
            trades.append(TradeRec(d0, code, "长期建仓", sh, price, amount, fee, 0.0,
                                   f"{NAME_MAP.get(code, code)} 长期持有，不择时"))
    lt_orig = {c: h.shares for c, h in lt_hold.items()}
    LOG(f"[init] 核心 {core_shares} 股指数 @{idx0:.2f}｜长期股 {len(lt_hold)} 只｜卫星现金 {sat_cash:,.0f}")

    # ---------- HMM 牛市概率（walk-forward，无未来函数）----------
    prob_hmm = None
    if cfg.hmm:
        try:
            full_idx = load_index("1990-01-01", cfg.end)
            prob_hmm = cs_hmm.hmm_bull_prob(full_idx).reindex(dates).ffill()
            # 年线用「全历史」算再对齐，否则回测区间前 200 天没年线，确认条件恒假
            globals()["_HMM_MA200"] = full_idx.rolling(200).mean().reindex(dates).ffill()
            LOG(f"[hmm] 概率序列就绪：覆盖 {int(prob_hmm.notna().sum())}/{len(dates)} 天"
                f"（阈值 {cfg.hmm_threshold:.0%}）")
        except Exception as exc:
            LOG(f"[hmm] 概率计算失败，本次降级为关闭：{exc}")
            prob_hmm = None

    eq_curve: list[tuple[pd.Timestamp, float]] = []
    exp_curve: list[tuple[pd.Timestamp, float]] = []
    core_curve: list[tuple[pd.Timestamp, float]] = []
    breaker_events: list[dict] = []
    peak = cfg.capital
    prev_bull = None
    breaker_on = False
    idx_ma20 = idx.rolling(20).mean()
    idx_ma200 = idx.rolling(200).mean()
    breaker_days = 0
    hmm_state = False
    hmm_up = 0
    hmm_dn = 0
    stats = {"skipped_limit": 0, "shrunk_impact": 0, "skipped_t1": 0, "skipped_cooldown": 0, "skipped_confirm": 0,
             "breaker_days": 0, "skipped_minhold": 0}
    interest_acc = 0.0          # 现金理财累计利息（sat_cash + reserve 按日计提）
    core_cash_net = 0.0         # 核心腿净现金流（减仓进 +，回补出 -），用于算核心真实收益

    def px_now(code: str, d) -> float:
        v = aligned[code].loc[d, "close"]
        return float(v) if not pd.isna(v) else float("nan")

    # 20 日均成交额：冲击成本/流动性检查用（ETF 与个股都有成交量）
    def blocked(code: str, d, side: str) -> bool:
        """当日能否成交：停牌（无原始成交）或一字板（涨停买不进/跌停卖不掉）。"""
        a = aligned.get(code)
        if a is None or "raw_close" not in a.columns:
            return False
        raw = a.loc[d, "raw_close"]
        if pd.isna(raw):
            return True                      # 当日根本没有成交 → 停牌/未上市
        if "volume" in a.columns and float(a.loc[d, "volume"]) <= 0:
            return True                      # 成交量为 0 → 停牌（数据源常见写法）
        prev = a["raw_close"].shift(1).loc[d]
        if pd.isna(prev) or float(prev) <= 0:
            return False
        pct = float(raw) / float(prev) - 1.0
        hi = float(a.loc[d, "high"]); lo = float(a.loc[d, "low"])
        if abs(hi - lo) > 1e-9:
            return False                     # 有振幅 = 不是一字板
        lim = 0.195 if str(code)[:2] in ("30", "68") else 0.095
        return (side == "buy" and pct > lim) or (side == "sell" and pct < -lim)

    def _adv20(code: str, d):
        """20 日均成交额（元）；拿不到返回 None。"""
        a = aligned.get(code)
        if a is None or "volume" not in a.columns:
            return None
        amt = (a["close"] * a["volume"]).rolling(20, min_periods=5).mean().loc[d]
        if pd.isna(amt) or float(amt) <= 0:
            return None
        return float(amt)

    def cap_shares(code: str, d, price: float, shares: float, cash=None) -> float:
        """流动性上限：单笔最多吃 impact_max_part × 20日均额，超出部分缩量（不是拒单）。"""
        if cfg.impact_coef <= 0 or cfg.impact_max_part <= 0:
            return shares
        adv = _adv20(code, d)
        if adv is None or price <= 0:
            return shares
        cap_notional = adv * cfg.impact_max_part
        cap_sh = lot(cap_notional / price) if cap_notional >= price * LOT else 0.0
        if cash is not None:
            cap_sh = min(cap_sh, cash)
        if shares > cap_sh:
            stats["shrunk_impact"] += 1
            if os.environ.get("CS_DEBUG_IMPACT"):
                LOG(f"[impact] 缩量 {code} {d.date()} {shares:,.0f} → {cap_sh:,.0f} 股"
                    f"（ADV {adv:,.0f} 元，上限 {cfg.impact_max_part:.0%}）")
            return cap_sh
        return shares

    def impact_bps(code: str, d, notional: float) -> float:
        """按成交量占比的线性冲击：notional/20日均额 × coef（缩量后必然 ≤ 上限）。"""
        if cfg.impact_coef <= 0:
            return 0.0
        adv = _adv20(code, d)
        if adv is None:
            return 0.0
        return float(cfg.impact_coef) * min(float(notional) / adv, cfg.impact_max_part)

    for pos, d in enumerate(dates):
        if cfg.stamp_history:
            _STAMP[0] = cfg.stamp_old if d < pd.Timestamp(cfg.stamp_cut) else cfg.stamp
        # 现金理财：卫星现金与储备按日计提货币基金收益（年化 cash_yield，按 243 日折算）
        if pos > 0 and cfg.cash_yield:
            _r = (1.0 + cfg.cash_yield) ** (1.0 / 243.0) - 1.0
            interest_acc += (sat_cash + reserve) * _r
            sat_cash *= (1.0 + _r)
            reserve *= (1.0 + _r)
        core_val = core_shares * float(idx.loc[d])
        lt_val = sum(h.shares * px_now(c, d) for c, h in lt_hold.items()
                     if not pd.isna(px_now(c, d)))
        sig_val = sum(h.shares * px_now(c, d) for c, h in sig_hold.items()
                      if not pd.isna(px_now(c, d)))
        eq = reserve + sat_cash + core_val + lt_val + sig_val
        eq_curve.append((d, eq))
        core_curve.append((d, core_val))
        exp_curve.append((d, (core_val + lt_val + sig_val) / eq if eq > 0 else 0.0))

        # ---- 波动目标化：近 vol_win 日实现波动 → 缩放系数 ----
        risk_scale = 1.0
        if cfg.vol_target and cfg.vol_target > 0 and len(eq_curve) > cfg.vol_win + 1:
            vals = [v for _, v in eq_curve[-(cfg.vol_win + 1):]]
            rets = [(vals[i] / vals[i - 1] - 1.0) for i in range(1, len(vals)) if vals[i - 1] > 0]
            if len(rets) > 2:
                mu = sum(rets) / len(rets)
                var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
                vol_now = (var ** 0.5) * (243 ** 0.5)
                if vol_now > 1e-9:
                    risk_scale = min(1.0, max(cfg.vol_floor, cfg.vol_target / vol_now))
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        if not breaker_on and dd <= -cfg.dd_trigger:
            breaker_on = True
            breaker_events.append({"date": str(d.date()), "event": "触发熔断",
                                   "drawdown": round(dd * 100, 2),
                                   "action": f"择时股目标仓位 ×{cfg.dd_factor:.0%}"})
            LOG(f"[breaker] {d.date()} 总资产回撤 {dd:.2%} → 择时股仓位打 {cfg.dd_factor:.0%}")
        elif breaker_on and dd >= -cfg.dd_recover:
            breaker_on = False
            breaker_events.append({"date": str(d.date()), "event": "解除熔断",
                                   "drawdown": round(dd * 100, 2), "action": "恢复目标仓位"})
        if breaker_on:
            breaker_days += 1
            stats["breaker_days"] = breaker_days
            if cfg.breaker_max_days and breaker_days >= cfg.breaker_max_days:
                breaker_on = False
                stats["breaker_timeouts"] = stats.get("breaker_timeouts", 0) + 1
                breaker_events.append({"date": str(d.date()), "event": "超时解除熔断",
                                       "drawdown": round(dd * 100, 2),
                                       "action": f"已连续 {breaker_days} 天，强制解除"})
                LOG(f"[breaker] {d.date()} 已连续 {breaker_days} 天 → 超时强制解除")
                breaker_days = 0
        # ---------- HMM 强行满仓（V4.0）----------
        # HMM 判定「确认牛市」时：① 熔断打折失效；② 宏观风控不减仓、已减的补回。
        # 熔断与宏观减仓是"疑似见顶"的保险，HMM 给的是"确认上行"的证据；
        # 两者打架时以上行确认优先——否则会像 2021-2025 那样被熔断锁死四年。
        hmm_bull = False
        if prob_hmm is not None:
            pv = prob_hmm.get(d, float("nan"))
            # 两道确认：HMM 概率达标 + 指数仍在年线上方。
            # 只用概率会在 2008-05 / 2015-06 这种「气泡顶」把仓位打满——
            # 那时 120 日累计涨幅仍然很好看，但价格已经跌回年线下方。
            ma200_s = globals().get("_HMM_MA200")
            above_ma200 = (ma200_s is not None and d in ma200_s.index
                           and not pd.isna(ma200_s.get(d, float("nan")))
                           and float(idx.loc[d]) > float(ma200_s.loc[d]))
            guard_ok = (above_ma200 or not cfg.hmm_ma200_guard)
            pv_ok = (not pd.isna(pv))
            if not hmm_state:
                # 进：概率 ≥ 进阈值 且 年线守卫通过，连续 confirm 天
                if pv_ok and float(pv) >= cfg.hmm_threshold and guard_ok:
                    hmm_up += 1
                else:
                    hmm_up = 0
                if hmm_up >= max(1, int(cfg.hmm_confirm)):
                    hmm_state = True
                    hmm_up = hmm_dn = 0
            else:
                # 出：概率 < 退阈值，或年线守卫失效，连续 confirm 天
                weak = (not pv_ok) or float(pv) < cfg.hmm_exit or not guard_ok
                if weak:
                    hmm_dn += 1
                else:
                    hmm_dn = 0
                if hmm_dn >= max(1, int(cfg.hmm_confirm)):
                    hmm_state = False
                    hmm_up = hmm_dn = 0
            hmm_bull = bool(hmm_state)
        if hmm_bull:
            stats["hmm_days"] = stats.get("hmm_days", 0) + 1
        exposure_mult = (1.0 if (hmm_bull or not breaker_on) else cfg.dd_factor) * risk_scale

        if pos == 0:
            continue

        # ---------- 0) 状态判定 + 换挡 ----------
        # bull=True 走上行模型；bull=False 走防守池。拿不到数据时沿用「上行」——
        # 因为裁判断电时，进攻模型的趋势闸门（MA20）本身就是一层保护。
        bull = True
        if bear_on and states is not None:
            st = states.get(d, None)
            if st is None or pd.isna(st):
                # 数据断了就沿用上一次状态（绝不因为取数失败而凭空换挡）
                bull = True if bull_state is None else bull_state
            else:
                st = bool(st)
                if bull_state is not None and st != bull_state:
                    regime_flips += 1
                bull_state = st
                bull = st
        bear_mode = bool(bear_on and not bull)
        if bear_mode:
            stats["bear_days"] = stats.get("bear_days", 0) + 1

        # 换挡：上行期不留防御仓，下行期不留进攻仓（T+1 约束仍然优先）
        if bear_on:
            for code in list(sig_hold):
                h = sig_hold[code]
                must_switch = (bull and code in bear_set) or (bear_mode and code not in bear_set)
                if not must_switch:
                    continue
                if h.buy_date is not None and pd.Timestamp(h.buy_date) >= d:
                    stats["skipped_t1"] += 1
                    continue
                close = px_now(code, d)
                if pd.isna(close):
                    continue
                price = close * (1 - slip_sell())
                amount = h.shares * price
                fee = sell_fee(amount, cfg.fees_on, etf=etf_mode)
                sat_cash += amount - fee
                pnl = (price - h.cost) * h.shares - fee
                realized[code] = realized.get(code, 0.0) + pnl
                why = "状态切换：转回进攻" if bull else "状态切换：转防守"
                trades.append(TradeRec(d, code, "卖出", h.shares, price, amount, fee, pnl, why))
                last_exit[code] = pos
                del sig_hold[code]
                entry_fill.pop(code, None)
                entry_last.pop(code, None)
                buy_pos.pop(code, None)
                stats["regime_switches"] = stats.get("regime_switches", 0) + 1

        # ---------- 0b) 全局宏观风控（V3.2）----------
        # 2018 的教训：系统性风险里 50% 的宽基底仓跌 30% 就是 −15pp 的拖累，
        # 光靠卫星腿防守救不回来。所以下行期把核心底仓与长期股按档位降到目标仓位，
        # 上行期再补回去。用「目标份额」而不是「按比例卖」——否则指数在年线附近
        # 反复穿越时，每次翻转都再砍一半，底仓会被一刀刀削没（2008 年就出现过连砍两次）。
        # 卖出的钱放 reserve（不进卫星预算，避免变相加仓）。
        if bear_on or (cfg.vol_target and cfg.vol_target > 0):
            # 减仓比例统一走 cs_risk.cut_ratio（唯一真相源），别再在这里复制一份映射表
            f_core = cut_ratio(cfg.core_bear)
            f_lt = cut_ratio(cfg.lt_bear)
            up = bool(bull or hmm_bull)
            base_core = core_shares_orig if (up or not bear_on) else core_shares_orig * (1.0 - f_core)
            tgt_core = base_core * (risk_scale if (cfg.vol_target and cfg.vol_target > 0) else 1.0)
            dead = (cfg.vol_band * core_shares_orig
                    if (cfg.vol_target and cfg.vol_target > 0) else 0.0)
            # 核心：下行减到目标，上行补回原份额
            if core_shares > tgt_core + dead + 1e-9:
                sh = core_shares - tgt_core
                price = float(idx.loc[d]) * (1 - slip_sell())
                amount = sh * price
                fee = sell_fee(amount, cfg.fees_on, etf=True)
                core_shares = tgt_core
                reserve += amount - fee
                core_cash_net += amount - fee
                trades.append(TradeRec(d, CORE_INDEX, "核心减仓", sh, price, amount, fee, 0.0,
                                       f"宏观风控：下行期核心降到 {1 - f_core:.0%} 目标" if bear_on else
                                       f"波动目标化：核心降到 {risk_scale:.0%} 目标"))
                LOG(f"[macro] {d.date()} 核心减仓至 {tgt_core:.1f} 份（卖出 {amount:,.0f} 元转入储备）")
                stats["macro_cut_days"] = stats.get("macro_cut_days", 0) + 1
            elif core_shares < tgt_core - dead - 1e-9 and reserve > 0:
                price = float(idx.loc[d]) * (1 + slip_buy())
                sh = min(tgt_core - core_shares, reserve / price)
                cost = sh * price
                fee = buy_fee(cost, cfg.fees_on)
                if cost + fee > reserve:
                    sh = max(0.0, (reserve - fee) / price)
                    cost = sh * price
                if sh > 0:
                    core_shares += sh
                    reserve -= cost + fee
                    core_cash_net -= cost + fee
                    trades.append(TradeRec(d, CORE_INDEX, "核心回补", sh, price, cost, fee, 0.0,
                                           "宏观风控：上行期补回核心底仓" if bear_on else
                                           "波动目标化：核心补回"))
                    LOG(f"[macro] {d.date()} 核心回补 {sh:.1f} 份（{cost:,.0f} 元）")
            # 长期股：同样按目标股数调整（只在宏观下行期，波动目标化不动长期股）
            for code, h in (list(lt_hold.items()) if bear_on else []):
                orig = lt_orig.get(code, 0)
                if up:
                    tgt = orig
                elif f_lt >= 1.0:
                    tgt = 0                                   # 明确要求清仓
                else:
                    # 半仓：100 股砍半 = 50 → 不足一手，floor 会变成 0，等于把底仓清光。
                    # 这里兜底「至少留一手」，只有 f_lt=1.0 才允许归零。
                    tgt = max(LOT, int(orig * (1.0 - f_lt)) // LOT * LOT) if orig >= LOT else orig
                if h.shares > tgt:
                    close = px_now(code, d)
                    if pd.isna(close):
                        continue
                    if blocked(code, d, "sell"):
                        stats["skipped_limit"] += 1
                        continue
                    sh = int(h.shares - tgt) // LOT * LOT
                    sh = cap_shares(code, d, close, sh)
                    if sh <= 0:
                        continue
                    _imp = impact_bps(code, d, sh * close)
                    price = close * (1 - (slip_sell() + _imp))
                    amount = sh * price
                    fee = sell_fee(amount, cfg.fees_on, etf=False)  # 长期股是股票：印花税 + 过户费
                    h.shares -= sh
                    reserve += amount - fee
                    trades.append(TradeRec(d, code, "长期减仓", sh, price, amount, fee, 0.0,
                                           f"宏观风控：下行期长期股降到 {1 - f_lt:.0%} 目标"))
                # 用 up（bull or hmm_bull）而不是 bull：HMM 越权牛时核心会回补，
                # 长期股若仍卡在 bull 上就永远补不回来（审计实测：越权牛 70 天里 70 天都空着）。
                elif up and h.shares < tgt and reserve > 0:
                    close = px_now(code, d)
                    if pd.isna(close):
                        continue
                    price = close * (1 + slip_buy())
                    want = tgt - h.shares
                    afford = int(reserve / (price * (1 + COMMISSION))) // LOT * LOT
                    sh = min(want, afford)
                    if sh <= 0:
                        continue
                    cost = sh * price
                    fee = buy_fee(cost, cfg.fees_on, stock=True)
                    if cost + fee > reserve:
                        continue
                    h.shares += sh
                    reserve -= cost + fee
                    trades.append(TradeRec(d, code, "长期回补", sh, price, cost, fee, 0.0,
                                           "宏观风控：上行期补回长期股"))

        # ---------- 1) 择时股卖出：跌破 MA20 ----------
        for code in list(sig_hold):
            h = sig_hold[code]
            row = aligned[code].loc[d]
            close = px_now(code, d)
            if pd.isna(close) or pd.isna(row["ma20"]):
                continue
            if h.buy_date is not None and pd.Timestamp(h.buy_date) >= d:
                stats["skipped_t1"] += 1
                continue
            # 持有期下限：没拿够 N 个交易日，破位也不清（熔断期除外，风控优先）
            if (cfg.min_hold > 0 and not breaker_on
                    and code in buy_pos and pos - buy_pos[code] < cfg.min_hold):
                stats["skipped_minhold"] += 1
                continue
            # 大盘闸门：指数还在 MA20 上方（大盘没走坏）时，不因为个股破位就清仓。
            # 逻辑：牛市里个股最常见的动作就是「假破位再拉回」，一刀切卖掉等于
            #       把利润让给震荡；大盘走坏时才执行个股破位清仓。
            if cfg.exit_gate_market:
                im = idx_ma20.loc[d]
                if not pd.isna(im) and float(idx.loc[d]) > float(im):
                    stats.setdefault("skipped_market_gate", 0)
                    stats["skipped_market_gate"] += 1
                    continue
            if bear_on and code in bear_set:
                # 防御资产的退出规则只有一条：跌破自己的趋势闸门（极端风控的最后一道）
                bma = row.get("bear_ma")
                should_exit = not bool(row.get("above_bear", False))
                why = (f"防御资产跌破 MA{cfg.bear_ma} {float(bma):.2f}（切现金）"
                       if bma is not None and not pd.isna(bma) else "防御资产数据缺失（切现金）")
            elif etf_mode:
                mom = row.get("mom")
                mom_neg = pd.isna(mom) or float(mom) <= 0
                should_exit = mom_neg or not bool(row["above_ma20"])
                why = (f"动量 {float(mom):+.1%} 转负" if (not pd.isna(mom) and float(mom) <= 0)
                       else f"收盘 {close:.2f} 跌破 MA20 {row['ma20']:.2f}")
            else:
                should_exit = not bool(row["above_ma20"])
                why = f"收盘 {close:.2f} 跌破 MA20 {row['ma20']:.2f}"
            if should_exit:
                price = close * (1 - slip_sell())
                amount = h.shares * price
                fee = sell_fee(amount, cfg.fees_on, etf=etf_mode)
                sat_cash += amount - fee
                pnl = (price - h.cost) * h.shares - fee
                realized[code] = realized.get(code, 0.0) + pnl
                trades.append(TradeRec(d, code, "卖出", h.shares, price, amount, fee, pnl, why))
                last_exit[code] = pos
                del sig_hold[code]
                entry_fill.pop(code, None)
                entry_last.pop(code, None)
                buy_pos.pop(code, None)

        # ---------- 2) 选股 ----------
        cands: list[dict] = []
        if bear_mode:
            # 下行期：只在防御池里挑「站上 MA20 + 20 日波动率最低」的 bear_top 只。
            # 一只都不合格 → picked 为空 → 全现金（这就是「绝不硬扛」的落地方式）。
            for code in bear_pool:
                if code in sig_hold or code not in aligned:
                    continue
                row = aligned[code].loc[d]
                vol = row.get("vol20")
                if vol is None or pd.isna(vol):
                    continue
                if not bool(row.get("above_bear", False)):
                    continue          # 跌破自己的 MA60 → 不买（宁可拿现金）
                if code in last_exit and (pos - last_exit[code]) < cfg.cooldown:
                    stats["skipped_cooldown"] += 1
                    continue
                cands.append({"code": code, "mom": 0.0, "vol20": float(vol),
                              "vol": float(vol), "industry": bear_pool[code][1]})
            cands.sort(key=lambda c: c["vol"])
            room = max(0, cfg.bear_top - len(sig_hold))
            picked = [{"code": c["code"], "ker14": 0.0, "mom": 0.0,
                       "vol20": c["vol20"], "industry": c["industry"]}
                      for c in cands[:room]]
        elif etf_mode:
            # ETF 轮动：动量 > 0 且站上 MA20 的里，挑动量最强的 K 只（横截面排序）
            for code in sleeve_codes:
                if code in sig_hold:
                    continue
                if bear_on and code in bear_set:
                    continue        # 防御资产只在下行期持有，上行期别拿它当动量票
                row = aligned[code].loc[d]
                mom = row.get("mom")
                if cfg.trend_ma and not bool(row.get("above_trend", False)):
                    continue                      # 未站上长期均线（如年线）不入场
                if pd.isna(mom) or float(mom) <= 0 or not bool(row["above_ma20"]):
                    continue
                if code in last_exit and (pos - last_exit[code]) < cfg.cooldown:
                    stats["skipped_cooldown"] += 1
                    continue
                cands.append({"code": code, "mom": float(mom),
                              "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                              "industry": ETF_UNIVERSE.get(code, ("ETF", "其他"))[1]})
            cands.sort(key=lambda c: -c["mom"])
            room = max(0, cfg.etf_top - len(sig_hold))
            picked = [{"code": c["code"], "ker14": 0.0, "mom": c["mom"],
                       "vol20": c["vol20"], "industry": c["industry"]}
                      for c in cands[:room]]
        else:
            for code in sleeve_codes:
                if code in sig_hold:
                    continue
                if bear_on and code in bear_set:
                    continue        # 同上：防御资产不参与个股择时
                row = aligned[code].loc[d]
                if pd.isna(row["ma20"]) or pd.isna(row["ker14"]):
                    continue
                if not bool(row.get("golden_n", False)):
                    continue
                if float(row["ker14"]) <= cfg.ker_min:
                    continue
                if code in last_exit and (pos - last_exit[code]) < cfg.cooldown:
                    stats["skipped_cooldown"] += 1
                    continue
                cands.append({"code": code,
                              "ker14": float(row["ker14"]),
                              "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                              "industry": INDUSTRY_MAP.get(code, "其他")})

            # 行业上限 + 持仓数上限（复用 portfolio 的挑票逻辑）
            picked = pf.pick_candidates(cands, sig_hold, cfg)
            slots = cfg.max_positions - len(sig_hold)
            if slots <= 0:
                picked = []
            picked = picked[:max(0, slots)]

        # ---------- 3) 目标权重（只在卫星现金 + 择时腿市值里分配）----------
        sat_signal_equity = sat_cash + sig_val
        if (picked or sig_hold) and sat_signal_equity > 0:
            if bear_mode:
                # 防守期：防御资产等权吃满卫星腿。刻意不叠 exposure_mult——
                # 熔断是「进攻受限」，而防守期本来就是低仓位风险，再打折等于不设防。
                codes_now = list(sig_hold) + [p["code"] for p in picked if p["code"] not in sig_hold]
                n = max(1, len(codes_now))
                weights = {c: 1.0 / n for c in codes_now}
            elif etf_mode:
                # ETF：等权持有 K 只（单票仍受 max_weight 约束，超出的留现金）
                codes_now = list(sig_hold) + [p["code"] for p in picked if p["code"] not in sig_hold]
                n = max(1, len(codes_now))
                # 单只 ETF 的上限要按「总资产」换算：max_weight=10% 指的是总仓位的 10%，
                # 换到卫星账本要除以卫星占比（25% → 40%）。旧的写法直接拿 10% 当账本比例，
                # 3 只 ETF 只吃掉 30% 的卫星现金、剩下七成常年闲置——牛市吃不饱的元凶。
                sleeve_share = max(1e-6, sat_signal_equity / max(eq, 1e-9))
                cap = min(1.0, cfg.max_weight / sleeve_share)
                weights = {c: min(cap, 1.0 / n) * exposure_mult for c in codes_now}
            else:
                pool: list[dict] = []
                for code, h in sig_hold.items():
                    row = aligned[code].loc[d]
                    if pd.isna(row["ker14"]):
                        continue
                    pool.append({"code": code, "ker14": float(row["ker14"]),
                                 "vol20": None if pd.isna(row["vol20"]) else float(row["vol20"]),
                                 "industry": h.industry})
                pool += picked
                weights = pf.target_weights(pool, cfg)
                weights = {k: v * exposure_mult for k, v in weights.items()}

            # 分批建仓：新仓先只买目标的 1/K，之后每 G 个交易日补一批，直到建满。
            # 目的：别在一个价格上一次性打满，避免「金叉当天买在最高点」。
            if cfg.tranches > 1:
                gap = max(1, cfg.tranche_gap)
                for code in list(weights):
                    filled = entry_fill.get(code)
                    if filled is None:
                        filled = 1                       # 第一批（今天就买）
                        entry_last[code] = pos
                    elif (filled < cfg.tranches
                          and pos - entry_last.get(code, pos) >= gap):
                        filled += 1                      # 补下一批
                        entry_last[code] = pos
                    entry_fill[code] = filled
                    weights[code] = weights[code] * (filled / cfg.tranches)

            # 先卖超配
            for code, w in sorted(weights.items(), key=lambda kv: kv[1]):
                if code not in sig_hold:
                    continue
                h = sig_hold[code]
                close = px_now(code, d)
                if pd.isna(close):
                    continue
                if blocked(code, d, "sell"):
                    stats["skipped_limit"] += 1
                    continue
                if h.buy_date is not None and pd.Timestamp(h.buy_date) >= d:
                    stats["skipped_t1"] += 1
                    continue
                delta = w * sat_signal_equity - h.shares * close
                if delta >= 0 or -delta < cfg.band * eq:
                    continue
                sh = lot(min(h.shares, -delta / close))
                sh = cap_shares(code, d, close, sh)
                if sh <= 0:
                    continue
                _imp = impact_bps(code, d, sh * close)
                price = close * (1 - (slip_sell() + _imp))
                amount = sh * price
                fee = sell_fee(amount, cfg.fees_on, etf=etf_mode)
                sat_cash += amount - fee
                pnl = (price - h.cost) * sh - fee
                realized[code] = realized.get(code, 0.0) + pnl
                h.shares -= sh
                trades.append(TradeRec(d, code, "卖出", sh, price, amount, fee, pnl,
                                       f"减仓到目标 {w:.1%}"))
                if h.shares <= 0:
                    last_exit[code] = pos
                    del sig_hold[code]
                    entry_fill.pop(code, None)
                    entry_last.pop(code, None)
                    buy_pos.pop(code, None)

            # 再买低配
            for code, w in sorted(weights.items(), key=lambda kv: -kv[1]):
                close = px_now(code, d)
                if pd.isna(close):
                    continue
                if blocked(code, d, "buy"):
                    stats["skipped_limit"] += 1
                    continue
                h = sig_hold.get(code)
                cur = h.shares * close if h else 0.0
                delta = w * sat_signal_equity - cur
                if delta <= 0 or delta < cfg.band * eq:
                    continue
                sh = lot(min(delta, sat_cash) / (close * (1 + slip_buy())))
                _avail = lot(sat_cash / (close * (1 + slip_buy())))
                sh = cap_shares(code, d, close, min(sh, _avail), _avail)
                if sh <= 0:
                    continue
                _imp = impact_bps(code, d, sh * close)
                price = close * (1 + (slip_buy() + _imp))
                amount = sh * price
                fee = buy_fee(amount, cfg.fees_on)
                if amount + fee > sat_cash:
                    continue
                sat_cash -= amount + fee
                if h:
                    h.cost = (h.cost * h.shares + price * sh) / (h.shares + sh)
                    h.shares += sh
                else:
                    sig_hold[code] = Holding(code, sh, price, d, INDUSTRY_MAP.get(code, "其他"))
                    buy_pos[code] = pos
                trades.append(TradeRec(d, code, "买入", sh, price, amount, fee, 0.0,
                                       (f"防御配置（低波动避险）→ 目标 {w:.1%}" if bear_mode
                                        else f"动量轮动 → 目标 {w:.1%}" if etf_mode
                                        else f"金叉+确认{cfg.confirm}日 → 目标 {w:.1%}")))

        if os.environ.get("CS_AUDIT") == "1":
            _d = str(d.date())
            assert sat_cash >= -1e-6, f"{_d} 卫星现金为负：{sat_cash:,.2f}"
            assert reserve >= -1e-6, f"{_d} 宏观储备为负：{reserve:,.2f}"
            assert core_shares >= -1e-9, f"{_d} 核心份数为负：{core_shares}"
            for _c, _h in list(lt_hold.items()):
                assert _h.shares >= 0, f"{_d} 长期 {_c} 股数为负"
                assert _h.shares % LOT == 0, f"{_d} 长期 {_c} 不是整手：{_h.shares}"
                assert _c in set(cfg.longterm), f"{_d} 长期池外标的：{_c}"
            for _c, _h in list(sig_hold.items()):
                assert _h.shares >= 0, f"{_d} 择时 {_c} 股数为负"
                assert _h.shares % LOT == 0, f"{_d} 择时 {_c} 不是整手：{_h.shares}"
                assert _c in aligned, f"{_d} 择时标的不在行情表里：{_c}"

    # ---------- 期末快照（在强平之前拍，模拟盘/实盘对照用）----------
    _end_state = {
        "date": str(dates[-1].date()),
        "core_shares": float(core_shares),
        "longterm": {c: int(h.shares) for c, h in lt_hold.items()},
        "lt_cost": {c: float(h.cost) for c, h in lt_hold.items()},
        "tactical": {c: int(h.shares) for c, h in sig_hold.items()},
        "tactical_cost": {c: float(h.cost) for c, h in sig_hold.items()},
        "sat_cash": float(sat_cash),
        "reserve": float(reserve),
        "equity": float(eq_curve[-1][1]) if eq_curve else float(cfg.capital),
        "breaker_on": bool(breaker_on),
        "bull_state": bool(bull_state) if (bear_on and bull_state is not None) else True,
        "hmm_prob": float(prob_hmm.get(dates[-1], 0.0)) if prob_hmm is not None else 0.0,
        # 供审计/模拟盘复算：期末收盘价与期初资金
        "index_close": float(idx.loc[dates[-1]]),
        "last_close": {c: float(aligned[c].loc[dates[-1], "close"])
                       for c in set(list(lt_hold) + list(sig_hold)) if c in aligned},
        "capital": float(cfg.capital),
    }

    # ---------- 期末平掉择时股（长期股与核心不动，展示真实持有）----------
    dN = dates[-1]
    for code in list(sig_hold):
        h = sig_hold[code]
        close = px_now(code, dN)
        if pd.isna(close):
            continue
        price = close * (1 - slip_sell())
        amount = h.shares * price
        fee = sell_fee(amount, cfg.fees_on)
        sat_cash += amount - fee
        pnl = (price - h.cost) * h.shares - fee
        realized[code] = realized.get(code, 0.0) + pnl
        trades.append(TradeRec(dN, code, "卖出", h.shares, price, amount, fee, pnl, "期末平仓"))
        del sig_hold[code]

    # ---------- 汇总 ----------
    eq_series = pd.Series({d: v for d, v in eq_curve}).sort_index()
    core_series = pd.Series({d: v for d, v in core_curve}).sort_index()
    exp_series = pd.Series({d: v for d, v in exp_curve}).sort_index()

    total_fee = sum(t.fee for t in trades)
    closed = [t for t in trades if t.side == "卖出"]
    realized_trades = sum(t.pnl for t in closed)
    unrealized = (eq_series.iloc[-1] - cfg.capital) - interest_acc - realized_trades
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    payoff = (sum(t.pnl for t in wins) / len(wins)) / abs(sum(t.pnl for t in losses) / len(losses)) \
        if wins and losses else float("nan")
    daily_ret = eq_series.pct_change().dropna()
    n = len(eq_series)
    years = n / 243.0
    total_ret = eq_series.iloc[-1] / cfg.capital - 1
    cagr = (eq_series.iloc[-1] / cfg.capital) ** (1 / years) - 1 if years > 0 else 0.0
    run_max = eq_series.cummax()
    mdd = float((eq_series / run_max - 1).min())
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(243)) if daily_ret.std() > 0 else 0.0

    # 基准：沪深300 全仓买入持有 + 静态 50/50（年初再平衡）
    idx_ret = idx.iloc[-1] / idx.iloc[0] - 1
    core_ret = core_series.iloc[-1] / core_series.iloc[0] - 1 if core_series.iloc[0] else 0.0
    # 核心真实收益 = 期末持仓市值 + 期间净现金流 - 期初市值（V4.6 转防守清仓时末值=0，不能被读成「亏 100%」）
    _core_pnl = core_series.iloc[-1] + core_cash_net - core_series.iloc[0]
    core_ret = _core_pnl / core_series.iloc[0] if core_series.iloc[0] else 0.0

    metrics = {
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "days": n, "capital": cfg.capital,
        "final_equity": round(float(eq_series.iloc[-1]), 2),
        "total_return": round(float(total_ret) * 100, 2),
        "cagr": round(float(cagr) * 100, 2),
        "max_drawdown": round(mdd * 100, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(float(cagr) / abs(mdd), 2) if mdd < 0 else 0.0,
        "win_rate": round(win_rate * 100, 2),
        "payoff": round(float(payoff), 2) if payoff == payoff else None,
        "trades": len(trades), "closed_trades": len(closed),
        "fees": round(total_fee, 2),
        "interest": round(float(interest_acc), 2),
        "realized_pnl": round(float(realized_trades), 2),
        "unrealized_pnl": round(float(unrealized), 2),
        "avg_exposure": round(float(exp_series.mean()) * 100, 2),
        "max_exposure": round(float(exp_series.max()) * 100, 2),
        "core_return": round(float(core_ret) * 100, 2),
        "index_return": round(float(idx_ret) * 100, 2),
        "core_weight": cfg.core_weight, "lt_share": cfg.lt_share,
        "cooldown": cfg.cooldown, "confirm": cfg.confirm,
        "breaker_events": len(breaker_events), "breaker_days": breaker_days,
        "regime": cfg.regime, "bear_days": stats.get("bear_days", 0),
        "regime_switches": stats.get("regime_switches", 0),
        "regime_flips": regime_flips,
        "skipped_t1": stats["skipped_t1"],
        "skipped_limit": stats.get("skipped_limit", 0),
        "shrunk_impact": stats.get("shrunk_impact", 0), "skipped_cooldown": stats["skipped_cooldown"],
        "params": {"band": cfg.band, "stamp": cfg.stamp, "fees_on": cfg.fees_on,
                   "max_weight": cfg.max_weight, "industry_max": cfg.industry_max,
                   "max_positions": cfg.max_positions},
    }

    by_stock: dict[str, float] = {}
    for code, v in realized.items():
        by_stock[code] = round(v, 2)
    by_ind: dict[str, float] = {}
    for code, v in realized.items():
        ind = INDUSTRY_MAP.get(code, "其他")
        by_ind[ind] = round(by_ind.get(ind, 0.0) + v, 2)

    return {"metrics": metrics, "trades": trades, "realized": by_stock,
            "by_industry": by_ind, "equity": eq_series, "exposure": exp_series,
            "core": core_series, "index": idx, "breaker_events": breaker_events,
            "end_state": _end_state, "config": asdict(cfg)}


# ============================================================================
# 六、报告与图
# ============================================================================
def write_report(path: str, res: dict) -> str:
    m = res["metrics"]
    cfg = res["config"]
    L: list[str] = []
    L.append(f"# 核心-卫星组合回测报告（{m['start']} ~ {m['end']}）\n")
    # 版本+关键参数强制打在报告头：报告脱离上下文后，也能看出是哪一版引擎、防守时怎么处理底仓
    _cb = cfg.get("core_bear", "-")
    _lb = cfg.get("lt_bear", "-")
    L.append(f"> **引擎 {ENGINE_VERSION}**｜core_bear={_cb}｜lt_bear={_lb}"
             f"｜cash_yield={cfg.get('cash_yield', 0.0):.1%}"
             f"｜核心占比 {cfg.get('core_weight', 0.0):.0%}｜HMM {'开' if cfg.get('hmm') else '关'}  ")
    L.append(f"> {ENGINE_VERSION} 要点：{VERSION_NOTES.get(ENGINE_VERSION, '')}\n")
    L.append("## 结构\n")
    L.append(f"- 核心账本 **{cfg['core_weight']:.0%}**：{CORE_NAME} 指数，期初买入、持有不动")
    L.append(f"- 卫星账本 **{1 - cfg['core_weight']:.0%}**：其中长期股 {cfg['lt_share']:.0%}"
             f"（{len(cfg['longterm'])} 只买入持有）、择时股 {1 - cfg['lt_share']:.0%}")
    L.append(f"- 摩擦闸门：冷却期 {cfg['cooldown']} 日、连续 {cfg['confirm']} 日确认、死区 {cfg['band']:.1%}")
    if cfg["tranches"] <= 1:
        entry_desc = "一次性打满"
    else:
        entry_desc = f"分 {cfg['tranches']} 批建仓、每 {cfg['tranche_gap']} 日补一批"
    hold_desc = "关闭" if cfg["min_hold"] <= 0 else f"{cfg['min_hold']} 个交易日"
    L.append(f"- 进场节奏：{entry_desc}；持有期下限 {hold_desc}")
    if cfg.get("regime", "off") != "off":
        pool_desc = ("国债+黄金" if cfg.get("bear_pool", "bond_gold") == "bond_gold"
                     else "国债+黄金+红利")
        L.append(f"- **状态切换（V3.1）**：沪深300 收盘 > MA200 走上行模型；< MA200 切防守模型"
                 f"（{pool_desc}里取波动率最低的 {cfg.get('bear_top', BEAR_TOP)} 只，"
                 f"全部跌破 MA{cfg.get('bear_ma', BEAR_MA)} 则 100% 现金；"
                 f"切换需连续 {cfg.get('regime_confirm', 1)} 日越线）。"
                 f"本期下行 {m.get('bear_days', 0)} 天、换挡卖出 {m.get('regime_switches', 0)} 笔、状态翻转 {m.get('regime_flips', 0)} 次")
    _cb = cfg.get("core_bear", "hold")
    _lb = cfg.get("lt_bear", "hold")
    if cfg.get("regime", "off") != "off" and (_cb != "hold" or _lb != "hold"):
        _map = {"half": "减半", "zero": "清仓", "hold": "死拿"}
        L.append(f"- **全局宏观风控（V3.2）**：下行期核心底仓{_map.get(_cb,'减半')}、长期股{_map.get(_lb,'减半')}"
                 f"（卖出所得进储备，上行期回补）；本期宏观减仓 {m.get('macro_cut_days', 0)} 个交易日")
    else:
        L.append(f"- 核心账本不参与状态切换（压舱石永远在场）")
    L.append(f"- 费用：佣金万 2.5（最低 5 元）、滑点 {SLIPPAGE:.2%}、印花税 {cfg['stamp']:.3%}（指数免）\n")

    L.append("## 绩效\n")
    L.append("| 指标 | 数值 |\n|---|---:|")
    rows = [
        ("期末资产", f"{m['final_equity']:,.2f} 元"),
        ("总收益", f"{m['total_return']:+.2f}%"),
        ("年化收益", f"{m['cagr']:+.2f}%"),
        ("最大回撤", f"{m['max_drawdown']:.2f}%"),
        ("夏普比率", f"**{m['sharpe']:.2f}**"),
        ("卡玛比率", f"{m['calmar']:.2f}"),
        ("胜率", f"{m['win_rate']:.2f}%"),
        ("盈亏比（均值）", f"{m['payoff']}" if m["payoff"] is not None else "—"),
        ("交易笔数 / 平仓", f"{m['trades']} / {m['closed_trades']}"),
        ("手续费", f"{m['fees']:,.2f} 元"),
        ("平均仓位 / 最高", f"{m['avg_exposure']:.2f}% / {m['max_exposure']:.2f}%"),
        ("核心账本收益", f"{m['core_return']:+.2f}%"),
        ("沪深300 基准", f"{m['index_return']:+.2f}%"),
        ("熔断", f"{m['breaker_events']} 次 / 熔断中 {m['breaker_days']} 天"),
        ("T+1 拦截 / 冷却拦截", f"{m['skipped_t1']} / {m['skipped_cooldown']}"),
    ]
    for k, v in rows:
        L.append(f"| {k} | {v} |")

    if m["sharpe"] >= SHARPE_LIVE:
        verdict = f"✅ 夏普 {m['sharpe']:.2f} ≥ {SHARPE_LIVE:.2f}：可上小仓验证（≤10% 资金，跑满 12 个月实盘对照）"
    elif m["sharpe"] >= SHARPE_WATCH:
        verdict = f"⛔ 夏普 {m['sharpe']:.2f} 在 {SHARPE_WATCH:.2f}~{SHARPE_LIVE:.2f}：继续模拟盘，不许调参"
    else:
        verdict = f"💀 夏普 {m['sharpe']:.2f} < {SHARPE_WATCH:.2f}：结构有问题，回滚上一版"
    L.append(f"\n> **{verdict}**\n")

    L.append("## 熔断记录\n")
    if res["breaker_events"]:
        for e in res["breaker_events"]:
            L.append(f"- {e['date']}　{e['event']}（回撤 {e['drawdown']}%）→ {e['action']}")
    else:
        L.append("- 无")

    L.append("\n## 按行业归因（已实现盈亏）\n")
    L.append("| 行业 | 盈亏（元） |\n|---|---:|")
    for ind, v in sorted(res["by_industry"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {ind} | {v:+,.2f} |")

    L.append("\n## 交易明细\n")
    L.append("| 日期 | 代码 | 方向 | 股数 | 价格 | 金额 | 费用 | 盈亏 | 备注 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for t in res["trades"]:
        L.append(f"| {t.date.date()} | {t.code} | {t.side} | {t.shares} | {t.price:.2f} | "
                 f"{t.amount:,.0f} | {t.fee:.2f} | {t.pnl:+,.2f} | {t.note} |")

    L.append("\n## 已知局限\n")
    L.append("1. 核心用**指数点位**模拟持有，没有计 ETF 的跟踪误差、申赎费、分红再投资；")
    L.append("2. 长期股是「期初一次性买入」，没有分批建仓，买入时点运气成分较大；")
    L.append("3. 择时股成交价 = 当日收盘 ± 滑点，未建模涨跌停/停牌；")
    L.append("4. 熔断按总资产算、只压择时股仓位——这是设计选择：核心是压舱石，不该被砍；")
    L.append("5. 15+5 的股票池是人工挑的，存在选择偏差；行业上限只约束「只数」，不约束相关性。")
    txt = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path


def plot(res: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    eq = res["equity"]
    idx = res["index"].reindex(eq.index).ffill()
    idx_n = idx / idx.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.1, 0.9]})
    ax = axes[0]
    ax.plot(eq.index, eq / eq.iloc[0], lw=1.6, color="#4C72B0", label="核心-卫星")
    ax.plot(idx_n.index, idx_n, lw=1.2, ls="--", color="#999999", label=f"{CORE_NAME}（全仓）")
    core_n = res["core"] / res["core"].iloc[0]
    ax.plot(core_n.index, core_n, lw=0.9, ls=":", color="#C44E52", label="核心账本")
    ax.axhline(1.0, color="#cccccc", lw=0.8)
    ax.set_ylabel("净值（起点 = 1）")
    ax.set_title(f"核心-卫星组合 {res['metrics']['start']} ~ {res['metrics']['end']}"
                 f"｜收益 {res['metrics']['total_return']:+.2f}%"
                 f"｜夏普 {res['metrics']['sharpe']}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    dd = (eq / eq.cummax() - 1) * 100
    ax.fill_between(dd.index, dd, 0, color="#C44E52", alpha=0.3)
    ax.axhline(-10, color="#888888", ls="--", lw=0.9, label="熔断线 −10%")
    ax.set_ylabel("回撤（%）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    exp = res["exposure"].reindex(eq.index).ffill() * 100
    ax.fill_between(exp.index, exp, 0, color="#4C72B0", alpha=0.35)
    ax.set_ylabel("仓位（%）")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ============================================================================
# 七、自检
# ============================================================================
def selftest() -> int:
    checks = []
    checks.append(("买入 10 万佣金 = 25 元", abs(buy_fee(100_000, True) - 25.0) < 1e-6, ""))
    checks.append(("买入 1000 元仍收最低 5 元", abs(buy_fee(1_000, True) - 5.0) < 1e-6, ""))
    _STAMP[0] = 0.001
    checks.append(("股票卖出 10 万 = 25 佣金 + 100 印花 + 1 过户 = 126 元",
                   abs(sell_fee(100_000, True) - 126.0) < 1e-6, f"{sell_fee(100_000, True)}"))
    checks.append(("指数卖出 10 万免印花税/过户 = 25 元",
                   abs(sell_fee(100_000, True, etf=True) - 25.0) < 1e-6,
                   f"{sell_fee(100_000, True, etf=True)}"))
    checks.append(("整手取整 1234 → 1200", lot(1234) == 1200, f"{lot(1234)}"))
    checks.append(("核心占比默认 20%（V4.6）", abs(CORE_WEIGHT - 0.20) < 1e-9, f"{CORE_WEIGHT:.2f}"))
    checks.append(("卫星 = 1 − 核心 = 80%", abs((1 - CORE_WEIGHT) - 0.80) < 1e-9, ""))
    checks.append(("长期股 5 只、普通股 15 只",
                   len(DEFAULT_LONGTERM) == 5 and len(DEFAULT_REGULAR) == 15,
                   f"{len(DEFAULT_LONGTERM)}/{len(DEFAULT_REGULAR)}"))
    checks.append(("行业表覆盖全部标的",
                   all(c in INDUSTRY_MAP for c in list(DEFAULT_REGULAR) + list(DEFAULT_LONGTERM)), ""))
    ok = 0
    print("=== core_satellite.py 自检 ===")
    for name, passed, detail in checks:
        print(f"{'✅' if passed else '❌'} {name}" + (f"　（{detail}）" if detail and not passed else ""))
        ok += 1 if passed else 0
    print(f"\n{ok}/{len(checks)} 项通过")
    return 0 if ok == len(checks) else 1


# ============================================================================
# 八、CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="核心-卫星组合：50% 宽基打底 + 50% 择时卫星")
    p.add_argument("--version", action="store_true",
                   help="打印引擎版本与关键默认参数后退出（每天跑之前先看一眼）")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--core", type=float, default=CORE_WEIGHT, help="核心账本占比，默认 0.20（V4.6）")
    p.add_argument("--lt-share", type=float, default=LT_SHARE, help="卫星里长期股的占比")
    p.add_argument("--max-weight", type=float, default=MAX_WEIGHT)
    p.add_argument("--industry-max", type=int, default=INDUSTRY_MAX)
    p.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    p.add_argument("--band", type=float, default=BAND)
    p.add_argument("--cooldown", type=int, default=COOLDOWN, help="卖出后冷却交易日数")
    p.add_argument("--confirm", type=int, default=CONFIRM, help="金叉需连续成立的天数")
    p.add_argument("--min-hold", type=int, default=MIN_HOLD,
                   help="持有期下限（交易日），没拿够不许破位清仓；0=关闭")
    p.add_argument("--tranches", type=int, default=TRANCHES,
                   help="分批建仓的批数，1=一次性打满")
    p.add_argument("--tranche-gap", type=int, default=TRANCHE_GAP,
                   help="相邻两批之间至少间隔的交易日")
    p.add_argument("--exit-gate-market", action="store_true",
                   help="大盘（沪深300）在 MA20 上方时，不因个股破位清仓")
    p.add_argument("--sleeve", choices=("stock", "etf"), default="etf",
                   help="择时腿：etf=行业 ETF 动量轮动（默认）；stock=个股金叉择时（旧版基线）")
    p.add_argument("--etf-top", type=int, default=ETF_TOP, help="ETF 轮动持有的只数")
    p.add_argument("--etf-mom", type=int, default=ETF_MOM, help="ETF 动量回看交易日数")
    p.add_argument("--etf-pool", choices=("12", "15", "17", "20"), default="20",
                   help="ETF 候选池：12=扩容前，15=去重密集池，17=20只去掉宽基，20=全部（默认）")
    p.add_argument("--trend-ma", type=int, default=0,
                   help="ETF 入场需站上的长期均线（如 250=年线）；0=关闭（默认）")
    p.add_argument("--regime", choices=("off", "ma200"), default="ma200",
                   help="状态切换：off=V2 永续进攻；ma200=沪深300 站上 MA200 才进攻，跌破切防守腿")
    p.add_argument("--bear-top", type=int, default=BEAR_TOP,
                   help="下行期持有几只防御资产（默认 2）")
    p.add_argument("--bear-ma", type=int, default=BEAR_MA,
                   help="防御资产的趋势闸门均线（默认 60；全部跌破则空仓）")
    p.add_argument("--bear-pool", choices=("bond_gold", "all"), default="bond_gold",
                   help="防御池：bond_gold=国债+黄金（默认，V3.1）；all=再加红利（V3.0 旧口径）")
    p.add_argument("--regime-confirm", type=int, default=REGIME_CONFIRM,
                   help="状态切换需连续 N 日越过 MA200 才算数（默认 2，1=当日即切）")
    p.add_argument("--regime-slope", type=int, default=0,
                   help="年线斜率窗口（交易日），0=只看是否跌破年线、不看斜率")
    p.add_argument("--regime-slope-min", type=float, default=0.0,
                   help="斜率低于此值才认定「年线拐头」，默认 0（走平或向下）")
    p.add_argument("--hmm", dest="hmm", action="store_true", default=True,
                   help="用 HMM 牛市概率强行满仓（默认开启）")
    p.add_argument("--no-hmm", dest="hmm", action="store_false",
                   help="关闭 HMM 闸门，回到 V3.3 行为")
    p.add_argument("--hmm-exit", type=float, default=0.70,
                   help="HMM 退出阈值（滞后带），默认 0.70")
    p.add_argument("--hmm-confirm", type=int, default=1,
                   help="HMM 进/出各需连续确认天数，默认 2")
    p.add_argument("--breaker-max-days", type=int, default=120,
                   help="熔断最长天数，到点强制解锁（0=不限），默认 120")
    p.add_argument("--hmm-threshold", type=float, default=0.85, help="牛市概率阈值（默认 0.85）")
    p.add_argument("--no-hmm-guard", dest="hmm_guard", action="store_false",
                   help="去掉「价在年线上方」的二次确认（仅做实验）")
    p.add_argument("--core-bear", choices=("hold", "half", "zero"), default="zero",
                   help="宏观风控：转防守时核心底仓处理（V4.6 默认 zero=清仓；--core-bear half 回到 V4.5）")
    p.add_argument("--lt-bear", choices=("hold", "half", "zero"), default="zero",
                   help="宏观风控：转防守时长期股处理（V4.6 默认 zero=清仓；--lt-bear half 回到 V4.5）")
    p.add_argument("--regime-band", type=float, default=0.03,
                   help="状态缓冲带（如 0.02 = 跌破年线 2%% 才转防守、站上 2%% 才回进攻），防年线附近来回抽插")
    p.add_argument("--dd-trigger", type=float, default=DD_TRIGGER)
    p.add_argument("--dd-factor", type=float, default=DD_FACTOR)
    p.add_argument("--dd-recover", type=float, default=DD_RECOVER)
    p.add_argument("--stamp", type=float, default=STAMP)
    p.add_argument("--commission", type=float, default=COMMISSION, help="佣金费率，默认 0.00025（万 2.5）")
    p.add_argument("--commission-min", type=float, default=COMMISSION_MIN, help="单笔最低佣金，默认 5 元")
    p.add_argument("--transfer", type=float, default=TRANSFER, help="过户费费率，默认 0.00001（0.001%%）")
    p.add_argument("--impact", type=float, default=0.0, help="固定冲击成本（单边附加滑点），如 0.0002 = 加 2bp")
    p.add_argument("--impact-coef", type=float, default=0.0,
                   help="按成交量占比的冲击系数（下单额/20日均额 × coef），如 0.05 = 占比 100%% 时 5%%")
    p.add_argument("--impact-max-part", type=float, default=0.05,
                   help="下单额超过 20 日均额该比例时拒单，默认 0.05")
    p.add_argument("--stamp-old", type=float, default=0.001, help="2023-08-28 之前的印花税率")
    p.add_argument("--no-stamp-history", action="store_true", help="关闭印花税按日期分段（全程用 --stamp）")
    p.add_argument("--vol-target", type=float, default=0.10,
                   help="目标年化波动（如 0.10）；0=关闭。开启后按目标/实现波动缩放总仓位")
    p.add_argument("--vol-floor", type=float, default=0.30, help="仓位缩放下限，默认 0.30")
    p.add_argument("--vol-band", type=float, default=0.05, help="核心调仓死区（占核心原始份额），默认 1%%")
    p.add_argument("--cash-yield", type=float, default=0.02,
                   help="闲置现金年化收益（如 0.015 = 货币基金 1.5%），默认 0 不计")
    p.add_argument("--no-fees", action="store_true")
    p.add_argument("--out", default="core_satellite_report.md")
    p.add_argument("--chart", default="core_satellite.png")
    p.add_argument("--json", action="store_true")
    p.add_argument("--selftest", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        _d = Config()
        print(f"core_satellite.py  {ENGINE_VERSION}")
        print(f"  要点：{VERSION_NOTES.get(ENGINE_VERSION, '')}")
        print(f"  默认：core_bear={_d.core_bear}／lt_bear={_d.lt_bear}"
              f"／cash_yield={_d.cash_yield:.1%}／核心占比={_d.core_weight:.0%}／HMM={'开' if _d.hmm else '关'}")
        print("  ⚠️ 同目录可能还有 core_satellite_v45.py（V4.5，core_bear=half）；"
              "paper.py / daily_report.py import 的是本文件 core_satellite.py")
        return 0
    if args.selftest:
        return selftest()

    cfg = Config(start=args.start, end=args.end, capital=args.capital,
                 core_weight=args.core, lt_share=args.lt_share,
                 max_weight=args.max_weight, industry_max=args.industry_max,
                 max_positions=args.max_positions, band=args.band,
                 cooldown=args.cooldown, confirm=args.confirm,
                 min_hold=args.min_hold, tranches=args.tranches,
                 tranche_gap=args.tranche_gap,
                 exit_gate_market=args.exit_gate_market,
                 sleeve=args.sleeve, etf_top=args.etf_top, etf_mom=args.etf_mom, etf_pool=args.etf_pool,
                 trend_ma=args.trend_ma,
                 regime=args.regime, bear_top=args.bear_top, bear_ma=args.bear_ma,
                 regime_band=args.regime_band, regime_confirm=args.regime_confirm,
                 regime_slope=args.regime_slope, regime_slope_min=args.regime_slope_min,
                 bear_pool=args.bear_pool, core_bear=args.core_bear, lt_bear=args.lt_bear,
        hmm=args.hmm, hmm_threshold=args.hmm_threshold,
        hmm_ma200_guard=args.hmm_guard,
        hmm_exit=args.hmm_exit, hmm_confirm=args.hmm_confirm,
        breaker_max_days=args.breaker_max_days,
                 dd_trigger=args.dd_trigger, dd_factor=args.dd_factor,
                 dd_recover=args.dd_recover, fees_on=not args.no_fees,
                 stamp=args.stamp, cash_yield=args.cash_yield,
                 vol_target=args.vol_target, vol_floor=args.vol_floor, vol_band=args.vol_band,
                 commission=args.commission, commission_min=args.commission_min,
                 transfer=args.transfer, impact=args.impact,
                 impact_coef=args.impact_coef, impact_max_part=args.impact_max_part,
                 stamp_old=args.stamp_old, stamp_history=not args.no_stamp_history)
    # 跑之前先自报家门：版本 + 关键参数，避免把 V4.5/V4.6 的行为混着解释
    print(f"【{version_banner(cfg)}】", file=sys.stderr)
    res = run(cfg)
    m = res["metrics"]
    write_report(args.out, res)
    plot(res, args.chart)

    print(f"期末 {m['final_equity']:,.2f} 元｜收益 {m['total_return']:+.2f}%"
          f"｜回撤 {m['max_drawdown']:.2f}%｜夏普 {m['sharpe']:.2f}"
          f"｜卡玛 {m['calmar']:.2f}｜胜率 {m['win_rate']:.2f}%"
          f"｜仓位均值 {m['avg_exposure']:.1f}%"
          f"｜指数基准 {m['index_return']:+.2f}%")
    if m["sharpe"] >= SHARPE_LIVE:
        print(f"✅ 夏普 {m['sharpe']:.2f} ≥ {SHARPE_LIVE:.2f} → 可上小仓验证（≤10% 资金，实盘对照 12 个月）")
    elif m["sharpe"] >= SHARPE_WATCH:
        print(f"⛔ 夏普 {m['sharpe']:.2f} 在 {SHARPE_WATCH:.2f}~{SHARPE_LIVE:.2f} → 继续模拟盘，**不许调参**")
    else:
        print(f"💀 夏普 {m['sharpe']:.2f} < {SHARPE_WATCH:.2f} → 结构有问题，回滚上一版")
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
