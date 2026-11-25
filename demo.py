# -*- coding: utf-8 -*-
"""quant-core / demo -- 用合成数据跑通整条流水线，不联网、不需要任何数据文件。

为什么要这个文件：引擎本身只认 akshare 取回来的真实行情，面试官/陌生人不一定
有网络、也不一定有 A 股数据。这里先把 `cs_data` 的三个取数函数换成固定 seed 的
几何随机漫步，再 import 引擎（引擎在 import 时就把函数名绑定了，所以必须先换）。

    python demo.py            # 默认 2019-01-01 ~ 2020-12-31
    python demo.py --start 2018-01-01 --end 2020-06-30

注意：**这里跑的不是策略的真实绩效**，它只证明「框架能跑、状态机在切换、
风控与日报口径是连着的」。真实回测请用 akshare 数据。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import cs_data
from cs_regime import BEAR_POOL

WARMUP_START = "2016-01-01"
DATA_END = "2021-12-31"


# ---------------------------------------------------------------- 合成行情
def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range(WARMUP_START, DATA_END)


def _price_path(seed: int, drift: float, vol: float, base: float = 100.0) -> np.ndarray:
    idx = _calendar()
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))


def _etf_frame(seed: int, drift: float, vol: float, mom: int) -> pd.DataFrame:
    idx = _calendar()
    n = len(idx)
    close = _price_path(seed, drift, vol)
    rng = np.random.default_rng(seed + 7)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.integers(2_000_000, 9_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    c = df["close"]
    df["ma5"] = c.rolling(5).mean()
    df["ma20"] = c.rolling(20).mean()
    df["mom"] = c / c.shift(mom) - 1.0
    df["vol20"] = c.pct_change().rolling(20).std()
    df["above_ma20"] = c >= df["ma20"]
    df["above_trend"] = True
    df["golden"] = False
    df["golden_n"] = False
    df["trend_up"] = df["ma5"] > df["ma20"]
    return df


def _build_universe() -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    idx = _calendar()
    # 指数：慢牛 + 一段熊市，好让状态闸门真的切换
    n = len(idx)
    rng = np.random.default_rng(2026)
    trend = np.concatenate(
        [
            np.full(n // 3, 0.0008),
            np.full(n // 3, -0.0016),
            np.full(n - 2 * (n // 3), 0.0012),
        ]
    )
    index_close = 3000.0 * np.exp(np.cumsum(trend + rng.normal(0.0, 0.010, n)))
    index = pd.Series(index_close, index=idx, name="close")

    frames: dict[str, pd.DataFrame] = {}
    codes = list(cs_data.ETF_UNIVERSE.keys())
    for i, code in enumerate(codes):
        seed = 1000 + i
        drift = 0.0004 if i % 3 else 0.0001
        vol = 0.012 + 0.004 * (i % 4)
        frames[code] = _etf_frame(seed, drift, vol, mom=20)
    for j, code in enumerate(sorted(BEAR_POOL.keys()) if isinstance(BEAR_POOL, dict) else list(BEAR_POOL)):
        frames[code] = _etf_frame(5000 + j, 0.00015, 0.004, mom=20)
    return index, frames


_INDEX, _FRAMES = _build_universe()


# ---------------------------------------------------------------- 替换数据层
def _patched_load_index(start: str, end: str, path: str = "") -> pd.Series:
    return _INDEX[( _INDEX.index >= pd.Timestamp(start)) & (_INDEX.index <= pd.Timestamp(end))]


def _patched_load_etf(code: str, start: str, end: str, mom: int, trend_ma: int = 0) -> pd.DataFrame:
    df = _FRAMES[code].copy()
    if mom != 20:
        c = df["close"]
        df["mom"] = c / c.shift(mom) - 1.0
    return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def _patched_load_stocks(codes, end, adjust="hfq", start=None):
    s = pd.Timestamp(start or WARMUP_START)
    e = pd.Timestamp(end)
    out = {}
    for code in codes:
        if code in _FRAMES:
            out[code] = _FRAMES[code][(_FRAMES[code].index >= s) & (_FRAMES[code].index <= e)].copy()
    return out


cs_data.load_index = _patched_load_index
cs_data.load_etf = _patched_load_etf
cs_data.load_stocks = _patched_load_stocks

# 注意：这一行必须在 patch 之后 —— 引擎 `from cs_data import load_index, ...`
import core_satellite as cs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="quant-core demo（合成数据，不联网）")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2020-12-31")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    a = p.parse_args()

    cfg = cs.Config(
        start=a.start,
        end=a.end,
        capital=a.capital,
        sleeve="etf",
        longterm={},
        hmm=False,          # 合成数据不跑 HMM，避免把随机数拟合当成结论
        core_bear="zero",
    )
    print(f"[demo] {a.start} ~ {a.end}  (合成数据 · 不联网)")
    res = cs.run(cfg)

    m = res.get("metrics", {}) if isinstance(res, dict) else getattr(res, "metrics", {})
    print("\n[demo] 跑通了。以下数字来自合成数据，只用于验证流水线：")
    for key, label in (
        ("total_return", "总收益 %"),
        ("max_drawdown", "最大回撤 %"),
        ("sharpe", "Sharpe"),
        ("trades", "成交笔数"),
        ("regime_switches", "状态切换次数"),
        ("bear_days", "防守天数"),
    ):
        if key in m:
            print(f"  {label:<12} {m[key]}")
    print("\n[demo] 真实回测：python core_satellite.py --start 2010-01-01 --end 2025-12-31")
    return 0


if __name__ == "__main__":
    sys.exit(main())
