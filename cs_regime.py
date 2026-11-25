"""裁判层：判断「今天该走上行模型还是防守模型」。

两个判据（V3.2.1）：
1. **跌破年线**：沪深300 收盘 < MA200（可加缓冲带，默认 0）
2. **年线拐头**：MA200 本身在走平/向下（`slope_win` 表示看几日斜率，
   斜率 < `slope_min`（默认 0）才算「拐头」）

两个条件同时满足才允许「认输」转防守；转回进攻只要站回年线（单条件）。
认输要两个证据，认错只要一个——否则 2024 年 9 月那种「年线还在下行、指数已经暴涨」
的行情会被判成熊市，整整错过一轮反弹。
"""
from __future__ import annotations

import pandas as pd

# ---------------- 防守池 ----------------
BEAR_POOL = {
    "511010": ("国债ETF", "债券"),
    "518880": ("黄金ETF", "贵金属"),
}
BEAR_POOL_ALL = {
    **BEAR_POOL,
    "510880": ("红利ETF", "红利"),
}
BEAR_TOP = 2
BEAR_MA = 60
REGIME_CONFIRM = 2


def bench_frame(close: pd.Series, window: int = 200, slope_win: int = 0) -> pd.DataFrame:
    """输入沪深300 收盘序列，输出 close / ma / slope（ma 的 N 日变化率）。"""
    ma = close.rolling(window, min_periods=window).mean()
    df = pd.DataFrame({"close": close, "ma": ma})
    if slope_win > 0:
        df["slope"] = df["ma"] / df["ma"].shift(slope_win) - 1.0
    else:
        df["slope"] = 0.0
    return df


def build_states(bench: pd.DataFrame, *, confirm: int = REGIME_CONFIRM, band: float = 0.0,
                 slope_win: int = 0, slope_min: float = 0.0) -> pd.Series:
    """逐日状态机：True=上行，False=下行。

    - 下行：收盘 < MA200×(1−band) 且（slope_win=0 或 MA200 斜率 < slope_min）
    - 上行：收盘 > MA200×(1+band)（不看斜率）
    - 连续 `confirm` 日满足才切换；条件一断计数清零
    """
    close = bench["close"].astype(float)
    ma = bench["ma"].astype(float)
    slope = bench["slope"].astype(float)
    use_slope = slope_win > 0

    bull = None
    bull_run = bear_run = 0
    out = []
    for i in range(len(bench)):
        c, m, s = close.iloc[i], ma.iloc[i], slope.iloc[i]
        if pd.isna(c) or pd.isna(m):
            out.append(True if bull is None else bool(bull))   # 数据断了沿用上次状态
            continue
        below = c < m * (1 - band)
        above = c > m * (1 + band)
        turning = (not use_slope) or (not pd.isna(s) and float(s) < slope_min)
        if bull is None:
            bull = not (below and turning)
            bull_run = bear_run = 0
        elif bull:
            if below and turning:
                bear_run += 1
                bull_run = 0
                if bear_run >= max(1, confirm):
                    bull, bear_run = False, 0
            else:
                bear_run = 0
        else:
            if above:
                bull_run += 1
                bear_run = 0
                if bull_run >= max(1, confirm):
                    bull, bull_run = True, 0
            else:
                bull_run = 0
        out.append(bool(bull))
    return pd.Series(out, index=bench.index, name="bull")


def count_flips(states: pd.Series) -> int:
    s = states.astype(bool)
    return max(0, int((s != s.shift(1)).sum() - 1))


def pick_defensive(rows: dict[str, pd.Series], pool: dict, top: int) -> list[dict]:
    """防守腿选品：站上自身趋势闸门的防御资产里，挑 20 日波动率最低的 `top` 只。

    `rows`：code -> 当日一行（含 vol20 / above_bear）。
    一只都不合格返回空列表 → 调用方拿现金，这就是「绝不硬扛」的落地。
    """
    cands = []
    for code in pool:
        row = rows.get(code)
        if row is None:
            continue
        vol = row.get("vol20")
        if vol is None or pd.isna(vol):
            continue
        if not bool(row.get("above_bear", False)):
            continue
        cands.append({"code": code, "vol": float(vol), "vol20": float(vol),
                      "industry": pool[code][1]})
    cands.sort(key=lambda c: c["vol"])
    return cands[:max(0, top)]
