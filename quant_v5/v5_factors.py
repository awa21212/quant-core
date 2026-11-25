# -*- coding: utf-8 -*-
"""v5_factors.py —— P2 因子层。

约定
----
- 输入：v5_data 缓存好的行情（真实价 + 后复权价 + 成交额 + 换手率）
- 输出：一张长表 `factor_panel.csv`：date, code, f_rev20, f_turn, f_lowvol, f_mom120, ...
- 所有因子在**截面内**做 rank → z（去极值），不做时序标准化，避免混入时间信息。

为什么只用这几类
----------------
A 股截面里最稳的三类：短期反转、换手/流动性、中期动量。
价值/质量要等 PIT 财务表补齐（share 数还没下），先留接口。

用法
----
    python v5_factors.py --build            # 生成因子面板
    python v5_factors.py --ic               # 因子 IC 报告
    python v5_factors.py --ic --horizon 5   # 5 日前瞻
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


# ---------------------------------------------------------------- 面板装载
def load_panel(codes: list[str] | None = None, start: str = "2016-01-01",
               end: str | None = None) -> dict[str, pd.DataFrame]:
    """把每只票的行情拼成 {字段: DataFrame(index=date, columns=code)}。"""
    uni = D.universe()
    codes = codes or uni["code"].tolist()
    fields = ("close", "adj_close", "amount", "turnover", "volume")
    buf: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    used, skipped = 0, 0
    for code in codes:
        df = D.load_px(code, start=start, end=end)
        if df is None or len(df) < 130:
            skipped += 1
            continue
        s = df.set_index("date")
        for f in fields:
            if f in s.columns:
                buf[f][code] = s[f].astype("float64")
        used += 1
    out = {}
    for f, d in buf.items():
        if d:
            out[f] = pd.DataFrame(d).sort_index()
    print(f"[panel] {used} 只入池，{skipped} 只跳过（数据不足），"
          f"{len(out.get('close', []))} 个交易日", flush=True)
    return out


# ---------------------------------------------------------------- 因子
def _cs_rank_z(df: pd.DataFrame) -> pd.DataFrame:
    """截面 rank → 标准正态近似（-1..1 中心化），NaN 保持。"""
    r = df.rank(axis=1, pct=True)
    z = r.sub(0.5).mul(2.0)          # 映射到 [-1, 1]
    z = z.where(df.notna())
    return z.clip(-1, 1)


def build_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = panel["close"]
    adj = panel["adj_close"]
    amount = panel.get("amount")
    turnover = panel.get("turnover")

    ret = adj.pct_change(fill_method=None)
    F: dict[str, pd.DataFrame] = {}

    # 1) 短期反转：过去 20 日跌幅越大，分数越高
    F["f_rev20"] = -(adj / adj.shift(20) - 1.0)

    # 2) 换手异常：近 5 日换手 / 近 60 日换手，越低越好（拥挤度）
    if turnover is not None:
        t5 = turnover.rolling(5, min_periods=3).mean()
        t60 = turnover.rolling(60, min_periods=30).mean()
        F["f_turn"] = -(t5 / t60.replace(0, np.nan))
    elif amount is not None:
        a5 = amount.rolling(5, min_periods=3).mean()
        a60 = amount.rolling(60, min_periods=30).mean()
        F["f_turn"] = -(a5 / a60.replace(0, np.nan))

    # 3) 低波动：20 日收益波动，越低越好
    F["f_lowvol"] = -ret.rolling(20, min_periods=15).std()

    # 4) 中期动量：跳过最近 20 日的 120 日涨幅（避免与反转打架）
    F["f_mom120"] = adj.shift(20) / adj.shift(140) - 1.0

    # 5) 距离 60 日高点（趋势强度）
    F["f_high60"] = -(close / close.rolling(60, min_periods=40).max() - 1.0)

    # 截面标准化：全部转成 [-1,1] 的分位，天然可比、不受量纲影响
    return {k: _cs_rank_z(v) for k, v in F.items()}


# ---------------------------------------------------------------- IC 报告
def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    """无 scipy 的 Spearman：两边先取秩，再算 Pearson。"""
    return a.rank().corr(b.rank())


def forward_return(panel: dict[str, pd.DataFrame], horizon: int = 1) -> pd.DataFrame:
    adj = panel["adj_close"]
    return adj.shift(-horizon) / adj - 1.0


def ic_report(F: dict[str, pd.DataFrame], panel: dict[str, pd.DataFrame],
              horizon: int = 1, step: int = 5) -> pd.DataFrame:
    fr = forward_return(panel, horizon)
    rows = []
    idx = F[list(F)[0]].index
    sample = idx[::step]
    for name, f in F.items():
        ics = []
        for d in sample:
            if d not in fr.index:
                continue
            a = f.loc[d]
            b = fr.loc[d]
            m = a.notna() & b.notna()
            if m.sum() < 30:
                continue
            ics.append(_rank_corr(a[m], b[m]))
        s = pd.Series(ics).dropna()
        if not len(s):
            continue
        ir = s.mean() / s.std() if s.std() else np.nan
        rows.append({"factor": name, "IC_mean": s.mean(), "IC_std": s.std(),
                     "ICIR": ir, "IC>0": (s > 0).mean(), "n": len(s)})
    return pd.DataFrame(rows).sort_values("IC_mean", ascending=False)


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0, help="只用前 N 只票（调试）")
    a = ap.parse_args(argv)

    codes = None
    if a.sample:
        codes = D.universe()["code"].head(a.sample).tolist()

    t0 = time.time()
    panel = load_panel(codes)
    F = build_factors(panel)
    print(f"[factors] {len(F)} 个因子：{list(F)}", flush=True)
    last = F[list(F)[0]].index[-1]
    valid = int(F[list(F)[0]].loc[last].notna().sum())
    print(f"[factors] 最新交易日 {last}，有效标的 {valid} 只，用时 {time.time()-t0:.1f}s", flush=True)

    if a.build:
        rows = []
        for name, f in F.items():
            s = f.stack().rename(name)
            rows.append(s)
        long = pd.concat(rows, axis=1).dropna(how="all")
        long.index.names = ["date", "code"]
        out = os.path.join(D.C.OUT, "factor_panel.csv.gz")
        long.to_csv(out, compression="gzip")
        print(f"[factors] 写出 {out}（{len(long):,} 行）", flush=True)

    if a.ic:
        rep = ic_report(F, panel, horizon=a.horizon, step=a.step)
        print(f"[ic] 前视 {a.horizon} 日，抽样步长 {a.step}：", flush=True)
        print(rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
        rep.to_csv(os.path.join(D.C.OUT, f"ic_h{a.horizon}.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
