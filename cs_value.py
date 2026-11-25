#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_value.py —— A 股价值因子（PIT 口径），给 portfolio / core_satellite 做权重倾斜用。

为什么单独一个文件
------------------
BRAIN 上的那套价值 alpha 用的是美股字段；A 股这边字段、口径、披露节奏都不同，必须重写：
  1) 财报是「年初至今累计」→ 先拼 TTM，不能直接比季度值；
  2) 必须用「公告日 NOTICE_DATE」而不是「报告期 REPORT_DATE」→ 否则是未来函数；
  3) 市值要用「不复权收盘价 × 总股本」，不能用后复权价（复权价会把市值放大几十倍）。

数据来源（akshare）
------------------
  stock_profit_sheet_by_report_em(symbol="SH600036")
      → REPORT_DATE / NOTICE_DATE / OPERATE_INCOME（累计营业总收入）/ OPERATE_PROFIT
  stock_individual_info_em(symbol="600036")
      → 总股本（股）
价格用 portfolio.load_one(code, end, adjust="") 取不复权价。

输出
----
  value_panel.csv：index=交易日，columns=股票代码，值=当日横截面 EP 分位（0~1，越大越便宜）

用法
----
  python cs_value.py --codes 600036,000333 --start 2015-01-01 --end 2025-12-31
  python cs_value.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "fund_cache")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


# ============================================================================
# 数据层
# ============================================================================
def _clear_proxy_env() -> None:
    """清掉系统代理残留（本机 127.0.0.1:7892 常导致 akshare 直接 ProxyError）。"""
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"


def _ak_retry(fn, label: str, times: int = 2):
    _clear_proxy_env()
    last = None
    for i in range(times):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [warn] {label} 第 {i + 1} 次失败：{e}")
    raise RuntimeError(f"{label} 失败：{last}")


def _sina_symbol(code: str) -> str:
    c = str(code).zfill(6)
    return ("sh" if c[0] in "69" else "bj" if c[0] in "48" else "sz") + c


def _em_symbol(code: str) -> str:
    c = str(code).zfill(6)
    return ("SH" if c[0] in "69" else "BJ" if c[0] in "48" else "SZ") + c


def fetch_fin(code: str, refresh: bool = False) -> pd.DataFrame:
    """取单只票的利润表（含公告日），落缓存。"""
    import akshare as ak

    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"fin_{str(code).zfill(6)}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=["REPORT_DATE", "NOTICE_DATE"])

    df = _ak_retry(
        lambda: ak.stock_profit_sheet_by_report_em(symbol=_em_symbol(code)),
        f"利润表 {code}",
    )
    keep = [c for c in
            ("REPORT_DATE", "NOTICE_DATE", "REPORT_TYPE",
             "OPERATE_INCOME", "OPERATE_PROFIT", "OPERATE_INCOME_YOY")
            if c in df.columns]
    out = df[keep].copy()
    out["REPORT_DATE"] = pd.to_datetime(out["REPORT_DATE"], errors="coerce")
    out["NOTICE_DATE"] = pd.to_datetime(out["NOTICE_DATE"], errors="coerce")
    out = out.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE")
    # 没有公告日的记录，用「报告期 + 90 天」保守估计（宁可晚，不可早）
    out["NOTICE_DATE"] = out["NOTICE_DATE"].fillna(out["REPORT_DATE"] + pd.Timedelta(days=90))
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def fetch_shares(code: str, refresh: bool = False) -> float:
    """总股本（股）。取不到返回 0。"""
    import akshare as ak

    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"shares_{str(code).zfill(6)}.csv")
    if os.path.exists(path) and not refresh:
        v = pd.read_csv(path)["shares"].iloc[0]
        return float(v)

    try:
        info = _ak_retry(lambda: ak.stock_individual_info_em(symbol=str(code).zfill(6)), f"股本 {code}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 取不到 {code} 总股本：{e}")
        return 0.0

    shares = 0.0
    try:
        m = dict(zip(info["item"].astype(str), info["value"]))
        for k in ("总股本", "total_share", "总股本(股)"):
            if k in m:
                shares = float(m[k])
                break
    except Exception:  # noqa: BLE001
        pass
    pd.DataFrame({"shares": [shares]}).to_csv(path, index=False)
    return shares


# ============================================================================
# TTM：把「年初至今累计」拼成「滚动 12 个月」
# ============================================================================
def to_ttm(fin: pd.DataFrame, col: str = "OPERATE_INCOME") -> pd.DataFrame:
    """
    输入含 REPORT_DATE / col（累计值）。
    输出追加 ttm 列：TTM = 本期累计 + 上年全年 − 上年同期累计。
    """
    df = fin.copy()
    df["year"] = df["REPORT_DATE"].dt.year
    df["md"] = df["REPORT_DATE"].dt.strftime("%m-%d")
    cum = {(r.year, r.md): getattr(r, col) for r in df.itertuples()}

    def _ttm(row) -> float:
        cur = getattr(row, col)
        if cur is None or pd.isna(cur):
            return np.nan
        y, md = row.year, row.md
        if md == "12-31":
            return float(cur)                      # 年报本身就是完整 12 个月
        fy = cum.get((y - 1, "12-31"))
        prev = cum.get((y - 1, md))
        if fy is None or prev is None or pd.isna(fy) or pd.isna(prev):
            return np.nan
        return float(cur) + float(fy) - float(prev)

    df["ttm"] = df.apply(_ttm, axis=1)
    return df


# ============================================================================
# 面板：每个交易日 → 各票 EP 分位
# ============================================================================
def build_panel(codes: Iterable[str], start: str, end: str,
                refresh: bool = False, out: Optional[str] = None) -> pd.DataFrame:
    import portfolio as pf  # 复用它的取数（含缓存、代理重试）

    codes = [str(c).zfill(6) for c in codes]
    end_ts = pd.Timestamp(end)
    rows: Dict[str, pd.Series] = {}

    for code in codes:
        print(f"[value] {code} 取财报…")
        fin = fetch_fin(code, refresh=refresh)
        fin = to_ttm(fin, "OPERATE_INCOME")
        fin = fin.dropna(subset=["ttm"]).sort_values(["NOTICE_DATE", "REPORT_DATE"])
        fin = fin.drop_duplicates(subset=["NOTICE_DATE"], keep="last")  # 同日多期取最新一期
        if fin.empty:
            print(f"  [warn] {code} 没有可用 TTM")
            continue

        shares = fetch_shares(code, refresh=refresh)
        if shares <= 0:
            print(f"  [warn] {code} 股本为 0，跳过")
            continue

        lb = max(420, (end_ts - pd.Timestamp(start)).days + 420)
        px = pf.load_one(code, end_ts, adjust="", lookback_days=lb)  # 不复权真实价
        if px is None or px.empty:
            continue
        close = px["close"].astype(float)

        # 每个公告日生效一版 TTM：用 notice_date 做 as-of 合并
        fin_idx = fin.set_index("NOTICE_DATE")["ttm"]
        ttm_daily = fin_idx.reindex(close.index.union(fin_idx.index)).sort_index().ffill()
        ttm_daily = ttm_daily.reindex(close.index).ffill()
        ep = ttm_daily / shares / close                   # 营业收益率
        rows[code] = ep

    if not rows:
        raise RuntimeError("一只票都没算出来")

    panel = pd.DataFrame(rows).loc[str(start):str(end)]
    # 横截面分位（越大＝越便宜）
    pct = panel.rank(axis=1, pct=True)
    pct.index.name = "date"

    out = out or os.path.join(BASE, "value_panel.csv")
    pct.to_csv(out, encoding="utf-8-sig")
    print(f"[value] 写入 {out}：{pct.shape[0]} 天 × {pct.shape[1]} 只")
    return pct


# ============================================================================
# 自检
# ============================================================================
def selftest() -> None:
    ok = 0
    # 1) TTM 拼接
    fin = pd.DataFrame({
        "REPORT_DATE": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-12-31",
                                       "2024-03-31"]),
        "OPERATE_INCOME": [100.0, 220.0, 500.0, 130.0],
    })
    t = to_ttm(fin, "OPERATE_INCOME")["ttm"].tolist()
    # 2024Q1: 130 + 500 - 100 = 530
    assert abs(t[3] - 530.0) < 1e-6, t
    ok += 1
    print("✓ 1/3 TTM 拼接正确")

    # 2) 年报本身就是 12 个月
    assert abs(t[2] - 500.0) < 1e-6
    ok += 1
    print("✓ 2/3 年报不重复累加")

    # 3) 分位方向：EP 越大分位越高
    p = pd.DataFrame({"A": [0.10], "B": [0.05], "C": [0.20]}).rank(axis=1, pct=True)
    assert p.loc[0, "C"] > p.loc[0, "A"] > p.loc[0, "B"]
    ok += 1
    print("✓ 3/3 分位方向正确（越便宜分越高）")

    print(f"\n自检 {ok}/3 通过")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A 股价值因子（PIT + TTM）")
    ap.add_argument("--codes", default="", help="逗号分隔，默认用 portfolio.UNIVERSE")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        selftest()
        return 0

    if a.codes:
        codes = [c.strip() for c in a.codes.split(",") if c.strip()]
    else:
        import portfolio as pf
        codes = list(pf.DEFAULT_UNIVERSE)
    build_panel(codes, a.start, a.end, refresh=a.refresh, out=a.out or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
