# -*- coding: utf-8 -*-
"""v5_data.py —— V5 数据层（P1）。

目标：任何一天 T，都能拿到「T 当时真正可见的数据」，且**不含未来信息**。

三张表
------
1. 行情：每只票一份 CSV（不复权 + 后复权两份），缓存到 cache/
2. 财务：每只票一份 CSV，字段含 report_date（报告期）与 notice_date（公告日）
   —— 做 PIT 的关键：只看 notice_date <= T 的记录
3. 行业：code → 行业（申万/东财一级），缓存到 meta/

用法
----
    python v5_data.py --universe           # 刷新股票池
    python v5_data.py --px 000001 600036   # 下载指定票的行情
    python v5_data.py --fin 000001         # 下载财务
    python v5_data.py --check 2023-06-30   # PIT 校验（挑 10 只票）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v5_config as C  # noqa: E402

# 清掉系统里可能残留的死代理（V4 踩过：127.0.0.1:7892 导致 ProxyError）
for _k in [k for k in os.environ if "PROXY" in k.upper() and k.upper() != "NO_PROXY"]:
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
import urllib.request  # noqa: E402
urllib.request.getproxies = lambda: {}


def log(msg: str) -> None:
    print(msg, flush=True)


def _retry(fn, label: str, n: int = C.RETRIES):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(C.RETRY_WAIT * (i + 1))
    log(f"  [warn] {label} 失败：{type(last).__name__}: {str(last)[:70]}")
    return None


# ---------------------------------------------------------------- 股票池
def universe(refresh: bool = False) -> pd.DataFrame:
    """中证 800 成分（沪深300 + 中证500）。"""
    path = os.path.join(C.META, "universe.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, dtype={"code": str})

    import akshare as ak
    frames = []
    for idx_code, idx_name in C.INDEX_CODES.items():
        df = _retry(lambda: ak.index_stock_cons(symbol=idx_code), f"index {idx_code}")
        if df is None or not len(df):
            continue
        cols = list(df.columns)
        c_code = next((c for c in cols if "代码" in c), None)
        c_name = next((c for c in cols if "名称" in c), None)
        c_date = next((c for c in cols if "日期" in c or "纳入" in c), None)
        part = pd.DataFrame({
            "code": df[c_code].astype(str).str.zfill(6),
            "name": df[c_name].astype(str) if c_name else "",
            "board": idx_name,
        })
        if c_date:
            part["in_date"] = df[c_date].astype(str)
        frames.append(part)

    if not frames:
        raise RuntimeError("股票池拉取失败（akshare 接口不可用）")

    uni = pd.concat(frames, ignore_index=True).drop_duplicates("code")
    uni.to_csv(path, index=False, encoding="utf-8")
    log(f"[universe] {len(uni)} 只（{uni['board'].value_counts().to_dict()}）")
    return uni


# ---------------------------------------------------------------- 行情
def _px_path(code: str, adjust: str) -> str:
    tag = "hfq" if adjust == "hfq" else "raw"
    return os.path.join(C.CACHE, f"px_{code}_{tag}.csv")


def fetch_px(code: str, adjust: str = "", start: str = C.PX_START,
             end: str | None = None, refresh: bool = False) -> pd.DataFrame | None:
    """单只票日线。adjust="" 取不复权，adjust="hfq" 取后复权。"""
    path = _px_path(code, adjust)
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, dtype={"date": str})
        if len(df):
            return df

    import akshare as ak
    end = end or datetime.now().strftime("%Y%m%d")
    s = start.replace("-", "")
    df = _retry(lambda: ak.stock_zh_a_hist(symbol=code, period="daily",
                                           start_date=s, end_date=end, adjust=adjust),
                f"px {code} {adjust or 'raw'}")
    if df is None or not len(df):
        return None
    ren = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
           "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount", "turnover")
            if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").drop_duplicates("date")
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def load_px(code: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    """返回对齐后的行情：以不复权价为锚，用后复权算收益（避免除权跳空）。

    列：date/open/high/low/close（真实价，元）/volume/amount/turnover/adj_close/ret
    """
    raw = fetch_px(code, "")
    hfq = fetch_px(code, "hfq")
    if raw is None or hfq is None:
        return None
    r = raw.copy()
    h = hfq[["date", "close"]].rename(columns={"close": "adj_close"})
    df = r.merge(h, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    # 后复权与真实价的锚点：首日真实价 / 首日复权价（V4 学到的：后复权价格会膨胀到几千块，
    # 直接用会把整手约束算错）
    if len(df) and df["adj_close"].iloc[0] > 0:
        anchor = df["close"].iloc[0] / df["adj_close"].iloc[0]
        df["adj_close"] = df["adj_close"] * anchor
    df["ret"] = df["adj_close"].pct_change()
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 财务（PIT）
def _fin_path(code: str) -> str:
    return os.path.join(C.CACHE, f"fin_{code}.csv")


def fetch_fin(code: str, refresh: bool = False) -> pd.DataFrame | None:
    """利润表关键科目，带 report_date 与 notice_date（公告日）。

    PIT 规则：T 日只能用 notice_date <= T 的记录。
    """
    path = _fin_path(code)
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, dtype={"report_date": str, "notice_date": str})
        return df if len(df) else None

    import akshare as ak
    sym = ("SH" if code[0] in "69" else "SZ") + code
    df = _retry(lambda: ak.stock_profit_sheet_by_report_em(symbol=sym), f"fin {code}")
    if df is None or not len(df):
        return None
    cols = list(df.columns)
    c_rep = next((c for c in cols if c.upper() == "REPORT_DATE"), None)
    c_notice = next((c for c in cols if c.upper() == "NOTICE_DATE"), None)
    c_oi = next((c for c in cols if c.upper() == "OPERATE_INCOME"), None)
    c_op = next((c for c in cols if c.upper() == "OPERATE_PROFIT"), None)
    c_np = next((c for c in cols if c.upper() in ("NETPROFIT", "PARENT_NETPROFIT")), None)
    if not (c_rep and c_notice):
        log(f"  [warn] {code} 财务表缺少 REPORT_DATE/NOTICE_DATE")
        return None
    out = pd.DataFrame({
        "report_date": pd.to_datetime(df[c_rep], errors="coerce").dt.strftime("%Y-%m-%d"),
        "notice_date": pd.to_datetime(df[c_notice], errors="coerce").dt.strftime("%Y-%m-%d"),
    })
    if c_oi:
        out["operate_income"] = pd.to_numeric(df[c_oi], errors="coerce")
    if c_op:
        out["operate_profit"] = pd.to_numeric(df[c_op], errors="coerce")
    if c_np:
        out["net_profit"] = pd.to_numeric(df[c_np], errors="coerce")
    out = out.dropna(subset=["report_date", "notice_date"]).sort_values("notice_date")
    out = out.drop_duplicates("report_date", keep="last")
    out.to_csv(path, index=False, encoding="utf-8")
    return out


def fin_visible(code: str, as_of: str) -> pd.DataFrame | None:
    """T 日可见的财务记录（notice_date <= T）。空表返回 None。"""
    df = fetch_fin(code)
    if df is None or not len(df):
        return None
    vis = df[df["notice_date"] <= as_of]
    return vis if len(vis) else None


# ---------------------------------------------------------------- PIT 校验
def pit_check(as_of: str, codes: list[str] | None = None) -> bool:
    """验证：T 日可见的数据里，没有任何一条来自 T 之后。"""
    uni = universe()
    codes = codes or uni["code"].head(10).tolist()
    ok = True
    log(f"[pit] 校验日 {as_of}，样本 {len(codes)} 只")
    for code in codes:
        df = load_px(code, end=as_of)
        if df is None or not len(df):
            log(f"  [warn] {code} 无行情")
            ok = False
            continue
        bad_px = df["date"] > as_of
        if bad_px.any():
            log(f"  [FAIL] {code} 行情里混入 {as_of} 之后的 {bad_px.sum()} 条")
            ok = False
        fin = fin_visible(code, as_of)
        raw_fin = fetch_fin(code)
        if raw_fin is not None and len(raw_fin):
            leaked = raw_fin[raw_fin["notice_date"] <= as_of].shape[0]
            total = raw_fin.shape[0]
            if fin is not None and len(fin) != leaked:
                log(f"  [FAIL] {code} PIT 过滤条数不一致")
                ok = False
            log(f"  {code}: 行情 {len(df)} 行（末 {df['date'].iloc[-1]}）｜"
                f"财务可见 {leaked}/{total} 期（最后一期公告 {fin['notice_date'].iloc[-1] if fin is not None else '—'}）")
        else:
            log(f"  {code}: 行情 {len(df)} 行｜财务缺失")
    log(f"[pit] 结论：{'通过' if ok else '有问题'}")
    return ok


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V5 数据层")
    ap.add_argument("--universe", action="store_true", help="刷新股票池")
    ap.add_argument("--px", nargs="*", help="下载指定票行情")
    ap.add_argument("--fin", nargs="*", help="下载指定票财务")
    ap.add_argument("--check", metavar="DATE", help="PIT 校验")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    if a.universe:
        uni = universe(refresh=True)
        log(uni.head(5).to_string(index=False))

    codes = a.px or []
    if a.limit:
        codes = codes[:a.limit]
    for c in codes:
        df = load_px(c)
        if df is not None:
            log(f"[px] {c} {len(df)} 行，{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    for c in (a.fin or []):
        df = fetch_fin(c)
        if df is not None:
            log(f"[fin] {c} {len(df)} 期，{df['notice_date'].iloc[0]} ~ {df['notice_date'].iloc[-1]}")

    if a.check:
        return 0 if pit_check(a.check) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
