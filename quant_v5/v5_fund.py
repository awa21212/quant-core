"""基本面因子：EP / BP / ROE / ROE 稳定性 + 组合，带 PIT。"""
import os, glob
import numpy as np, pandas as pd
import v5_config as C, v5_data as D

ABS = os.path.join(C.CACHE, "abs")
CN = {"归母净利润": "np", "每股净资产": "bps", "基本每股收益": "eps",
      "净资产收益率(ROE)": "roe", "股东权益合计(净资产)": "equity"}

def _ttm(df, col):
    """把年内累计值转 TTM：Q4 直接用；否则 本年累计 + 上年年报 - 上年同期累计。"""
    d = df.copy()
    d["y"] = d["report_date"].dt.year
    d["m"] = d["report_date"].dt.month
    annual = d[d.m == 12].set_index("y")[col]
    out = []
    for _, r in d.iterrows():
        if r.m == 12:
            out.append(r[col])
        else:
            prev_annual = annual.get(r.y - 1, np.nan)
            prev_same = d[(d.y == r.y - 1) & (d.m == r.m)][col]
            prev_same = float(prev_same.iloc[0]) if len(prev_same) else np.nan
            out.append(r[col] + prev_annual - prev_same)
    return pd.Series(out, index=d.index)

def load_one(code):
    pa = os.path.join(ABS, f"{code}.csv")
    pf = D._fin_path(code)
    if not (os.path.exists(pa) and os.path.exists(pf)): return None
    a = pd.read_csv(pa).rename(columns=CN)
    f = pd.read_csv(pf, dtype={"report_date": str, "notice_date": str})
    if "np" not in a.columns or "notice_date" not in f.columns: return None
    a["report_date"] = pd.to_datetime(a["report_date"])
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    m = a.merge(f[["report_date", "notice_date"]], on="report_date", how="left")
    m = m.dropna(subset=["notice_date"]).sort_values("report_date").reset_index(drop=True)
    if len(m) < 6: return None
    m["notice_date"] = pd.to_datetime(m["notice_date"])
    m["np_ttm"] = _ttm(m, "np")
    m["eps_ttm"] = _ttm(m, "eps")
    ann = m[m.report_date.dt.month == 12].copy()
    ann = ann.sort_values("report_date").tail(3)
    m["roe_stab"] = -float(ann["roe"].std()) if len(ann) >= 2 else np.nan
    m["roe_ttm"] = m["np_ttm"] / m["equity"].replace(0, np.nan)
    return m[["notice_date", "np_ttm", "eps_ttm", "bps", "roe_ttm", "roe_stab"]]

def factor_frame(dates, codes):
    """返回 dict[name] = DataFrame(index=dates, columns=codes)。"""
    rows = {}
    for code in codes:
        m = load_one(code)
        if m is not None: rows[code] = m
    out = {k: pd.DataFrame(index=dates, columns=list(rows), dtype=float)
           for k in ["EP", "BP", "ROE", "ROEstab"]}
    for code, m in rows.items():
        nd = m["notice_date"].values
        for d in dates:
            pos = np.searchsorted(nd, np.datetime64(d), side="right") - 1
            if pos < 0: continue
            r = m.iloc[pos]
            out["EP"].at[d, code] = r["eps_ttm"]
            out["BP"].at[d, code] = r["bps"]
            out["ROE"].at[d, code] = r["roe_ttm"]
            out["ROEstab"].at[d, code] = r["roe_stab"]
    return out, rows
