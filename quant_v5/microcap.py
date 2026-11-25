# -*- coding: utf-8 -*-
"""微盘股样本（市值 15–60 亿，取最小的 500 只）上的 J×K 反转/惯性检验。
目的：检验「反转只在微盘存在」这一假设——若成立，则 CSI800 上的消失是宇宙问题而非时代问题。
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import akshare as ak
import v5_data as D, v5_config as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = os.path.join(C.OUT, "micro_codes.csv")

def pick():
    if os.path.exists(OUT):
        df = pd.read_csv(OUT, dtype={"code": str})
        print(f"[pick] 复用 {len(df)} 只")
        return df["code"].tolist()
    spot = ak.stock_zh_a_spot_em()
    cap = [c for c in spot.columns if "总市值" in c][0]
    spot = spot[~spot["名称"].str.contains("ST|退", na=False)]
    spot = spot[~spot["代码"].astype(str).str.startswith(("4", "8", "9"))]
    spot = spot[(spot[cap] > 1.5e9) & (spot[cap] < 6e9)]
    spot = spot.sort_values(cap).head(N)
    spot[["代码", "名称"]].rename(columns={"代码": "code", "名称": "name"}).to_csv(OUT, index=False, encoding="utf-8")
    print(f"[pick] 选出 {len(spot)} 只，总市值区间 {spot[cap].min()/1e8:.1f}–{spot[cap].max()/1e8:.1f} 亿")
    return spot["代码"].astype(str).tolist()


def _sina_sym(code):
    return ("sh" if code[0] in "69" else "sz") + code

def get_px(code, adj):
    path = D._px_path(code, adj)
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return True
    try:
        D.fetch_px(code, adjust=adj)
        if os.path.exists(path) and os.path.getsize(path) > 200:
            return True
    except Exception:
        pass
    try:
        df = ak.stock_zh_a_daily(symbol=_sina_sym(code), start_date="20150101",
                                 end_date="20261110", adjust=("hfq" if adj == "hfq" else ""))
        if df is None or len(df) < 200: return False
        df = df.rename(columns={"date": "date"})
        keep = [c for c in ["date","open","high","low","close","volume","amount","turnover"] if c in df.columns]
        if "amount" not in df.columns: df["amount"] = df["close"] * df["volume"]
        if "turnover" not in df.columns: df["turnover"] = np.nan
        df[keep].to_csv(path, index=False)
        return True
    except Exception:
        return False

codes = pick()
t0 = time.time(); bad = 0
for i, code in enumerate(codes, 1):
    try:
        if not get_px(code, "hfq"): bad += 1
        time.sleep(0.6)
    except Exception as e:
        bad += 1
    if i % 25 == 0:
        print(f"[dl] {i}/{len(codes)} 用时 {time.time()-t0:.0f}s 失败 {bad}", flush=True)
print(f"[dl] 完成 {len(codes)} 只，失败 {bad}，用时 {time.time()-t0:.0f}s")

buf = {}
for code in codes:
    p = D._px_path(code, "hfq")
    if not os.path.exists(p): continue
    df = pd.read_csv(p)
    if len(df) < 200: continue
    buf[code] = df.set_index("date")["close"].astype(float)
adj = pd.DataFrame(buf); adj.index = pd.to_datetime(adj.index)
m = adj.resample("ME").last()
print(f"[panel] 微盘 {adj.shape[1]} 只 × {len(m)} 个月")

def run(J, K):
    ret = m.pct_change(); form = m.shift(1)/m.shift(1+J) - 1
    w = []
    for i in range(J+1, len(m)-K):
        t = m.index[i]; s = form.loc[t].dropna()
        if len(s) < 100: continue
        r = s.rank(pct=True)
        hi = s.index[r > 0.9]; lo = s.index[r <= 0.1]
        for k in range(1, K+1):
            if i+k >= len(m): break
            a = ret.loc[m.index[i+k], hi].mean(); b = ret.loc[m.index[i+k], lo].mean()
            if np.isfinite(a) and np.isfinite(b): w.append(a-b)
    w = np.array(w)
    if len(w) < 20: return None
    wb = w[::K]
    t = w.mean()/(w.std(ddof=1)/np.sqrt(len(w)))
    tn = wb.mean()/(wb.std(ddof=1)/np.sqrt(len(wb))) if len(wb) > 5 else np.nan
    return (J, K, w.mean()*100, t, tn, len(w))

print("\n列：J 形成期、K 持有期、多空 %/月、t（重叠）、t（非重叠）")
for J, K in [(1,1),(1,3),(3,3),(6,6),(12,12)]:
    r = run(J, K)
    if r: print(f"J={r[0]:>2} K={r[1]:>2}  多空 {r[2]:+.3f}%  t {r[3]:+.2f}  t_nb {r[4]:+.2f}  n={r[5]}")
