"""因子筛选：IC + Q5-Q1 买入持有价差 + t 值。"""
import sys, os, numpy as np, pandas as pd
import v5_data as D, v5_factors as F, v5_fund as FU

H = 40          # 持有期（交易日）
SAMPLE = 40     # 抽样步长
START, END = "2016-01-01", "2026-11-10"

def q_test(score_at, adj, dates):
    rows = []
    for i in range(60, len(dates) - H - 1, SAMPLE):
        d0, d1 = dates[i], dates[i + H]
        s = score_at(d0)
        if s is None or len(s.dropna()) < 100: continue
        s = s.dropna().sort_values()
        n = len(s) // 5
        qs = []
        for k in range(5):
            codes = list(s.index[k * n:(k + 1) * n]) if k < 4 else list(s.index[4 * n:])
            r = (adj.loc[d1, codes] / adj.loc[d0, codes] - 1).dropna()
            qs.append(float(r.mean()))
        rows.append([qs[0], qs[4], qs[4] - qs[0]])
    if len(rows) < 8: return None
    df = pd.DataFrame(rows, columns=["Q1", "Q5", "sp"])
    t = df["sp"].mean() / (df["sp"].std() / np.sqrt(len(df)))
    ann = df["sp"].mean() * 243 / H
    q1 = (1 + df["Q1"]).prod() ** (243 / (len(df) * H)) - 1
    q5 = (1 + df["Q5"]).prod() ** (243 / (len(df) * H)) - 1
    return dict(n=len(df), spread_ann=ann * 100, t=t, win=(df["sp"] > 0).mean() * 100,
                q1=q1 * 100, q5=q5 * 100)

def main():
    panel = F.load_panel(start=START, end=END)
    adj = panel["adj_close"]
    dates = list(adj.index)
    px = panel["close"]

    fac = F.build_factors(panel)          # 价量因子
    fund, _ = FU.factor_frame(dates, list(adj.columns))

    def mk(name):
        if name in fac:
            def f(d): return fac[name].loc[d]
            return f
        if name == "EP_v":      # 每股收益/股价
            return lambda d: (fund["EP"].loc[d] / px.loc[d]).replace([np.inf, -np.inf], np.nan)
        if name == "BP_v":
            return lambda d: (fund["BP"].loc[d] / px.loc[d]).replace([np.inf, -np.inf], np.nan)
        if name == "ROE":
            return lambda d: fund["ROE"].loc[d].replace([np.inf, -np.inf], np.nan)
        if name == "ROEstab":
            return lambda d: fund["ROEstab"].loc[d]
        if name == "VAL":       # EP+BP 等权
            return lambda d: (mk("EP_v")(d).rank(pct=True) + mk("BP_v")(d).rank(pct=True)) / 2
        if name == "QUAL":      # ROE + 稳定性
            return lambda d: (mk("ROE")(d).rank(pct=True) + mk("ROEstab")(d).rank(pct=True)) / 2
        if name == "VQ":        # 价值+质量
            return lambda d: (mk("VAL")(d).rank(pct=True) + mk("QUAL")(d).rank(pct=True)) / 2
        raise KeyError(name)

    names = ["f_lowvol", "f_rev20", "EP_v", "BP_v", "ROE", "ROEstab", "VAL", "QUAL", "VQ"]
    out = []
    for nm in names:
        f = mk(nm)
        r = q_test(f, adj, dates)
        if r is None:
            print(f"{nm:10s} 样本不足"); continue
        out.append(dict(factor=nm, **r))
        print(f"{nm:10s} Q5-Q1 年化 {r['spread_ann']:+6.2f}%  t={r['t']:+5.2f}  胜率 {r['win']:4.1f}%  Q1 {r['q1']:+6.2f}%  Q5 {r['q5']:+6.2f}%  n={r['n']}")
    pd.DataFrame(out).to_csv("out/screen.csv", index=False)
    print("[done] out/screen.csv")

if __name__ == "__main__":
    main()
