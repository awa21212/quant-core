
# -*- coding: utf-8 -*-
"""微盘反转策略原型：3 个月形成期 + 3 个月持有 + 退市预警过滤 + ADV 上限 + 真实成本。
宇宙：610 只（500 存活 + 110 已退市），生存偏差已内置（死掉的票在死之前一直在池子里）。
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import v5_config as C

def load(kind="close"):
    live = pd.read_csv(os.path.join(C.OUT, "micro_codes.csv"), dtype={"code": str})["code"].tolist()
    dead = pd.read_csv(os.path.join(C.OUT, "delist_codes.csv"), dtype={"code": str})["code"].tolist()
    buf = {}
    for c in live + dead:
        p = os.path.join(C.CACHE, f"px_{c}_hfq.csv")
        if not os.path.exists(p): continue
        d = pd.read_csv(p)
        if len(d) < 200: continue
        d["date"] = pd.to_datetime(d["date"])
        s = d.set_index("date")
        col = s["close"].astype(float) if kind == "close" else s["amount"].astype(float)
        buf[c] = col
    df = pd.DataFrame(buf).sort_index()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--hold-n", type=int, default=30, help="持仓只数")
    ap.add_argument("--form", type=int, default=3, help="形成期月数")
    ap.add_argument("--hold", type=int, default=3, help="持有期月数")
    ap.add_argument("--min-adv", type=float, default=5e6, help="20 日平均成交额下限（元）")
    ap.add_argument("--adv-part", type=float, default=0.05, help="单票最多占 ADV 比例")
    ap.add_argument("--slip", type=float, default=0.0002, help="单边滑点，默认 2bp")
    ap.add_argument("--impact-coef", type=float, default=0.02, help="冲击成本系数（0=关）")
    ap.add_argument("--min-ratio", type=float, default=0.30, help="距高点比例下限（低于则视为濒死剔除）")
    ap.add_argument("--skip-pct", type=float, default=0.0, help="跳过最惨的 X% 不买（0=买极端档）")
    ap.add_argument("--no-filter", action="store_true", help="关闭退市预警过滤（对照组）")
    ap.add_argument("--no-cost", action="store_true")
    a = ap.parse_args()

    close = load("close"); amt = load("amount")
    m_close = close.resample("ME").last()
    m_amt = amt.resample("ME").mean()
    high = m_close.cummax()
    ratio = m_close / high
    print(f"[universe] {m_close.shape[1]} 只，{m_close.index[0].date()} ~ {m_close.index[-1].date()}")

    months = m_close.index
    eq = a.capital; equity = []
    pos = {}            # code -> shares
    cost_total = 0.0
    impact_total = 0.0
    trades = 0
    skipped_illiquid = 0
    cohort = []         # 当前持仓的建仓月份
    peak = eq

    start_i = a.form + 1
    for i in range(start_i, len(months) - 1):
        t = months[i]
        # 每月末按收盘价估值
        px = m_close.loc[t]
        val = sum(sh * px.get(c, 0) for c, sh in pos.items())
        eq = val + cash if "cash" in dir() else a.capital
        # 初始化 cash
        if i == start_i:
            cash = a.capital
            eq = a.capital

        equity.append((t, eq))

        # 到调仓月：清仓并按新信号重建
        if (i - start_i) % a.hold == 0:
            # 卖出
            for c, sh in list(pos.items()):
                p = px.get(c, np.nan)
                if not np.isfinite(p) or p <= 0:
                    pos.pop(c); continue
                p_sell = p * (1 - a.slip)
                gross = sh * p_sell
                av_s = float(m_amt.loc[t].get(c, np.nan))
                imp = 0.0
                if a.impact_coef > 0 and np.isfinite(av_s) and av_s > 0:
                    imp = gross * a.impact_coef * min(gross / av_s, 1.0)
                fee = 0 if a.no_cost else (gross * 0.00025 + gross * 0.0005 + gross * 0.00001)
                cash += gross - fee - imp; cost_total += fee + imp; trades += 1
                impact_total += imp
            pos.clear()

            form_ret = m_close.loc[t] / m_close.loc[months[i - a.form]] - 1
            adv = m_amt.loc[t]
            ok = form_ret.notna() & adv.notna() & (adv >= a.min_adv) & (px > 0)
            if not a.no_filter:
                ok &= (ratio.loc[t] >= a.min_ratio)      # 距高点过低视为濒死，剔除
            cand = form_ret[ok].sort_values()      # 跌得最狠的排在前面
            if a.skip_pct > 0:
                k0 = int(len(cand) * a.skip_pct)
                cand = cand.iloc[k0:]
            picks = cand.head(a.hold_n).index.tolist()
            skipped_illiquid += int((~ok).sum())

            if picks:
                w = 1.0 / len(picks)
                for c in picks:
                    p = px.get(c, np.nan)
                    if not np.isfinite(p) or p <= 0: continue
                    budget = eq * w
                    av = adv.get(c, np.nan)
                    if np.isfinite(av) and av > 0:
                        budget = min(budget, av * a.adv_part * a.hold)
                    if not np.isfinite(budget) or budget <= 0:
                        continue
                    p_buy = p * (1 + a.slip)
                    sh = int(budget / p_buy // 100) * 100
                    if np.isfinite(av) and av > 0 and a.impact_coef > 0:
                        sh_max = int(av * a.adv_part * a.hold / p_buy // 100) * 100
                        sh = min(sh, sh_max)
                    if sh <= 0: continue
                    gross = sh * p_buy
                    imp = 0.0
                    if a.impact_coef > 0 and np.isfinite(av) and av > 0:
                        imp = gross * a.impact_coef * min(gross / av, 1.0)
                    fee = 0 if a.no_cost else (gross * 0.00025 * 1.0 + gross * 0.00001)
                    fee = max(fee, 5.0) if not a.no_cost else 0.0
                    if gross + fee + imp > cash: continue
                    cash -= gross + fee + imp; cost_total += fee + imp; trades += 1
                    impact_total += imp
                    pos[c] = sh

    # 收尾
    t = months[-1]; px = m_close.loc[t]
    val = sum(sh * px.get(c, 0) for c, sh in pos.items())
    eq = val + cash
    equity.append((t, eq))
    ser = pd.Series(dict(equity)).sort_index()
    ret = ser.pct_change().dropna()
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / a.capital) ** (1 / yrs) - 1
    dd = (ser / ser.cummax() - 1).min()
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(12) if len(ret) > 12 else np.nan

    # 基准：全池等权月度再平衡
    bench = m_close.pct_change().mean(axis=1).loc[ser.index[0]:ser.index[-1]].fillna(0)
    bench_eq = a.capital * (1 + bench).cumprod()

    print(f"\n{'参数':<28}{'终值':>14}{'年化':>9}{'最大回撤':>10}{'夏普':>8}{'换手笔数':>9}{'总费用':>12}")
    print(f"{'策略' + ('(无过滤)' if a.no_filter else '(带退市过滤)') + ('/零成本' if a.no_cost else ''):<28}"
          f"{ser.iloc[-1]:>14,.0f}{cagr*100:>8.2f}%{dd*100:>9.2f}%{sharpe:>8.2f}{trades:>9}{cost_total:>12,.0f}")
    print(f"{'全池等权基准':<28}{bench_eq.iloc[-1]:>14,.0f}"
          f"{(bench_eq.iloc[-1]/a.capital)**(1/yrs)*100-100:>8.2f}%"
          f"{(bench_eq/bench_eq.cummax()-1).min()*100:>9.2f}%"
          f"{(bench.mean()/bench.std(ddof=1)*np.sqrt(12)):>8.2f}{'-':>9}{'-':>12}")
    print(f"\n费用占本金 {cost_total/a.capital*100:.2f}%｜累计 {cost_total:,.0f} 元｜流动性不达标被剔 {skipped_illiquid} 次(月度累计)")

if __name__ == "__main__":
    main()
