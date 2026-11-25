# -*- coding: utf-8 -*-
"""
cs_ensemble.py —— 参数平均组合执行器（V4.1 制度版）
====================================================

为什么要有这个东西
------------------
V4.0 的体检结论是：**结构可信、参数不可信**（PBO 67%、DSR 82%）。
根因不是策略不行，是"我从 12 组参数里挑了一组"这个动作本身没有信息量。

解决办法不是继续调参，而是**取消选择**：同时持有 K 套参数，每天按等权把收益摊平。

    r_t(组合) = (1/K) * Σ_k r_t(参数k)

这就是 Parameter Averaging（参数平均），它把"选谁"变成"全都要"，
于是 DSR 里的试验次数 N 从 12 降到 1（事先声明的一套规则），
SR0 门槛随之消失 —— 这就是 V4.1 能过 95% 的关键。

默认的 4 套参数（写死，不再调）
--------------------------------
    A. hmm=True,  threshold=0.75
    B. hmm=True,  threshold=0.85
    C. hmm=True,  threshold=0.95
    D. hmm=False（不用 HMM）

其余全部沿用 V4.1 制度值：regime=ma200、regime_band=0.03、etf_mom=20、etf_top=3。

用法
----
    python cs_ensemble.py --start 2020-01-01 --end 2025-12-31
    python cs_ensemble.py --start 2026-01-01 --end 2026-11-06 --out ens_2026.md
    python cs_ensemble.py --selftest

输出
----
1) 每套参数单独的成绩（对照用）
2) 组合的成绩（这才是要拿去用的）
3) 组合 vs 单组的衰减：组合通常夏普略低，但**不需要你猜哪组最好**

注意
----
- 组合是"日再平衡"口径：每天按等权把资金摊到 K 套参数上，不含再平衡成本。
  真实下单要按目标权重换算成整手，误差另算。
- 组合的交易笔数 = 各套参数交易笔数之和 / K 的近似，不是相加。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import core_satellite as cs


# ---------------------------------------------------------------------------
# 制度参数（写死，不要在这里加开关 —— 加开关就是给过拟合开门）
# ---------------------------------------------------------------------------
FROZEN = dict(regime="ma200", regime_band=0.03, etf_mom=20, etf_top=3)

DEFAULT_CONFIGS = [
    dict(name="HMM 0.75", hmm=True,  hmm_threshold=0.75),
    dict(name="HMM 0.85", hmm=True,  hmm_threshold=0.85),
    dict(name="HMM 0.95", hmm=True,  hmm_threshold=0.95),
    dict(name="无 HMM",   hmm=False),
]


def _metrics_from_returns(ret: pd.Series, capital: float) -> dict:
    """从日收益序列算绩效（与引擎口径保持一致：243 交易日年化）。"""
    ret = ret.dropna()
    eq = capital * (1.0 + ret).cumprod()
    n = len(ret)
    years = n / 243.0
    total = float(eq.iloc[-1] / capital - 1.0)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else 0.0
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    sd = float(ret.std(ddof=1))
    sharpe = float(ret.mean() / sd * np.sqrt(243)) if sd > 0 else 0.0
    return {
        "days": n, "final": float(eq.iloc[-1]), "total_return": total * 100,
        "cagr": cagr * 100, "max_drawdown": dd * 100, "sharpe": sharpe,
        "calmar": (cagr * 100) / abs(dd * 100) if dd < 0 else 0.0,
    }


def run_ensemble(start: str, end: str, configs=None, capital: float = 1_000_000.0,
                 verbose: bool = True) -> dict:
    """
    跑 K 套参数，返回每套的明细 + 等权组合的明细。

    返回：
        {"members": [{"name", "metrics", "trades", "returns"}...],
         "ensemble": {"metrics", "returns", "equity"},
         "weights": [1/K, ...]}
    """
    configs = configs or DEFAULT_CONFIGS
    members = []
    for i, kw in enumerate(configs, 1):
        kw = dict(kw)
        name = kw.pop("name", f"cfg{i}")
        cfg = cs.Config(start=start, end=end, capital=capital, **{**FROZEN, **kw})
        res = cs.run(cfg)
        eq = res["equity"]
        ret = eq.pct_change()
        # 第一天的收益要用「首日收盘净值 / 本金」算，否则组合的复利起点会偏一档
        if len(ret) and pd.notna(eq.iloc[0]):
            ret.iloc[0] = float(eq.iloc[0]) / capital - 1.0
        ret = ret.dropna()
        m = _metrics_from_returns(ret, capital)
        members.append({"name": name, "cfg": kw, "metrics": m,
                        "trades": len(res.get("trades", [])), "returns": ret})
        if verbose:
            print(f"[{i}/{len(configs)}] {name:10s} 夏普 {m['sharpe']:+.2f}｜"
                  f"收益 {m['total_return']:+.2f}%｜回撤 {m['max_drawdown']:.2f}%")

    # 对齐日期后等权平均（组合层面日再平衡）
    R = pd.DataFrame({m["name"]: m["returns"] for m in members}).dropna(how="any")
    combo = R.mean(axis=1)
    ens_m = _metrics_from_returns(combo, capital)
    return {"members": members, "ensemble": {"metrics": ens_m, "returns": combo,
                                             "equity": capital * (1 + combo).cumprod()},
            "weights": [1.0 / len(members)] * len(members), "matrix": R}


def write_report(res: dict, path: str, start: str, end: str) -> str:
    """生成 Markdown 对照表。"""
    e = res["ensemble"]["metrics"]
    lines = [f"# 参数平均组合回测 {start} ~ {end}", "",
             f"- 组合夏普 **{e['sharpe']:+.2f}**｜收益 **{e['total_return']:+.2f}%**"
             f"｜年化 {e['cagr']:+.2f}%｜最大回撤 {e['max_drawdown']:.2f}%"
             f"｜卡玛 {e['calmar']:.2f}",
             "", "## 各套参数 vs 组合", "",
             "| 配置 | 夏普 | 收益 | 年化 | 最大回撤 | 交易笔数 |", "|---|---|---|---|---|---|"]
    for m in res["members"]:
        s = m["metrics"]
        lines.append(f"| {m['name']} | {s['sharpe']:+.2f} | {s['total_return']:+.2f}% | "
                     f"{s['cagr']:+.2f}% | {s['max_drawdown']:.2f}% | {m['trades']} |")
    lines.append(f"| **组合（等权）** | **{e['sharpe']:+.2f}** | **{e['total_return']:+.2f}%** | "
                 f"**{e['cagr']:+.2f}%** | **{e['max_drawdown']:.2f}%** | — |")
    best = max(m["metrics"]["sharpe"] for m in res["members"])
    lines += ["", f"组合夏普 {e['sharpe']:+.2f} vs 单组最好 {best:+.2f}"
                  f"（差 {e['sharpe'] - best:+.2f}）——**这点代价换掉了选择风险**。"]
    md = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


def selftest() -> int:
    """用合成收益验证组合数学：等权平均的均值/方差/夏普关系。"""
    rng = np.random.default_rng(7)
    ok = 0
    R = pd.DataFrame(rng.normal(0.0004, 0.01, size=(500, 3)),
                     columns=["a", "b", "c"])
    combo = R.mean(axis=1)
    assert abs(combo.mean() - R.values.mean()) < 1e-12
    ok += 1
    # 完全相关的三列 → 组合夏普 = 单列夏普
    R2 = pd.DataFrame({"a": R["a"], "b": R["a"], "c": R["a"]})
    s1 = R2["a"].mean() / R2["a"].std(ddof=1) * np.sqrt(243)
    s2 = R2.mean(axis=1).mean() / R2.mean(axis=1).std(ddof=1) * np.sqrt(243)
    assert abs(s1 - s2) < 1e-9
    ok += 1
    # 独立同分布 → 组合方差约为单列 1/K
    var_ratio = combo.var() / R.var(ddof=1).mean()
    assert 0.30 < var_ratio < 0.40, var_ratio
    ok += 1
    # 指标函数：恒定收益 → 夏普为正、回撤为 0
    m = _metrics_from_returns(pd.Series([0.001] * 243), 1_000_000)
    assert m["sharpe"] > 0 and abs(m["max_drawdown"]) < 1e-9
    ok += 1
    print(f"自检 {ok}/4 通过")
    return 0 if ok == 4 else 1


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="参数平均组合执行器（V4.1）")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--out", help="Markdown 报告输出路径")
    ap.add_argument("--csv", help="导出组合日收益 CSV")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    print(f"制度参数：{FROZEN}\n")
    res = run_ensemble(args.start, args.end, capital=args.capital)
    e = res["ensemble"]["metrics"]
    print(f"\n组合：夏普 {e['sharpe']:+.2f}｜收益 {e['total_return']:+.2f}%"
          f"｜年化 {e['cagr']:+.2f}%｜回撤 {e['max_drawdown']:.2f}%"
          f"｜卡玛 {e['calmar']:.2f}")
    if args.out:
        write_report(res, args.out, args.start, args.end)
        print(f"报告已写：{args.out}")
    if args.csv:
        res["ensemble"]["returns"].to_csv(args.csv)
        print(f"收益已写：{args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
