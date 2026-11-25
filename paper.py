"""模拟盘 —— 用最新可得行情跑一次当前引擎（V4.6），输出「今天收盘后账户长什么样」。

它不预测明天，只做一件事：把回测引擎跑到最新交易日，然后把**强平之前**的账户状态
（现金 / 核心指数 / 长期股 / 卫星择时腿 / 每笔持仓的成本与盈亏）打印成人能看的表，
并存一份 JSON 快照，方便逐日对比。

用法：
    python paper.py                       # 从 2026-01-01 开始跑，止于最新交易日
    python paper.py --start 2025-01-01    # 换起跑日（会影响成交路径，不影响规则）
    python paper.py --capital 500000
    python paper.py --end 2026-11-06      # 手动指定截止日（默认自动取指数最后一根）
    python paper.py --no-hmm              # 关掉 HMM 闸门（对照用）
    python paper.py --no-chart            # 不画图
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import core_satellite as cs
import cs_data
import cs_hmm
import cs_regime
from cs_data import NAME_MAP

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")

_EXTRA_NAMES = {"511010": "国债ETF", "518880": "黄金ETF", "510880": "红利ETF", "510300": "沪深300ETF"}


def name_of(code: str) -> str:
    return NAME_MAP.get(code) or _EXTRA_NAMES.get(code) or code


# --------------------------------------------------------------------------- #
# 1) 行情与状态
# --------------------------------------------------------------------------- #
def latest_trade_date() -> pd.Timestamp:
    """指数最后一根 K 线的日期 = 最新交易日。"""
    return cs_data.load_index("2025-01-01", "2026-12-31").index[-1]


def market_state(end: pd.Timestamp) -> dict:
    """裁判 + HMM：今天该不该进攻的两个证据。"""
    close = cs_data.load_index("1990-01-01", str(end.date()))
    bench = cs_regime.bench_frame(close, window=200, slope_win=0)
    states = cs_regime.build_states(bench, confirm=2, band=0.03, slope_win=0, slope_min=0.0)
    prob = cs_hmm.hmm_bull_prob(close)
    p = float(prob.iloc[-1])
    guard = bool(close.iloc[-1] > bench["ma"].iloc[-1])
    bull = bool(states.iloc[-1])
    return {
        "date": str(end.date()),
        "index_close": round(float(close.iloc[-1]), 2),
        "ma200": round(float(bench["ma"].iloc[-1]), 2),
        "above_ma200": guard,
        "regime_bull": bull,
        "regime_text": "进攻期" if bull else "防守期",
        "hmm_bull_prob": round(p, 4),
        "hmm_fire": bool(p >= 0.85 and guard),
    }


def _is_etf(code: str) -> bool:
    """5 打头 = 沪市 ETF，15 打头 = 深市 ETF；0/3/6/8/9 打头是个股。"""
    return code.startswith(("5", "15"))


def price_map(codes, end) -> dict[str, float]:
    """取每个标的在截止日的收盘价（ETF 走 ETF 通道，个股走个股通道）。"""
    out: dict[str, float] = {}
    stocks = [c for c in dict.fromkeys(codes) if c != cs.CORE_INDEX and not _is_etf(c)]
    stock_df = {}
    if stocks:
        try:
            stock_df = cs_data.load_stocks(stocks, str(end.date()), start="2025-01-01")
        except Exception:
            stock_df = {}
    for c in dict.fromkeys(codes):
        try:
            if c == cs.CORE_INDEX:
                s = cs_data.load_index("2025-01-01", str(end))
                out[c] = float(s.iloc[-1])
            elif _is_etf(c):
                d = cs_data._safe_load_etf(c, "2025-01-01", str(end), 20)
                if d is not None and len(d):
                    out[c] = float(d["close"].iloc[-1])
            else:
                d = stock_df.get(c)
                if d is not None and len(d):
                    out[c] = float(d["close"].iloc[-1])
        except Exception:
            continue
    return out


def real_prices(codes, end) -> dict[str, float]:
    """真实（不复权）价格：ETF 本来就不复权，个股要单独取 adjust="" 的口径。"""
    import portfolio as pf
    out: dict[str, float] = {}
    for c in dict.fromkeys(codes):
        try:
            if c == cs.CORE_INDEX:
                continue
            if _is_etf(c):
                d = cs_data._safe_load_etf(c, "2025-01-01", str(end), 20)
                if d is not None and len(d):
                    out[c] = float(d["close"].iloc[-1])
            else:
                d = pf.load_one(c, pd.Timestamp(end), adjust="")
                if d is not None and len(d):
                    out[c] = float(d["close"].iloc[-1])
        except Exception:
            continue
    return out


def order_sheet(rows, prices_real, eq, proxy="510300"):
    """把「回测口径的权重」翻译成「真实价格下能下单的股数」。

    回测用后复权价，股数在真实世界里对不上（招行后复权 229 元 vs 实际 42 元）；
    但**权重**是尺度无关的，所以这里按权重 × 真实资金 ÷ 真实价格 反推整手股数。
    """
    out = []
    for r in rows:
        code = r["代码"]
        px = prices_real.get(proxy) if r["分组"] == "核心" else prices_real.get(code)
        if not px:
            continue
        target_val = r["权重"] / 100 * eq
        sh = int(target_val // (px * 100)) * 100
        if sh <= 0:
            continue
        out.append({"分组": r["分组"], "代码": proxy if r["分组"] == "核心" else code,
                    "名称": (_EXTRA_NAMES["510300"] if r["分组"] == "核心" else r["名称"]),
                    "股数": sh, "价格": round(px, 3), "金额": round(sh * px, 2),
                    "权重": round(sh * px / eq * 100, 2)})
    return out


# --------------------------------------------------------------------------- #
# 2) 账户 → 人看的表
# --------------------------------------------------------------------------- #
def holdings_table(st: dict, prices: dict[str, float], eq: float) -> list[dict]:
    rows: list[dict] = []

    core_sh = float(st.get("core_shares") or 0.0)
    px = prices.get(cs.CORE_INDEX)
    if core_sh > 1e-9 and px:
        rows.append({"分组": "核心", "代码": "000300", "名称": "沪深300 指数",
                     "股数": round(core_sh, 2), "成本": None, "现价": round(px, 2),
                     "市值": round(core_sh * px, 2), "盈亏率": None})

    for group, key, cost_key, tag in (("长期", "longterm", "lt_cost", "·长期"),
                                      ("择时", "tactical", "tactical_cost", "·择时")):
        sh_map = st.get(key) or {}
        cost_map = st.get(cost_key) or {}
        for code, sh in sh_map.items():
            px = prices.get(code)
            if not px or not sh or int(sh) <= 0:
                continue
            cost = cost_map.get(code)
            rows.append({"分组": group, "代码": code, "名称": name_of(code) + tag,
                         "股数": int(sh), "成本": round(cost, 3) if cost else None,
                         "现价": round(px, 3), "市值": round(sh * px, 2),
                         "盈亏率": round((px / cost - 1) * 100, 2) if cost else None})

    rows.sort(key=lambda r: -r["市值"])
    for r in rows:
        r["权重"] = r["市值"] / eq * 100
    return rows


def write_snapshot_md(path, snap, rows, cash, eq):
    ms, m = snap["market"], snap["metrics"]
    _vp = snap.get("version_params", {})
    _cy = _vp.get("cash_yield", 0.02) or 0.0
    _cb = _vp.get("core_bear", "?")
    _lb = _vp.get("lt_bear", "?")
    _ver = snap.get("version", "?")
    _map = {"half": "减半", "zero": "清仓", "hold": "死拿"}
    L = [f"# 模拟盘快照 · {snap['date']}", "",
         f"> **引擎 {_ver}**｜转防守时核心{_map.get(_cb, _cb)}、长期股{_map.get(_lb, _lb)}"
         f"｜现金计息 {_cy*100:.1f}%｜波动目标 10%｜核心死区 5%｜裁判 MA200 + 3% 缓冲带"
         f"｜HMM 开｜费用与 T+1 全开",
         f"> 起跑 {snap['start']}｜初始资金 {snap['capital']:,.0f}｜**强平前**持仓", "",
         "## 市场状态", "", "| 项 | 值 |", "|---|---|",
         f"| 沪深300 收盘 | {ms['index_close']:.2f} |",
         f"| MA200 | {ms['ma200']:.2f} |",
         f"| 位置 | {'站上' if ms['above_ma200'] else '跌破'} |",
         f"| 裁判 | {ms['regime_text']} |",
         f"| HMM 牛市概率 | {ms['hmm_bull_prob']*100:.2f}% |",
         f"| HMM 闸门 | {'触发' if ms['hmm_fire'] else '未触发'} |", "",
         "## 账户", "", "| 项 | 值 |", "|---|---|",
         f"| 期末权益 | {snap['equity']:,.2f} |",
         f"| 累计收益 | {m.get('total_return', 0):+.2f}% |",
         f"| 年化 | {m.get('cagr', 0):+.2f}% |",
         f"| 最大回撤 | {m.get('max_drawdown', 0):.2f}% |",
         f"| 夏普 | {m.get('sharpe', 0):.2f} |",
         f"| 平均仓位 | {m.get('avg_exposure', 0):.1f}% |",
         f"| 交易笔数 | {m.get('trades', 0)} |",
         f"| 手续费 | {m.get('fees', 0):,.2f} |",
         f"| 利息（现金理财） | {m.get('interest', 0):+,.2f} |",
         f"| 交易已实现 | {m.get('realized_pnl', 0):+,.2f} |",
         f"| 持仓浮动（未实现） | {m.get('unrealized_pnl', 0):+,.2f} |", "",
         "## 持仓（强平前）", ""]
    if rows:
        L += ["| 分组 | 代码 | 名称 | 股数 | 成本 | 现价 | 市值 | 权重 | 浮盈 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            pnl = f"{r['盈亏率']:+.2f}%" if r["盈亏率"] is not None else "—"
            cost = f"{r['成本']:.3f}" if r["成本"] is not None else "—"
            L.append(f"| {r['分组']} | {r['代码']} | {r['名称']} | {r['股数']:,} | {cost} "
                     f"| {r['现价']:.3f} | {r['市值']:,.0f} | {r['权重']:.2f}% | {pnl} |")
        tot = sum(r["市值"] for r in rows)
        L.append(f"| | | **持仓合计** | | | | {tot:,.0f} | {tot/eq*100:.2f}% | |")
    else:
        L.append("（空仓）")
    L += ["", f"**现金** {cash:,.0f}（{cash/eq*100:.2f}%）——含核心减仓后的储备", "",
          "## 按真实价格折算的下单表", "",
          "> 回测用后复权价（招行 229 元 vs 实际 42 元），股数不能直接照抄；按**权重**换算成真实整手。", ""]
    if snap.get("orders_real"):
        L += ["| 分组 | 代码 | 名称 | 股数 | 价格 | 金额 | 权重 |", "|---|---|---|---:|---:|---:|---:|"]
        for o in snap["orders_real"]:
            L.append(f"| {o['分组']} | {o['代码']} | {o['名称']} | {o['股数']:,} | {o['价格']:.3f} "
                     f"| {o['金额']:,.0f} | {o['权重']:.2f}% |")
        used = sum(o["金额"] for o in snap["orders_real"])
        L.append(f"| | | **合计** | | | {used:,.0f} | {used/eq*100:.2f}% |")
    L += ["", f"**剩余现金** {eq-sum(o['金额'] for o in snap.get('orders_real') or []):,.0f}", "",
          "## 近 30 天成交", ""]
    if snap["recent_trades"]:
        L += ["| 日期 | 方向 | 标的 | 股数 | 价格 | 盈亏 |", "|---|---|---|---:|---:|---:|"]
        for t in snap["recent_trades"]:
            L.append(f"| {t['date']} | {t['side']} | {NAME_MAP.get(t['code'], t['code'])} "
                     f"| {t['shares']} | {t['price']:.3f} | {t['pnl']:+,.0f} |")
    else:
        L.append("（无）")
    L += ["", "## 说明", "",
          "- 纸上模拟：收盘价成交、100 股整手、T+1、买卖都扣佣金，卖出加印花税。",
          "- 「今日」= 最新有行情的交易日；周末/节假日自动顺延到上一交易日。",
          "- 持仓取自引擎 `run()` 的 `end_state`（强平前），与回测账本同源。",
          "- 不是投资建议，历史表现不代表未来。"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------- #
# 3) 主流程
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="模拟盘：跑到最新交易日的账户快照（引擎版本见 --version）")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None, help="默认自动取指数最后一根 K 线")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--no-hmm", action="store_true")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args(argv)

    end = pd.Timestamp(args.end) if args.end else latest_trade_date()
    os.makedirs(PAPER_DIR, exist_ok=True)
    tag = end.strftime("%Y-%m-%d")

    print("=" * 80)
    print(f"模拟盘 · {getattr(cs, 'ENGINE_VERSION', 'V4.6')} · 起跑 {args.start} → 止于 {end.date()}（最新交易日）")
    print("=" * 80)

    ms = market_state(end)
    print(f"\n【市场状态 {ms['date']}】")
    print(f"  沪深300        {ms['index_close']:>10.2f}    MA200 {ms['ma200']:>10.2f}"
          f"    {'站上' if ms['above_ma200'] else '跌破'}")
    print(f"  裁判（MA200+缓冲带） {ms['regime_text']}")
    print(f"  HMM 牛市概率   {ms['hmm_bull_prob']*100:>9.2f}%    闸门 "
          f"{'触发（强行满仓）' if ms['hmm_fire'] else '未触发'}")

    # 封版参数：显式写死，避免以后修改 Config 默认值导致历史模拟盘口径漂移。
    # 2026-11-24 补：core_bear / lt_bear 以前没写死 → 悄悄跟了 V4.6 的 zero；
    # 结果 report 头部还印着「V4.4 口径」，而实际行为是「转防守清仓」。
    # 现在两个都显式钉住，并且印进报告头。
    cfg = cs.Config(start=args.start, end=str(end.date()), capital=args.capital,
                    hmm=not args.no_hmm, vol_target=0.10, cash_yield=0.02, vol_band=0.05,
                    vol_floor=0.30, regime="ma200", regime_band=0.03,
                    core_bear="zero", lt_bear="zero")
    res = cs.run(cfg)
    m, st = res["metrics"], res.get("end_state") or {}
    eq = float(st.get("equity") or res["equity"].iloc[-1])

    codes = [cs.CORE_INDEX] + list(st.get("longterm") or {}) + list(st.get("tactical") or {})
    prices = price_map(codes, end)
    rows = holdings_table(st, prices, eq)
    pos_val = sum(r["市值"] for r in rows)
    cash = float(st.get("sat_cash") or 0.0) + float(st.get("reserve") or 0.0)

    # 真实价格口径的下单表（回测是后复权，股数不能直接照抄）
    rp = real_prices([r["代码"] for r in rows if r["分组"] != "核心"] + ["510300"], end)
    orders = order_sheet(rows, rp, eq)

    print(f"\n【账户】初始 {args.capital:,.0f} → 期末 {eq:,.0f}（{m['total_return']:+.2f}%）"
          f"｜回撤 {m['max_drawdown']:.2f}%｜夏普 {m['sharpe']:.2f}｜仓位均值 {m['avg_exposure']:.1f}%")
    if st:
        print(f"  熔断 {'开' if st.get('breaker_on') else '关'}｜裁判 "
              f"{'进攻' if st.get('bull_state') else '防守'}｜HMM {st.get('hmm_prob', 0)*100:.1f}%")

    print("\n【持仓（强平前）】")
    if not rows:
        print("  （空仓）")
    else:
        print(f"  {'分组':<4}{'代码':<8}{'名称':<16}{'股数':>10}{'成本':>9}{'现价':>9}"
              f"{'市值':>13}{'权重':>8}{'浮盈':>9}")
        for r in rows:
            cost = f"{r['成本']:.3f}" if r["成本"] is not None else "—"
            pnl = f"{r['盈亏率']:+.2f}%" if r["盈亏率"] is not None else "—"
            sh = f"{r['股数']:,.2f}" if isinstance(r["股数"], float) else f"{r['股数']:,}"
            print(f"  {r['分组']:<4}{r['代码']:<8}{r['名称']:<16}{sh:>10}{cost:>9}{r['现价']:>9.3f}"
                  f"{r['市值']:>13,.0f}{r['权重']:>7.2f}%{pnl:>9}")
        print(f"  {'':<4}{'':<8}{'持仓合计':<16}{'':>10}{'':>9}{'':>9}{pos_val:>13,.0f}{pos_val/eq*100:>7.2f}%")
    print(f"  {'':<4}{'':<8}{'现金':<16}{'':>10}{'':>9}{'':>9}{cash:>13,.0f}{cash/eq*100:>7.2f}%")

    print("\n【按真实价格折算的下单表】（回测用后复权，这里换算成能真下单的整手）")
    print(f"  {'分组':<4}{'代码':<8}{'名称':<16}{'股数':>8}{'价格':>10}{'金额':>13}{'权重':>8}")
    for o in orders:
        print(f"  {o['分组']:<4}{o['代码']:<8}{o['名称']:<16}{o['股数']:>8,}{o['价格']:>10.3f}"
              f"{o['金额']:>13,.0f}{o['权重']:>7.2f}%")
    used = sum(o["金额"] for o in orders)
    print(f"  {'':<4}{'':<8}{'合计':<16}{'':>8}{'':>10}{used:>13,.0f}{used/eq*100:>7.2f}%"
          f"   剩余现金 {eq-used:,.0f}（{(eq-used)/eq*100:.2f}%）")

    recent = [t for t in res["trades"]
              if str(getattr(t, "date", ""))[:10] >= str((end - pd.Timedelta(days=30)).date())
              and "期末平仓" not in str(getattr(t, "note", ""))]
    print(f"\n【近 30 天成交 {len(recent)} 笔】")
    for t in recent[-10:]:
        side = getattr(t, "side", "")
        code = getattr(t, "code", "")
        print(f"  {str(getattr(t, 'date', ''))[:10]}  {side:<4}{name_of(code):<14}"
              f"{int(getattr(t, 'shares', 0)):>8} 股 @ {float(getattr(t, 'price', 0)):.3f}")

    snap = {
        "date": tag, "start": args.start, "capital": args.capital,
        "equity": round(eq, 2),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "version": getattr(cs, "ENGINE_VERSION", "V4.4"),
        "version_params": {"vol_target": 0.10, "cash_yield": cfg.cash_yield, "vol_band": 0.05,
                           "vol_floor": 0.30, "regime": "ma200", "regime_band": 0.03,
                           "hmm": bool(cfg.hmm),
                           "core_bear": cfg.core_bear, "lt_bear": cfg.lt_bear,
                           "core_weight": cfg.core_weight},
        "market": ms, "end_state": st,
        "holdings": rows, "cash": round(cash, 2), "orders_real": orders,
        "recent_trades": [{"date": str(getattr(t, "date", ""))[:10],
                           "side": getattr(t, "side", ""), "code": getattr(t, "code", ""),
                           "shares": int(getattr(t, "shares", 0)),
                           "price": float(getattr(t, "price", 0) or 0),
                           "pnl": float(getattr(t, "pnl", 0) or 0)} for t in recent[-20:]],
        "config": {k: v for k, v in asdict(cfg).items()
                   if k in ("core_weight", "lt_share", "max_weight", "band", "regime", "regime_band",
                            "hmm", "hmm_threshold", "etf_top", "etf_mom", "core_bear", "lt_bear", "cash_yield")},
    }

    json_path = os.path.join(PAPER_DIR, f"paper_{tag}.json")
    md_path = os.path.join(PAPER_DIR, f"paper_{tag}.md")
    png_path = os.path.join(PAPER_DIR, f"paper_{tag}.png")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=str)
    write_snapshot_md(md_path, snap, rows, cash, eq)
    if not args.no_chart:
        try:
            cs.plot(res, png_path)
        except Exception as e:
            print(f"  [warn] 绘图失败：{e}")

    print(f"\n快照：{json_path}")
    print(f"报告：{md_path}")
    if not args.no_chart:
        print(f"曲线：{png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
