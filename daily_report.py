"""量化日报 —— 每天两个时点自动落盘（15:30 收盘 / 19:00 新闻与政策）。

设计原则
--------
1. **只读行情、只写日志**：不碰回测引擎的缓存目录结构，所有产出落到 D 盘独立文件夹。
2. **单点失败不拖垮全局**：每个数据源都包了 try/except，某一源挂了照样出日报，只是那一节写「取数失败」。
3. **幂等**：同一天重复跑会覆盖当天文件，不会重复追加（新闻除外——新闻按标题去重追加）。

两个子命令
----------
python daily_report.py close     # 15:30：收盘数据 + 模拟盘盈亏 + 个股新闻
python daily_report.py policy    # 20:00：新闻联播 + 财联社电报 + 政策关键词
python daily_report.py index     # 重建根目录的 index.md（最近 30 天清单）

目录结构
--------
.\\
  2026-11-09\\
    1530_market.json      指数/ETF 行情快照 + 裁判/HMM 状态
    1530_paper.json       当天模拟盘快照（从 paper.py 拷过来）
    1530_news.jsonl       20 只票的当日新闻
    1530_report.md        人读日报（含盈亏、持仓、新闻摘要）
    1530_equity.png       净值曲线
    1900_policy.jsonl     新闻联播/财联社/经济日历
    1900_report.md        政策摘要（含关键词命中）
  logs\\close_2026-11-09.log
  index.md
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QUANT_REPORT_DIR", "reports")   # 日报落盘目录（可用环境变量覆盖）
LOG_DIR = os.path.join(ROOT, "logs")
SEEN = os.path.join(ROOT, "_seen_hashes.txt")      # MD5 去重库（一行一个）
RETRIES = 3                                        # 失败后重试次数
RETRY_WAIT = 300                                   # 每次间隔 5 分钟

import core_satellite as cs          # noqa: E402
import cs_data                       # noqa: E402
import cs_hmm                        # noqa: E402
import cs_regime                     # noqa: E402
try:
    import news_harvest              # 可选：本仓库不含新闻抓取（涉及外部站点与凭据）
except Exception:                    # noqa: BLE001
    news_harvest = None
import paper as paper_mod            # noqa: E402

# 政策相关关键词：用于 19:00 那一轮的打标
POLICY_WORDS = [
    # 货币与监管
    "央行", "人民银行", "降准", "降息", "LPR", "MLF", "逆回购", "货币政策", "流动性", "社融",
    "证监会", "上交所", "深交所", "北交所", "IPO", "再融资", "退市", "印花税", "注册制", "减持新规",
    # 财政与顶层会议
    "国常会", "国务院", "政治局", "发改委", "财政部", "中央经济工作会议", "两会", "专项债",
    "稳增长", "扩内需", "促消费", "基建投资",
    # 地产
    "楼市", "房价", "限购", "保交楼", "城中村", "房企",
    # 外需与汇率
    "关税", "出口", "汇率", "人民币汇率", "外汇储备",
    # 产业
    "芯片", "半导体", "人工智能", "新能源", "光伏", "储能", "军工", "人形机器人", "算力",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def kill_proxy() -> None:
    """彻底绕开系统代理。

    这台机器上残留着 `127.0.0.1:7892`（代理软件退了但注册表里还写着），
    requests 会读注册表 → 所有请求 ProxyError。光删环境变量不够，
    连 urllib 的 getproxies 一起换成空函数才保险。
    """
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    import urllib.request
    urllib.request.getproxies = lambda: {}          # noqa: E731
    try:
        import requests
        requests.utils.getproxies = lambda: {}      # 老版本内部缓存
    except Exception:
        pass


def today_dir(d: pd.Timestamp | None = None) -> str:
    d = d or pd.Timestamp.now()
    p = os.path.join(ROOT, d.strftime("%Y-%m-%d"))
    os.makedirs(p, exist_ok=True)
    return p


def safe(name: str):
    """装饰器：捕获异常并记录，返回 None 而不是让整个日报崩掉。"""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:
                log(f"  [warn] {name} 失败：{type(e).__name__}: {e}")
                return None
        return wrapper
    return deco


# --------------------------------------------------------------------------- #
# 行情快照
# --------------------------------------------------------------------------- #
@safe("行情快照")
def market_snapshot(end: pd.Timestamp) -> dict:
    close = cs_data.load_index("1990-01-01", str(end.date()))
    bench = cs_regime.bench_frame(close, window=200, slope_win=0)
    states = cs_regime.build_states(bench, confirm=2, band=0.03)
    prob = cs_hmm.hmm_bull_prob(close)
    idx_last = float(close.iloc[-1])
    ma = float(bench["ma"].iloc[-1])

    etf_rows = []
    for code, (name, ind) in cs_data.ETF_UNIVERSE.items():
        d = cs_data._safe_load_etf(code, "2024-01-01", str(end), 20)
        if d is None or not len(d):
            continue
        etf_rows.append({
            "code": code, "name": name, "industry": ind,
            "close": round(float(d["close"].iloc[-1]), 3),
            "ret_1d": round(float(d["close"].pct_change().iloc[-1]) * 100, 2),
            "mom20": round(float(d.get("mom", pd.Series([float("nan")])).iloc[-1]) * 100, 2),
            "above_ma20": bool(float(d["close"].iloc[-1]) > float(d["ma20"].iloc[-1])),
        })
    etf_rows.sort(key=lambda r: -(r["mom20"] if r["mom20"] == r["mom20"] else -99))

    return {
        "date": str(end.date()),
        "index": {"code": cs.CORE_INDEX, "close": round(idx_last, 2),
                  "ma200": round(ma, 2), "above": bool(idx_last > ma),
                  "gap_pct": round((idx_last / ma - 1) * 100, 2)},
        "regime_bull": bool(states.iloc[-1]),
        "regime_text": "进攻期" if bool(states.iloc[-1]) else "防守期",
        "hmm_prob": round(float(prob.iloc[-1]), 4),
        "etf_count": len(etf_rows),
        "etf": etf_rows,
    }


# --------------------------------------------------------------------------- #
# 15:30 收盘
# --------------------------------------------------------------------------- #
def run_paper(capital: float = 1_000_000.0) -> dict | None:
    """就地跑一次模拟盘（复用 paper.py），返回它的快照 dict。"""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            paper_mod.main(["--capital", str(capital)])
    except SystemExit as e:
        if e.code not in (0, None):
            log(f"  [warn] paper 退出码 {e.code}")
    except Exception as e:
        log(f"  [warn] paper 异常：{type(e).__name__}: {e}")
        return None
    finally:
        text = buf.getvalue()
        if text:
            log(text[-2500:])

    pdir = os.path.join(BASE, "paper")
    # 只认「paper_YYYY-MM-DD.json」这种正本，按修改时间取最新；
    # 不能按文件名排序——备份 paper_..._v42.json 会排在正本后面被误取（曾导致日报取到旧口径）。
    pat = re.compile(r"^paper_\d{4}-\d{2}-\d{2}\.json$")
    cands = []
    if os.path.isdir(pdir):
        for f in os.listdir(pdir):
            if pat.match(f):
                fp = os.path.join(pdir, f)
                cands.append((os.path.getmtime(fp), fp))
    if not cands:
        return None
    latest = max(cands)[1]
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def cmd_close(args) -> int:
    stamp = pd.Timestamp.now()
    ddir = today_dir(stamp)
    day = stamp.strftime("%Y-%m-%d")
    log(f"== 15:30 收盘日报 · {day} ==")

    # 新浪日线当天往往不出（要等到晚上），东财有 → 先把缓存补到当天，
    # 否则 15:30 这趟会拿着昨天的收盘价算，日报看起来"成功"但数字是旧的。
    try:
        import _refresh_tail
        _refresh_tail.main()
    except Exception as e:  # noqa: BLE001
        log(f"  [warn] 行情补齐失败（继续用现有缓存）：{type(e).__name__}: {e}")

    snap = run_paper(args.capital)
    if snap:
        with open(os.path.join(ddir, "1530_paper.json"), "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2, default=str)
        png = os.path.join(BASE, "paper", f"paper_{snap['date']}.png")
        if os.path.exists(png):
            shutil.copy(png, os.path.join(ddir, "1530_equity.png"))

    # 行情滞后检查：数据源（新浪/东财）有时当天不出数，
    # 若模拟盘用的是几天前的收盘价，必须在日报顶部显式标出来，不能静默沿用。
    stale_days = 0
    if snap:
        try:
            stale_days = max(0, (pd.Timestamp.now().normalize() - pd.Timestamp(snap["date"]).normalize()).days)
        except Exception:
            stale_days = 0
        if stale_days >= 1:
            log(f"  [warn] 行情滞后 {stale_days} 天：模拟盘最新交易日是 {snap['date']}，不是今天")

    end = pd.Timestamp(snap["date"]) if snap else paper_mod.latest_trade_date()
    mk = market_snapshot(end)
    if mk:
        with open(os.path.join(ddir, "1530_market.json"), "w", encoding="utf-8") as f:
            json.dump(mk, f, ensure_ascii=False, indent=2)

    news = None
    try:
        codes = list(cs_data.DEFAULT_REGULAR) + list(cs_data.DEFAULT_LONGTERM)
        news = news_harvest.harvest(codes, days=1, with_globals=True, min_rel=1)
        log(f"  新闻：{news}")
        # harvest 只回计数，这里补上累计条数与今日样本，供报告用
        try:
            with open(news_harvest.ARCHIVE, encoding="utf-8") as f:
                rows = [json.loads(x) for x in f if x.strip()]
            news["total"] = len(rows)
            by_tag: dict[str, int] = {}
            days_set = set()
            for r in rows:
                by_tag[r.get("tag", "stock")] = by_tag.get(r.get("tag", "stock"), 0) + 1
                days_set.add(str(r.get("ts", ""))[:10])
            news["by_tag"] = by_tag
            news["days"] = len(days_set)
            news["market"] = sum(1 for r in rows if r.get("code") == "MARKET")
            news["stock"] = sum(1 for r in rows if r.get("code") != "MARKET")
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            picked = [r for r in rows if str(r.get("ts", "")).startswith(today)
                      or str(r.get("harvested", "")).startswith(today)]
            news["samples"] = picked[-10:] or rows[-5:]
        except Exception as e:
            log(f"  [warn] 读新闻库失败：{type(e).__name__}: {e}")
    except Exception as e:
        log(f"  [warn] 新闻抓取失败：{type(e).__name__}: {e}")

    write_close_report(ddir, day, snap, mk, news, stale_days=stale_days,
                       practice_capital=getattr(args, "practice_capital", 0.0) or 0.0)
    with contextlib.suppress(Exception):
        cmd_index(None)          # 顺手刷新根目录索引
    log(f"  输出目录：{ddir}")
    if stale_days >= 1 and not getattr(args, "allow_stale", False):
        log(f"  [FAIL] 行情滞后 {stale_days} 天：日报数字不是当天的，退出码 3"
            f"（确认要放行就加 --allow-stale）")
        return 3
    return 0


def practice_section(snap, capital: float) -> list[str]:
    """零成本练习：把 100 万模拟盘的权重，等比缩到「如果你只有 X 元」。

    取整规则：四舍五入到整手（不是向下取整）——否则 2.3% 权重在 10 万下会变成 0 手，
    长期股全部消失，看起来像"策略不买长期股"。买不起就明确写 0，不偷偷跳过。
    """
    if not snap or not snap.get("orders_real"):
        return []
    orders = sorted(snap["orders_real"], key=lambda o: -float(o["权重"]))
    L = [f"## 如果你只有 {capital:,.0f} 元", "",
         "> 同样的信号，把你的钱缩到这个规模会买什么（纯推演，不动真钱）。", "",
         "| 分组 | 代码 | 名称 | 股数 | 手数 | 价格 | 金额 | 权重 |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    left = capital
    used = 0.0
    lot_sum = 0.0
    min_cap = 0.0
    for o in orders:
        px = float(o["价格"])
        w = float(o["权重"]) / 100.0
        lot = px * 100
        lot_sum += lot
        want = capital * w
        lots = int(round(want / lot))
        if lots == 0 and want >= lot * 0.5:
            lots = 1
        shares = lots * 100
        amt = shares * px
        if amt > left:            # 钱不够就不买这一只，不留负现金
            shares, amt = 0, 0.0
        left -= amt
        used += amt
        if shares > 0 and w > 0:
            min_cap = max(min_cap, lot / w)
        mark = " ※" if (shares == 0 and lots == 0) else ""
        L.append(f"| {o['\u5206\u7ec4']} | {o['\u4ee3\u7801']} | {o['\u540d\u79f0']} | {shares:,} | "
                 f"{shares // 100} | {px:.3f} | {amt:,.0f} | {amt / capital * 100:.2f}%{mark} |")
    L += ["", f"**合计 {used:,.0f} 元（{used / capital * 100:.1f}%），剩余现金 {left:,.0f} 元**", ""]

    bought = sum(1 for o in orders if True)
    L += ["", "※ 目标是 0 手：该仓位在这么小的资金下不足半手，跳过。"] if "※" in "".join(L) else []
    L += [f"- 全部标的各买一手，最少需要 **{lot_sum:,.0f} 元**；",
          f"- 按目标权重买，建议至少 **{min_cap:,.0f} 元**（低于这个数，某只票一手就超过它的目标权重）。"]
    if capital < lot_sum:
        L += ["", "> ⚠️ 现在的钱不够把 7 只各买一手，只能挑一部分——"
                  "这就是小资金的真实约束，不是策略不买，是买不进去。"]
    L.append("")
    return L


def write_close_report(ddir, day, snap, mk, news, stale_days: int = 0, practice_capital: float = 0.0):
    L = [f"# 量化日报 · {day}（15:30 收盘）", ""]

    if stale_days >= 1:
        latest = snap["date"] if snap else "未知"
        L += [
            f"> ⚠️ **数据滞后 {stale_days} 天**：以下盈亏基于 **{latest}** 的收盘价，"
            f"不是今天（{day}）。行情源尚未发布最新交易日数据时会出现这种情况，"
            f"不代表策略今天没有动作。",
            "",
        ]

    if mk:
        ix = mk["index"]
        L += ["## 市场", "", "| 项 | 值 |", "|---|---|",
              f"| 沪深300 | {ix['close']:.2f} |",
              f"| MA200 | {ix['ma200']:.2f} |",
              f"| 位置 | {'站上' if ix['above'] else '跌破'}（{ix['gap_pct']:+.2f}%） |",
              f"| 裁判 | {mk['regime_text']} |",
              f"| HMM 牛市概率 | {mk['hmm_prob']*100:.2f}% |", ""]
        top = [r for r in mk["etf"] if r["mom20"] == r["mom20"]][:5]
        if top:
            L += ["**行业动量前五**（20 日）", "", "| 代码 | 名称 | 收盘 | 日涨跌 | 20日动量 | 站上MA20 |",
                  "|---|---|---:|---:|---:|---|"]
            for r in top:
                L.append(f"| {r['code']} | {r['name']} | {r['close']:.3f} | {r['ret_1d']:+.2f}% "
                         f"| {r['mom20']:+.2f}% | {'✓' if r['above_ma20'] else '✗'} |")
            L.append("")

    if snap:
        m = snap["metrics"]
        _vp = snap.get("version_params", {}) or {}
        _map = {"half": "减半", "zero": "清仓", "hold": "死拿"}
        _cb = _map.get(_vp.get("core_bear", "?"), _vp.get("core_bear", "?"))
        _lb = _map.get(_vp.get("lt_bear", "?"), _vp.get("lt_bear", "?"))
        _cy = _vp.get("cash_yield", 0.02) or 0.0
        L += ["## 模拟盘", "", "| 项 | 值 |", "|---|---|",
              f"| 引擎 | **{snap.get('version', '?')}**｜转防守时核心{_cb}、长期股{_lb}｜现金计息 {_cy:.1%} |",
              f"| 最新交易日 | {snap['date']} |",
              f"| 权益 | {snap['equity']:,.2f} |",
              f"| 累计收益 | {m.get('total_return', 0):+.2f}% |",
              f"| 最大回撤 | {m.get('max_drawdown', 0):.2f}% |",
              f"| 夏普 | {m.get('sharpe', 0):.2f} |",
              f"| 现金 | {snap.get('cash', 0):,.0f} |",
              f"| 利息（现金理财） | {m.get('interest', 0):+,.2f} |",
              f"| 交易已实现 | {m.get('realized_pnl', 0):+,.2f} |",
              f"| 持仓浮动（未实现） | {m.get('unrealized_pnl', 0):+,.2f} |", ""]
        held = next((r for r in ([m] if isinstance(m, list) else []) if False), None)  # 占位，保持结构清晰
        rows = snap.get("orders_real") or []
        if rows:
            L += ["**当前配置（真实价格口径）**", "", "| 分组 | 代码 | 名称 | 股数 | 价格 | 金额 | 权重 |",
                  "|---|---|---|---:|---:|---:|---:|"]
            for o in rows:
                L.append(f"| {o['分组']} | {o['代码']} | {o['名称']} | {o['股数']:,} | {o['价格']:.3f} "
                         f"| {o['金额']:,.0f} | {o['权重']:.2f}% |")
            L.append("")

    if news:
        L += ["## 新闻", "", f"- 新增入库：**{news.get('added', 0) + news.get('global_added', 0)}** 条（个股 {news.get('added', 0)}｜全市场 {news.get('global_added', 0)}），"
            f"去重跳过 {news.get('skipped', 0) + news.get('global_skipped', 0)} 条",
            f"- 库状态：累计 **{news.get('total', 0)}** 条｜覆盖 **{news.get('days', 0)}** 天｜"
            f"个股 {news.get('stock', 0)} · 全市场 {news.get('market', 0)}"
            f"（政策 {news.get('by_tag', {}).get('policy', 0)} / 行业 {news.get('by_tag', {}).get('industry', 0)} / 个股 {news.get('by_tag', {}).get('stock', 0)}）",
            f"- 距可回测：还需 **{max(0, 120 - news.get('days', 0))}** 个交易日（门槛 120 天 / 5000 条）",
              f"- 累计库：{news.get('total', '—')} 条", ""]
        for item in (news.get("samples") or [])[:10]:
            L.append(f"- `{item.get('code', '')}` {item.get('title', '')[:60]}")
        L.append("")

    if practice_capital and practice_capital > 0:
        L += practice_section(snap, practice_capital)

    L += ["---", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    with open(os.path.join(ddir, "1530_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------- #
# 20:00 政策与行业新闻（MD5 去重 + 分类 + 摘要）
# --------------------------------------------------------------------------- #
# 宏观政策：货币、财政、监管、国际
MACRO_WORDS = [
    "央行", "人民银行", "降准", "降息", "LPR", "MLF", "逆回购", "公开市场操作",
    "货币政策", "流动性", "社融", "信贷", "M2",
    "证监会", "上交所", "深交所", "北交所", "IPO", "再融资", "注册制", "退市",
    "印花税", "减持新规", "监管", "立案", "处罚", "罚款",
    "国常会", "国务院", "政治局", "中央经济工作会议", "两会", "发改委", "财政部",
    "专项债", "国债", "财政政策", "减税", "降费", "稳增长", "扩内需", "促消费",
    "楼市", "房价", "限购", "保交楼", "城中村", "房企", "房贷",
    "关税", "贸易摩擦", "出口", "汇率", "人民币汇率", "外汇储备", "美联储", "非农", "CPI", "PMI",
]
# 行业事件：利好 / 利空（只用来分栏，不参与打分）
GOOD_WORDS = [
    "中标", "订单", "签约", "合作", "涨价", "提价", "扩产", "投产", "量产", "获批",
    "专利", "突破", "超预期", "预增", "扭亏", "回购", "增持", "分红", "上调",
    "涨停", "新高", "翻倍", "放量",
]
BAD_WORDS = [
    "减持", "亏损", "预亏", "下调", "跌价", "降价", "停产", "召回", "处罚",
    "立案", "调查", "退市", "违约", "爆雷", "商誉减值", "计提", "终止", "解禁",
    "跌停", "新低", "腰斩", "诉讼",
]
INDUSTRY_WORDS = ["芯片", "半导体", "光伏", "储能", "锂电", "新能源", "军工", "医药",
                  "白酒", "地产", "煤炭", "有色", "钢铁", "化工", "汽车", "机器人",
                  "算力", "AI", "人工智能", "消费", "银行", "券商", "保险"]


def _log_title() -> str:
    return "20:00"


@safe("新闻联播")
def fetch_cctv(day: pd.Timestamp) -> list[dict]:
    import akshare as ak
    df = ak.news_cctv(date=day.strftime("%Y%m%d"))
    out = []
    for _, r in df.iterrows():
        out.append({"source": "cctv", "ts": str(r.get("date", "")),
                    "title": str(r.get("title", "")), "content": str(r.get("content", ""))})
    return out


@safe("财联社电报")
def fetch_cls() -> list[dict]:
    import akshare as ak
    df = ak.stock_info_global_cls()
    out = []
    for _, r in df.head(120).iterrows():
        out.append({"source": "cls",
                    "ts": f"{r.get('发布日期', '')} {r.get('发布时间', '')}".strip(),
                    "title": str(r.get("标题", "")), "content": str(r.get("内容", ""))})
    return out


@safe("东财全球快讯")
def fetch_em_global() -> list[dict]:
    import akshare as ak
    df = ak.stock_info_global_em()
    out = []
    for _, r in df.head(120).iterrows():
        out.append({"source": "em", "ts": str(r.get("发布时间", "")),
                    "title": str(r.get("标题", "")), "content": str(r.get("摘要", ""))})
    return out


def norm_text(s: str) -> str:
    s = re.sub(r"财联社\s*\d+月\d+日电[，,：:]?", "", s or "")
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s)


def md5_of(title: str, content: str) -> str:
    """通稿去重的核心：把标题+正文前 200 字归一化后取 MD5。"""
    key = norm_text(title)[:60] + "|" + norm_text(content)[:200]
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_seen() -> dict[str, str]:
    """hash -> 首次出现日期。跨日重复才算真重复（同一天重跑要能补全当天文件）。"""
    if not os.path.exists(SEEN):
        return {}
    out: dict[str, str] = {}
    with open(SEEN, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                out.setdefault(parts[0], parts[1])
    return out


def save_seen(pairs: list[tuple[str, str]]) -> None:
    if not pairs:
        return
    os.makedirs(ROOT, exist_ok=True)
    seen = load_seen()
    with open(SEEN, "a", encoding="utf-8") as f:
        for h, d in pairs:
            if h not in seen:
                f.write(f"{h} {d}\n")


def classify(title: str, content: str) -> tuple[str, str, list[str]]:
    """返回 (category, polarity, hits)。category: macro / industry / other。"""
    text = f"{title} {content}"
    macro_hits = [w for w in MACRO_WORDS if w in text]
    good = [w for w in GOOD_WORDS if w in text]
    bad = [w for w in BAD_WORDS if w in text]
    ind = [w for w in INDUSTRY_WORDS if w in text]

    if macro_hits:
        return "macro", ("利空" if len(bad) > len(good) else "利好" if good else "中性"), macro_hits
    if ind or good or bad:
        pol = "利空" if len(bad) > len(good) else "利好" if good or ind else "中性"
        return "industry", pol, (good + bad + ind)
    return "other", "中性", []


def brief(text: str, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    return s[:n] + ("…" if len(s) > n else "")


def cmd_policy(args) -> int:
    stamp = pd.Timestamp.now()
    ddir = today_dir(stamp)
    day = stamp.strftime("%Y-%m-%d")
    log(f"== 20:00 宏观政策/行业新闻 · {day} ==")

    items: list[dict] = []
    for fn in (fetch_cctv, fetch_cls, fetch_em_global):
        res = fn(stamp) if fn is fetch_cctv else fn()
        if res:
            items.extend(res)
            log(f"  {fn.__name__}: {len(res)} 条")
    if not items:
        raise RuntimeError("三个新闻源全部失败")

    seen = load_seen()
    fresh: list[dict] = []
    new_hashes: list[str] = []
    dup_in_batch = 0
    batch: set[str] = set()
    for it in items:
        title = (it.get("title") or "").strip()
        content = (it.get("content") or "").strip()
        if not title:
            title = brief(content, 60)
            it["title"] = title
        h = md5_of(title, content)
        if h in batch:                       # 同一批里重复
            dup_in_batch += 1
            continue
        prev_day = seen.get(h)
        if prev_day and prev_day != day:     # 跨天重复 → 真正要挡的
            dup_in_batch += 1
            continue
        batch.add(h)
        new_hashes.append((h, day))
        cat, pol, hits = classify(title, content)
        fresh.append({
            "hash": h, "ts": it.get("ts", ""), "source": it["source"],
            "category": cat, "polarity": pol,
            "title": title, "summary": brief(content, 200),
            "hits": hits,
        })
    log(f"  去重：{len(items)} 条 → 入库 {len(fresh)} 条（跨天/批内重复跳过 {dup_in_batch} 条）")

    # 与当天已有文件合并（重试/手动补跑时不能把之前的成果冲掉）
    path = os.path.join(ddir, "2000_news.jsonl")
    merged: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        old = json.loads(line)
                        merged[old["hash"]] = old
                    except Exception:
                        pass
    for it in fresh:
        merged[it["hash"]] = it
    day_items = sorted(merged.values(), key=lambda x: x.get("ts", ""), reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in day_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_policy_report(ddir, day, day_items, len(items), dup_in_batch, len(fresh))
    save_seen(new_hashes)
    with contextlib.suppress(Exception):
        cmd_index(None)
    log(f"  当日累计写入 {path}")
    return 0


def write_policy_report(ddir, day, items, total, dup, added=0):
    macro = [x for x in items if x["category"] == "macro"]
    good = [x for x in items if x["category"] == "industry" and x["polarity"] == "利好"]
    bad = [x for x in items if x["category"] == "industry" and x["polarity"] == "利空"]

    L = [f"# 宏观政策 / 行业新闻 · {day}（20:00）", "",
         f"本次抓取 **{total}** 条，本次入库 **{added}** 条，跳过重复 **{dup}** 条；当日累计 **{len(items)}** 条。", "",
         f"- 宏观政策：**{len(macro)}** 条｜行业利好：**{len(good)}** 条｜行业利空：**{len(bad)}** 条", ""]

    def section(title, rows):
        L.append(f"## {title}（{len(rows)}）")
        L.append("")
        if not rows:
            L.append("（空）")
            L.append("")
            return
        for x in rows[:25]:
            L.append(f"- **[{x['source']}]** {x['title'][:90]}")
            if x["summary"]:
                L.append(f"  - {x['summary']}")
            if x["hits"]:
                L.append(f"  - 关键词：{'、'.join(dict.fromkeys(x['hits']))[:60]}")
        L.append("")

    section("宏观政策", macro)
    section("行业利好", good)
    section("行业利空", bad)

    from collections import Counter
    c = Counter(w for x in items for w in x["hits"])
    if c:
        L += ["## 关键词热度", "", "| 关键词 | 次数 |", "|---|---:|"]
        for w, n in c.most_common(12):
            L.append(f"| {w} | {n} |")
        L.append("")

    L += ["---", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    with open(os.path.join(ddir, "2000_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------- #
# 索引
# --------------------------------------------------------------------------- #
def cmd_index(args) -> int:
    os.makedirs(ROOT, exist_ok=True)
    dirs = sorted([d for d in os.listdir(ROOT)
                   if os.path.isdir(os.path.join(ROOT, d)) and d[:2] == "20"], reverse=True)[:30]
    L = ["# 量化日报索引", "", f"根目录：`{ROOT}`｜更新于 {datetime.now():%Y-%m-%d %H:%M}", "",
         "| 日期 | 收盘日报 | 政策日报 | 模拟盘权益 |", "|---|---|---|---:|"]
    for d in dirs:
        p = os.path.join(ROOT, d)
        has_close = os.path.exists(os.path.join(p, "1530_report.md"))
        has_policy = os.path.exists(os.path.join(p, "2000_report.md"))
        eq = "—"
        pj = os.path.join(p, "1530_paper.json")
        if os.path.exists(pj):
            try:
                with open(pj, encoding="utf-8") as f:
                    eq = f"{json.load(f).get('equity', 0):,.0f}"
            except Exception:
                pass
        L.append(f"| {d} | {'✓' if has_close else '—'} | {'✓' if has_policy else '—'} | {eq} |")
    L += ["", "## 用法", "", "```bash",
          "python daily_report.py close     # 15:30",
          "python daily_report.py policy    # 20:00",
          "python daily_report.py index", "```", ""]
    with open(os.path.join(ROOT, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"index.md 已更新（{len(dirs)} 天）")
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
_RUN_START = 0.0


def _fresh(path: str) -> bool:
    """文件存在且是这次运行之后写的，才算本次成功。"""
    try:
        return os.path.exists(path) and os.path.getmtime(path) >= _RUN_START - 2
    except OSError:
        return False


def _close_ok() -> bool:
    d = today_dir()
    need = ["1530_report.md", "1530_paper.json", "1530_market.json"]
    return all(_fresh(os.path.join(d, n)) for n in need)


def _policy_ok() -> bool:
    d = today_dir()
    return _fresh(os.path.join(d, "2000_news.jsonl"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="量化日报：15:30 收盘 / 20:00 政策")
    sub = p.add_subparsers(dest="cmd", required=True)
    c1 = sub.add_parser("close", help="收盘日报")
    c1.add_argument("--capital", type=float, default=1_000_000.0)
    c1.add_argument("--practice-capital", type=float, default=100_000.0,
                    help="日报附「如果你只有这么多钱」推演，0=关闭")
    c1.add_argument("--allow-stale", action="store_true",
                    help="行情滞后也返回 0（默认滞后即退出码 3，防「成功但数字是旧的」）")
    c1.set_defaults(func=cmd_close)
    c2 = sub.add_parser("policy", help="宏观政策与行业新闻")
    c2.set_defaults(func=cmd_policy)
    c3 = sub.add_parser("index", help="重建索引")
    c3.set_defaults(func=cmd_index)
    for sp in (c1, c2):
        sp.add_argument("--retries", type=int, default=RETRIES, help="失败重试次数")
        sp.add_argument("--retry-wait", type=int, default=RETRY_WAIT, help="重试间隔（秒）")
    args = p.parse_args(argv)

    kill_proxy()
    # 计划任务的工作目录是 C:\Windows\System32，相对路径缓存会写不进去 → 先切回项目目录
    os.chdir(BASE)
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    logpath = os.path.join(LOG_DIR, f"{args.cmd}_{stamp}.log")

    buf = io.StringIO()

    class Tee:
        def write(self, s_):
            sys.__stdout__.write(s_)
            buf.write(s_)
        def flush(self):
            sys.__stdout__.flush()

    global _RUN_START
    _RUN_START = time.time()
    ok_check = {"close": _close_ok, "policy": _policy_ok}.get(args.cmd)
    tries = 1 if args.cmd == "index" else max(1, args.retries + 1)
    old = sys.stdout
    rc = 1
    for attempt in range(1, tries + 1):
        sys.stdout = Tee()
        try:
            log(f"\n----- 第 {attempt}/{tries} 次尝试 {datetime.now():%H:%M:%S} -----")
            rc = args.func(args)
            ok = (rc == 0) and (ok_check() if ok_check else True)
        except Exception:
            traceback.print_exc()
            ok = False
        finally:
            sys.stdout = old
        if ok:
            break
        if attempt < tries:
            log(f"[retry] 本次未成功，{args.retry_wait} 秒（{args.retry_wait//60} 分钟）后重试…")
            time.sleep(args.retry_wait)
            buf.write(f"[retry] 第 {attempt} 次失败，准备第 {attempt+1} 次\n")

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(logpath, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
