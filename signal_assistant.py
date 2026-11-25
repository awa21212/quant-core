# -*- coding: utf-8 -*-
"""
================================================================================
  量化交易策略信号判定助手（Signal Assistant）
================================================================================

【它做什么】
    给定「标的 + 交易日 T」，只用 T 日收盘及之前的数据，按硬规则判定该不该开仓，
    最后在 stdout 输出一段**纯 JSON**（可直接被别的程序 / 大模型 / 交易系统消费）。

    stdout 只有 JSON；所有诊断信息、表格、警告一律走 stderr —— 这样管道里拿到的
    永远是干净的机器可读结果：
        python signal_assistant.py --symbol 000001 --date 2026-11-06 --position flat > signal.json

【判定顺序（严格按优先级）】
    0. 铁律止损（最高优先级，--stop-loss，默认 3%）：持仓成本已知且收盘价跌破
       成本 × (1 − 3%) → 无条件清仓 100%，不看均线、不看情绪、不许扛单。
    1. 风控兜底：持仓中且收盘价 < MA20 → 清仓，不再看任何其它条件。
    2. 重大利空硬否决：减持 / 处罚 / 立案 / 业绩暴雷 / 重大诉讼 / 退市风险 →
       情绪条件直接不通过（信息上的"一票否决"）。
    3. 技术条件：金叉(trend) + 情绪(sentiment) 同时为真 → 开仓；
       KER **不再当硬门槛**，只用来决定"开多大仓"（见下一节）。
    4. 再平衡（可选，--rebalance）：已持仓且没破位时，比较「当前权重 vs KER 目标权重」，
       差额够大就给出 加仓/减仓（默认只建议、不动作）；浮亏时不许加仓。

【三铁律（--size-mode iron + --stop-loss + fundamental_filter.py）】
    铁律1 仓位：震荡（KER<0.30）只投 20%，趋势（KER≥0.30）上 80%，
                 浮亏时封顶 40%（不给亏损单加码）→ 任何情况都留至少 20% 现金。
    铁律2 过滤：不买垃圾股。signal_assistant 只算技术面，**底子脏不脏**要看
                 fundamental_filter.py（ST/退、PE≤0、连续两年亏损、治理雷）。
                 回测时用 pipeline.py --filter 先过一遍，被拉黑直接 exit。
    铁律3 止损：跌破买入成本 3% 无条件清仓（--stop-loss 可改，0 = 关闭）。

【KER 的定位：做仓位，不做门槛】
    金叉常发生在趋势刚起步、价格还在震荡区的时候，KER 天然偏低；把 KER>0.3 当硬门槛，
    会把大部分机会直接掐掉（实测 2023 年 5 次金叉里只有 1 次能过 0.3）。
    所以改成**仓位系数**（--size-mode 四选一）：
        linear（默认）：weight = max_weight × clip((KER - lo) / (hi - lo), 0, 1)
                        默认 lo=0.10、hi=0.60 → KER≤0.10 不建仓，≥0.60 满仓，中间线性过渡
        tier（分档）：  KER <0.15 → 30%；0.15~0.30 → 50%；0.30~0.50 → 70%；>0.50 → 100%
        iron（铁律）：  KER <0.30 → 20%；KER ≥0.30 → 80%；浮亏封顶 40%（留现金、浮盈才加）
        none：           固定满仓（等价于只用"趋势+情绪"）
    效果：KER 低就轻仓试错、KER 高才重仓吃趋势，而不是"要么全仓要么不做"。
    注意：linear 模式下 KER ≤ lo 时目标仓位是 0，那就不算开仓（观望），但原因是"仓位不足"，
    不是"KER 不合格"——KER 不再是门槛，只是把仓位调到 0 而已。

【数据清洗（在算指标之前跑，结果写进 JSON 的 data_quality）】
    1. 同一天重复行情（数据源拼接造成）→ 只保留最后一条；
    2. close 缺失 / ≤0 / 非数字 → 丢弃该行；
    3. open/high/low 缺失 → 用当日 close 补；
    4. OHLC 内部矛盾（high < max(open,close) 或 low > min(open,close)）→ 就地修正；
    5. 零成交量且 high==low（数据源给的"假停牌行"）→ 丢弃；
    6. 孤立毛刺（单日涨跌 > 21%，次日反向回吞 > 50%，且价格回到跳变前水平 ±5% 内）→
       判为脏数据（错录/串行/单位错），用前一日收盘替换；
    7. 连续同向跳空（未复权除权残留）→ 只标记 suspect_adjustments，不擅自改（改错了更糟）；
    8. 交易日间隔 > 10 天 → 记为 gap（长假正常，连续停牌会露馅）；
    9. 清洗后不足 30 根 K 线 → 指标不可靠，直接"数据不足/观望"。

【口径定义（很重要，全部写在代码里，避免"看起来像"）】
  * MA5 / MA20：最近 5 / 20 个交易日**收盘价简单平均**，含 T 日。
  * 金叉：MA5_T > MA20_T 且 MA5_{T-1} <= MA20_{T-1}。只算"当日刚上穿"，
    已经在上方多日属于"趋势延续"，不产生新开仓信号（但会写进 reasons）。
  * KER14 (Kaufman 效率系数)：
        KER = |C_T - C_{T-14}| / Σ|C_i - C_{i-1}|   (i = T-13 .. T，共 14 项)
    分母为 0 → 视为无效(ker14 = null)，此时仓位按最低档处理（不建仓）。
    KER 只影响 position_size（目标仓位），不影响"能不能开仓"。
  * AI 情绪分 S：取 **T 日 15:00 往前推 24 小时**窗口内的新闻/公告/研报
    （即 (T-1 15:00, T 15:00]），逐条让本地 Ollama 打一个 [-1, 1] 的分
    （-1 极度利空，+1 极度利好），再**等权平均**得到 S，裁剪到 [-1, 1]。
    口径：等权、每条稿件权重相同，不做时间衰减（可加 --decay 开启半衰期 12h）。
    S > 0.2 视为情绪配合。
  * 重大利空：关键词匹配（兜底，召回优先）+ 可选 LLM 复核（--llm-veto）。

【为什么窗口是 (T-1 15:00, T 15:00] 而不是 "T 日 00:00 起 24 小时"】
    A 股 15:00 收盘。T 日收盘后（15:00 之后）发布的公告，你在收盘前根本不可能看到，
    用它做 T 日决策就是**未来函数**。所以窗口右端严格卡在 T 日 15:00（含），
    左端 24 小时前。周一的窗口自然覆盖周末——这符合"周末的消息周一开盘才反映"的现实。

【禁止未来函数的工程保证】
    * 所有行情切片都是 df.loc[:T]，T 之后的行在计算时物理上不存在；
    * 新闻只取发布时刻 <= T 15:00 的条目；
    * 脚本自带 `--audit` 自检：把 T 之后的所有收盘价随机篡改 +80%，再算一遍，
      两次输出必须完全一致，否则判定"有未来函数"并报错（返回码 3）。

【依赖】
    pip install pandas numpy akshare
    （matplotlib 不需要；情绪打分需要本机 Ollama，没有也能跑，只是情绪的兜底口径不同）

【数据源与复权口径】
    优先东方财富（stock_zh_a_hist, adjust="qfq"），失败切新浪（stock_zh_a_daily, 同样前复权），
    抓到的区间会缓存成 data_<代码>_<起>_<止>.csv，之后同一区间不再联网。
    注意：两家的「前复权基准」不同，所以**同一个交易日的绝对价位可能对不上**
    （实测 2023-07-25：东财 9.55 / 新浪 9.30）。这不影响金叉判定与 KER 这类相对指标，
    但绝对价格不要跨数据源比较；一次运行内只用一个数据源，保证自洽。

【已知局限（诚实说明）】
    * 东方财富的个股新闻接口只返回「最近」一批新闻，历史日期（如 2023 年）基本取不到，
      这时情绪分会如实为 null → 输出"数据不足/观望"，绝不瞎编分数。
      要做历史回测，请用 --news-file 手工喂当时可见的新闻。
    * 关键词利空检测是「召回优先」，会有误报（如"减持计划终止"也会命中"减持"），
      可用 --llm-veto 让模型复核一遍。
    * LLM 打的是"观点分"，不是事实分；同一批新闻换模型结果会变，所以缓存里记录了模型
      未记录（按标题+时间键），换模型建议先删 sentiment_cache.json。

【用法】
    # 1) 最简单的用法：今天（或最近交易日），空仓，看能不能买
    python signal_assistant.py --symbol 000001 --position flat

    # 2) 指定历史日期回看（注意：新闻接口只能拿到"最近"的新闻，
    #    历史日期大概率没有新闻 → 会如实输出"数据不足/观望"，不会瞎编）
    python signal_assistant.py --symbol 000001 --date 2023-07-25 --position flat

    # 3) 手工喂新闻（回测 / 离线验证用）：一行一个 JSON
    #    {"ts": "2023-07-25 09:30:00", "title": "平安银行：上半年净利润同比增长", "text": "..."}
    python signal_assistant.py --symbol 000001 --date 2023-07-25 --news-file news.jsonl

    # 4) 只想要技术面，不碰网络和模型
    python signal_assistant.py --symbol 000001 --date 2023-07-25 --sentiment-source none

    # 5) 未来函数自检
    python signal_assistant.py --symbol 000001 --date 2023-07-25 --audit

    # 6) 持仓中，按 KER 做再平衡（目标 41.3%，当前 20% → 建议加仓 21.3%）
    python signal_assistant.py --symbol 000001 --date 2023-07-25 \
        --position long --current-weight 0.20 --rebalance

【再平衡（--rebalance）】
    默认只输出"目标仓位"当建议，不产生调仓动作（调用方自己决定要不要执行）。
    加上 --rebalance 并给出 --current-weight 后，持仓状态下会比较「当前权重 vs KER 目标权重」：
        * 差额 < --rebalance-min（默认 5%）→ 持有不动（调仓收益盖不住摩擦成本）；
        * 差额 > 0 → position_action = "加仓"，rebalance_delta 为正；
        * 差额 < 0 → position_action = "减仓"，rebalance_delta 为负。
    position_action 因此多了两个取值：加仓 / 减仓（原来的 开仓/清仓/持有/不操作 不变）。
    注意：这是**基于 KER 变化的目标仓位再平衡**，不含手续费/滑点/整手约束，
    真要落单请自己套一层执行层（按股数取整、扣费、A 股 T+1）。

【返回码】
    0 正常出信号（含"观望"）   2 行情数据拿不到   3 自检发现未来函数
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd

# 控制台编码兜底：Windows 默认 GBK，打印中文/emoji 会炸
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

# ------------------------------------------------------------------------------
# 全局参数（都可以用命令行覆盖）
# ------------------------------------------------------------------------------
MA_FAST = 5
MA_SLOW = 20
KER_N = 14
SENT_MIN = 0.2          # 情绪阈值
NEWS_WINDOW_HOURS = 24  # 情绪窗口：T 日 15:00 往前推 24 小时
CLOSE_HOUR = 15         # A 股收盘时刻

# KER → 目标仓位（不设门槛，只调仓位）
SIZE_MODE = "linear"    # linear / tier / none
KER_LO = 0.10           # linear：KER 到此值仓位为 0
KER_HI = 0.60           # linear：KER 到此值满仓
MAX_WEIGHT = 1.0        # 单票最大仓位（1.0 = 满仓）
TIER_TABLE = [          # tier 模式：(KER 下限, KER 上限, 目标仓位)
    (0.00, 0.15, 0.30),
    (0.15, 0.30, 0.50),
    (0.30, 0.50, 0.70),
    (0.50, 99.0, 1.00),
]
MIN_BARS = 30           # 清洗后少于此根数，指标不可靠 → 观望

# ===== 铁律仓位（--size-mode iron）=====
# 铁律1「永远留一半现金」：震荡 20%，趋势 80%，任何情况都不到 100%（最坏也留 20% 现金）。
# 铁律「浮盈加仓，浮亏不加仓」：趋势里若持仓浮盈 → 维持 80%（允许再平衡往上推）；
#                              若浮亏 → 上限压到 40%，不给亏损单加码。
IRON_KER = 0.30         # KER 到此值才算「趋势明确」
IRON_LOW = 0.20         # 震荡市目标仓位
IRON_HIGH = 0.80        # 趋势市目标仓位
IRON_LOSS_CAP = 0.40    # 浮亏时的仓位上限
# ===== 铁律3 止损 =====
STOP_LOSS = 0.03        # 收盘价跌破「成本价 × (1 − 3%)」→ 无条件清仓
SPIKE_PCT = 0.21        # 单日涨跌超过 21%（A 股涨跌停上限）视为可疑跳变
REVERSAL_RATIO = 0.50   # 次日反向回吞超过一半 → 判定为脏数据毛刺（否则可能是真跳空/停复牌）
RET_TOLERANCE = 0.05   # 毛刺修复后价格需回到跳变前水平的 5% 以内，才算「错录」
REBALANCE_MIN = 0.05   # --rebalance：目标权重与当前权重差小于 5% 就不动（省摩擦成本）

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TIMEOUT = 90
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "sentiment_cache.json")

# 重大利空关键词（召回优先：宁可多报，让 LLM 复核去砍误报）
HARD_NEGATIVE_KEYWORDS = [
    "减持", "清仓式减持", "司法拍卖", "处罚", "罚款", "行政处罚", "立案", "立案调查",
    "被调查", "留置", "逮捕", "业绩暴雷", "业绩预亏", "预亏", "首亏", "商誉减值",
    "重大诉讼", "诉讼", "仲裁", "退市风险", "退市", "风险警示", "*ST", "被冻结",
    "股份冻结", "债务违约", "逾期", "监管函", "问询函", "警示函", "通报批评",
    "公开谴责", "涉嫌", "违法", "违规", "财务造假", "审计意见", "非标",
]
# 出现这些词时，上面的关键词大概率是"辟谣/澄清"，不算利空
NEGATION_WORDS = ["澄清", "否认", "辟谣", "不实", "不存在", "未发生"]

# 情绪打分的正向 / 负向词典：只在「没有 Ollama」或「--sentiment-source heuristic」时使用
POSITIVE_WORDS = [
    "增持", "回购", "中标", "预增", "扭亏", "超预期", "创新高", "涨停", "利好",
    "增长", "上调", "买入", "推荐", "受益", "突破", "合作", "签约", "获批",
    "分红", "派息", "龙头", "放量", "净流入", "评级", "看好",
]
NEGATIVE_WORDS = [
    "减持", "亏损", "预亏", "下滑", "处罚", "立案", "诉讼", "退市", "跌停",
    "下调", "卖出", "利空", "减持计划", "解禁", "风险", "警示", "问询",
    "违约", "冻结", "商誉减值", "爆雷", "低于预期", "净流出",
]


# ==============================================================================
# 小工具
# ==============================================================================
def log(msg: str = "") -> None:
    """诊断信息一律走 stderr，保证 stdout 只有 JSON。"""
    print(msg, file=sys.stderr, flush=True)


def _round(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def _to_code6(symbol: str) -> str:
    """把 'sz000001' / '000001.SZ' 之类统一成 6 位数字代码。"""
    s = str(symbol).strip().lower()
    m = re.search(r"(\d{6})", s)
    if not m:
        raise ValueError(f"看不懂的股票代码：{symbol}")
    return m.group(1)


def _to_sina_symbol(code6: str) -> str:
    """新浪接口要交易所前缀：6→sh，0/3→sz，4/8→bj。"""
    if code6.startswith("6"):
        return "sh" + code6
    if code6.startswith(("0", "3")):
        return "sz" + code6
    if code6.startswith(("4", "8")):
        return "bj" + code6
    return "sz" + code6


def _with_proxy_fallback(fn, label: str):
    """
    先正常请求；失败后清掉系统代理再试一次。

    典型场景：本机装过 Clash/V2Ray，代理软件退了但 Windows 系统代理还指着 127.0.0.1:7890，
    requests 直接抛 ProxyError，看起来像"接口挂了"。
    """
    try:
        return fn()
    except Exception as first_exc:
        keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy")
        backup = {k: os.environ.get(k) for k in keys}
        try:
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                      "http_proxy", "https_proxy", "all_proxy"):
                os.environ.pop(k, None)
            os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
            log(f"      [warn] {label} 首次请求失败，绕过系统代理重试...")
            return fn()
        except Exception:
            raise first_exc
        finally:
            for k, v in backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ==============================================================================
# 一、行情数据
# ==============================================================================
def _norm_eastmoney(df: pd.DataFrame) -> pd.DataFrame:
    """东财（stock_zh_a_hist）的列名是中文。"""
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "open": pd.to_numeric(df["开盘"], errors="coerce"),
        "close": pd.to_numeric(df["收盘"], errors="coerce"),
        "high": pd.to_numeric(df["最高"], errors="coerce"),
        "low": pd.to_numeric(df["最低"], errors="coerce"),
        "volume": pd.to_numeric(df["成交量"], errors="coerce"),
    })
    return out


def _norm_sina(df: pd.DataFrame) -> pd.DataFrame:
    """新浪（stock_zh_a_daily）的列名是英文，volume 单位是「股」，这里换算成「手」。"""
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce") / 100.0,
    })
    return out


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    行情清洗：把数据源的"脏东西"处理掉，并在 report 里如实记账。

    为什么必须清洗：akshare 背后是各家网页接口，拼接、重复、停牌日占位行、
    未复权残留都很常见。脏数据进 MACD/均线，出来的信号再漂亮也是假的。

    清洗顺序 = 先"去假"（重复/缺失/占位），再"纠错"（OHLC 自相矛盾），最后"标记"（异常跳空）。
    最后一步刻意不改数据：除权跳空是真实发生的价格断层，擅自平滑会把趋势吃掉，
    所以只记录、交给下游自行决定。
    """
    report = {
        "input_rows": int(len(raw)),
        "dropped_duplicates": 0,
        "dropped_invalid_close": 0,
        "dropped_placeholder_rows": 0,
        "ohlc_fixed": 0,
        "spikes_fixed": 0,
        "suspect_adjustments": 0,
        "gaps_over_10d": 0,
        "output_rows": 0,
        "warnings": [],
    }
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"]), report

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1) 同一天多条：保留最后一条（后写的通常是修正后的）
    before = len(df)
    df = df.drop_duplicates(subset=["date"], keep="last")
    report["dropped_duplicates"] = before - len(df)

    # 2) 按日期升序（后面的 rolling/差分全依赖这个顺序）
    df = df.sort_values("date").reset_index(drop=True)

    # 3) 数值化 + close 非法行剔除
    for col in ("open", "close", "high", "low", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    before = len(df)
    df = df[df["close"].notna() & (df["close"] > 0)]
    report["dropped_invalid_close"] = before - len(df)
    if df.empty:
        return df.reset_index(drop=True), report

    # 4) 其余 OHLC 缺失用 close 补；成交量缺失按 0 处理（后面会被当成可疑行）
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    # 5) OHLC 自相矛盾修正：high 必须 >= max(open, close)，low 必须 <= min(open, close)
    hi = df[["open", "close", "high"]].max(axis=1)
    lo = df[["open", "close", "low"]].min(axis=1)
    bad_ohlc = int(((df["high"] < hi - 1e-9) | (df["low"] > lo + 1e-9)).sum())
    df["high"], df["low"] = hi, lo
    report["ohlc_fixed"] = bad_ohlc

    # 6) 占位行：零成交量 + 开高低收全相同 → 数据源伪造的"停牌行"，会污染均线
    before = len(df)
    placeholder = (df["volume"] <= 0) & (df["high"] == df["low"]) & (df["open"] == df["close"])
    df = df[~placeholder]
    report["dropped_placeholder_rows"] = before - len(df)
    df = df.reset_index(drop=True)

    # 7) 孤立毛刺修复：单日涨跌超 21%，次日**反向回吞一半以上**，并且价格回到跳变前的水平
    #    典型脏数据形态：某个价被错录（小数点错位/单位错/串行），第二天又回到原趋势。
    #    注意：不能写成「前后两天都平静」——真毛刺的次日必然大幅反向，那样定义永远抓不到。
    #    也不能只看反向：连续同向跳空（除权断点）也会反向回来，所以先用「连续同向」把它们排掉。
    if len(df) >= 2:
        ret = df["close"].pct_change()
        big = ret.abs() > SPIKE_PCT
        same_dir = np.sign(ret) == np.sign(ret.shift(1, fill_value=0.0))
        cluster = big & big.shift(1, fill_value=False) & same_dir
        report["suspect_adjustments"] = int(cluster.sum())

    if len(df) >= 3:
        nxt = ret.shift(-1)
        reversal = (np.sign(nxt) == -np.sign(ret)) & (nxt.abs() > REVERSAL_RATIO * ret.abs())
        back_to_base = ((df["close"].shift(-1) - df["close"].shift(1)).abs()
                        <= RET_TOLERANCE * df["close"].shift(1))
        candidate = (ret.abs() > SPIKE_PCT) & reversal.fillna(False) & back_to_base.fillna(False)
        isolated = candidate & ~cluster.fillna(False)
        idx = df.index[isolated]
        for i in idx:
            if 0 < i < len(df):
                prev_close = float(df.at[i - 1, "close"])
                for col in ("open", "close", "high", "low"):
                    df.at[i, col] = prev_close
        report["spikes_fixed"] = len(idx)

    # 8)（上面已统计）连续大跳空多半是没复权干净的除权日：只标记，不动数据

    # 9) 交易日间隔检查：>10 天大概率是长假，也可能是长期停牌
    if len(df) >= 2:
        gaps = df["date"].diff().dt.days.fillna(0)
        report["gaps_over_10d"] = int((gaps > 10).sum())

    report["output_rows"] = int(len(df))
    if report["spikes_fixed"]:
        report["warnings"].append(f"修复 {report['spikes_fixed']} 个孤立毛刺（>21% 单日跳变）")
    if report["suspect_adjustments"]:
        report["warnings"].append(
            f"发现 {report['suspect_adjustments']} 处连续跳空，疑似未复权除权，已保留原值")
    if report["gaps_over_10d"]:
        report["warnings"].append(f"{report['gaps_over_10d']} 处交易日间隔超过 10 天（长假或停牌）")
    if report["output_rows"] < MIN_BARS:
        report["warnings"].append(f"清洗后仅 {report['output_rows']} 根 K 线，不足 {MIN_BARS}，指标不可靠")

    return df.reset_index(drop=True), report


def size_from_ker(ker: float | None, mode: str = SIZE_MODE, lo: float = KER_LO,
                  hi: float = KER_HI, max_weight: float = MAX_WEIGHT, *,
                  holding: bool = False, cost_price: float | None = None,
                  close: float | None = None) -> tuple[float, str]:
    """
    KER → 目标仓位（0~1）。返回 (weight, 说明文字)。

    注意：这里**不是**在判"能不能买"，而是在决定"买多少"。
    KER 无效（null）时按最低档处理：权重 0。

    iron 模式（三铁律）：
      KER <  0.30（震荡）        → 20%
      KER >= 0.30（趋势）且浮盈  → 80%（浮盈加仓，允许再平衡推上去）
      KER >= 0.30（趋势）且浮亏  → 40%（不给亏损单加码）
    任何情况下都受 max_weight 约束，默认 1.0，但铁律模式内部最高只到 80%。
    """
    if ker is None:
        return 0.0, "KER14 无效（分母为 0 或历史不足）→ 目标仓位 0"

    if mode == "none":
        return float(max_weight), f"固定仓位模式（--size-mode none）→ 目标仓位 {max_weight:.0%}"

    if mode == "iron":
        if ker < IRON_KER:
            w = IRON_LOW
            return min(w, max_weight), (
                f"铁律仓位：KER {ker:.3f} < {IRON_KER:.2f}（震荡市）→ "
                f"只投 {min(w, max_weight):.0%}，其余现金待命")
        w = IRON_HIGH
        note = f"铁律仓位：KER {ker:.3f} ≥ {IRON_KER:.2f}（趋势明确）→ {IRON_HIGH:.0%}"
        if holding and cost_price and close is not None:
            if close > cost_price:
                note += f"；浮盈（{close:.2f} > 成本 {cost_price:.2f}）→ 维持 {IRON_HIGH:.0%} 并允许加仓"
            else:
                w = IRON_LOSS_CAP
                note += f"；浮亏（{close:.2f} ≤ 成本 {cost_price:.2f}）→ 不加仓，上限压到 {IRON_LOSS_CAP:.0%}"
        return min(w, max_weight), note

    if mode == "tier":
        for low, high, weight in TIER_TABLE:
            if low <= ker < high:
                w = min(weight, max_weight)
                return float(w), f"分档模式：KER {ker:.3f} 落在 [{low:.2f}, {high:.2f}) → 目标仓位 {w:.0%}"
        return float(max_weight), f"分档模式：KER {ker:.3f} 超出表范围 → 目标仓位 {max_weight:.0%}"

    # 默认 linear
    if hi <= lo:
        return float(max_weight), f"线性模式参数异常（lo={lo} hi={hi}）→ 目标仓位 {max_weight:.0%}"
    ratio = (ker - lo) / (hi - lo)
    weight = float(np.clip(ratio, 0.0, 1.0)) * float(max_weight)
    if weight <= 0:
        return 0.0, f"线性模式：KER {ker:.3f} ≤ {lo:.2f} → 目标仓位 0（趋势太弱，不值得开仓）"
    return weight, (f"线性模式：KER {ker:.3f} 在 [{lo:.2f}, {hi:.2f}] 内 → "
                    f"目标仓位 {weight:.1%}（满仓系数 {max_weight:.0%}）")


def _cache_path(code6: str, start: str, end: str, adjust: str = "qfq") -> str:
    # 前复权/后复权必须分开缓存：同一区间两种口径的价格能差十倍
    suffix = "" if adjust == "qfq" else f"_{adjust}"
    return os.path.join(HERE, f"data_{code6}_{start}_{end}{suffix}.csv")


def fetch_prices(code6: str, end_date: pd.Timestamp, lookback_days: int = 220,
                 adjust: str = "qfq") -> pd.DataFrame | None:
    """
    抓「截止 end_date」的日线。优先缓存，其次东财，最后新浪。

    adjust：qfq 前复权（默认）/ hfq 后复权 / "" 不复权。
    深度历史回测（比如 2015）必须用 hfq：东财的前复权是「减法」调整，往前追溯十年，
    高分红股的复权价会被减成负数（招商银行 2015 年的 qfq 价就是 -5 块），
    后复权是「乘法」调整，永远不会出现负价。
    返回列：date / open / close / high / low / volume（升序）。
    """
    start_date = end_date - pd.Timedelta(days=lookback_days)
    start_s, end_s = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
    cache = _cache_path(code6, start_s, end_s, adjust)

    if os.path.exists(cache):
        try:
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if not df.empty and df["date"].max() >= end_date:
                log(f"      行情来源：本地缓存 {os.path.basename(cache)}（{len(df)} 根 K 线）")
                return df
        except Exception:
            pass

    import akshare as ak

    def _eastmoney_raw():
        return ak.stock_zh_a_hist(symbol=code6, period="daily",
                                  start_date=start_s, end_date=end_s, adjust=adjust)

    def _sina_raw():
        return ak.stock_zh_a_daily(symbol=_to_sina_symbol(code6), start_date=start_s,
                                   end_date=end_s, adjust=adjust or "")

    for name, fetch_raw, norm in (("东方财富", _eastmoney_raw, _norm_eastmoney),
                                  ("新浪", _sina_raw, _norm_sina)):
        try:
            raw = _with_proxy_fallback(fetch_raw, name)
            df = norm(raw)
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if df.empty:
                continue
            df.to_csv(cache, index=False, encoding="utf-8")
            log(f"      行情来源：{name}（{len(df)} 根 K 线，已缓存 {os.path.basename(cache)}）")
            return df
        except Exception as exc:  # 网络/限流/接口变更都算
            log(f"      [warn] {name} 取数失败：{type(exc).__name__}: {exc}")
            continue

    # 网络全失败时，退一步：找本地任何一份包含 end_date 的缓存
    for fn_name in sorted(os.listdir(HERE), reverse=True):
        if not (fn_name.startswith(f"data_{code6}_") and fn_name.endswith(".csv")):
            continue
        try:
            df = pd.read_csv(os.path.join(HERE, fn_name))
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if not df.empty and df["date"].max() >= end_date:
                log(f"      [warn] 联网失败，改用历史缓存 {fn_name}（{len(df)} 根 K 线）")
                return df
        except Exception:
            continue
    return None


# ==============================================================================
# 二、技术指标（只用 T 及之前的数据）
# ==============================================================================
def compute_tech(df: pd.DataFrame, trade_date: pd.Timestamp) -> dict:
    """
    注意：df 里可能有 T 之后的行情（比如用整年数据跑历史某天），
    这里用 loc[:trade_date] 物理切断，保证"未来"的行不参与任何计算。
    """
    sub = df.loc[df["date"] <= trade_date].reset_index(drop=True)
    res: dict[str, Any] = {
        "bars": len(sub),
        "has_price": False,
        "close": None, "ma5": None, "ma20": None,
        "ma5_prev": None, "ma20_prev": None,
        "golden_cross": False, "trend_above": False,
        "ker14": None, "ker_valid": False,
    }
    if sub.empty or sub["date"].iloc[-1] != trade_date:
        return res  # T 不是交易日 / 当天没有行情

    close = sub["close"].astype(float)
    ma_fast = close.rolling(MA_FAST).mean()
    ma_slow = close.rolling(MA_SLOW).mean()

    res["has_price"] = True
    res["close"] = float(close.iloc[-1])
    if len(sub) >= MA_FAST:
        res["ma5"] = float(ma_fast.iloc[-1])
        if len(sub) >= MA_FAST + 1:
            res["ma5_prev"] = float(ma_fast.iloc[-2])
    if len(sub) >= MA_SLOW:
        res["ma20"] = float(ma_slow.iloc[-1])
        if len(sub) >= MA_SLOW + 1:
            res["ma20_prev"] = float(ma_slow.iloc[-2])

    if None not in (res["ma5"], res["ma20"], res["ma5_prev"], res["ma20_prev"]):
        res["trend_above"] = res["ma5"] > res["ma20"]
        res["golden_cross"] = (res["ma5"] > res["ma20"]) and (res["ma5_prev"] <= res["ma20_prev"])

    # KER14：分子 |C_T - C_{T-14}|，分母是最近 14 个日收益的绝对值之和
    if len(sub) >= KER_N + 1:
        c = close
        num = abs(float(c.iloc[-1]) - float(c.iloc[-1 - KER_N]))
        den = float(c.diff().abs().iloc[-KER_N:].sum())
        if den > 0:
            res["ker14"] = num / den
            res["ker_valid"] = True
    return res


# ==============================================================================
# 三、新闻 / 公告 + 情绪打分
# ==============================================================================
@dataclass
class NewsItem:
    ts: datetime
    title: str
    text: str = ""
    src: str = "news"

    @property
    def key(self) -> str:
        raw = f"{self.ts:%Y-%m-%d %H:%M}|{self.title}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def news_window(trade_date: pd.Timestamp) -> tuple[datetime, datetime]:
    """情绪窗口：(T-1 15:00, T 15:00]，严格不含 T 日收盘之后的信息。"""
    end = trade_date.to_pydatetime().replace(hour=CLOSE_HOUR, minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=NEWS_WINDOW_HOURS)
    return start, end


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _load_news_file(path: str) -> list[NewsItem]:
    """离线新闻：一行一个 JSON，字段 ts/title/text 可选 src。"""
    items: list[NewsItem] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(obj.get("ts") or obj.get("time") or obj.get("date"))
            title = str(obj.get("title") or "").strip()
            if ts and title:
                items.append(NewsItem(ts=ts, title=title,
                                      text=str(obj.get("text") or ""),
                                      src=str(obj.get("src") or "file")))
    return items


def fetch_live_news(code6: str, win_start: datetime, win_end: datetime,
                    with_notices: bool = True, with_reports: bool = False) -> list[NewsItem]:
    """
    拉「个股新闻（+ 公告 / 研报）」。任何一路失败都不致命，返回拿到的部分。

    诚实提示：东方财富的个股新闻接口返回的是**最近**一批新闻，历史日期（比如 2023 年）
    基本拿不到。这时候情绪会如实为 null → 输出"数据不足/观望"，而不是编一个分数。
    """
    import akshare as ak

    items: list[NewsItem] = []

    # 1) 个股新闻
    try:
        raw = _with_proxy_fallback(lambda: ak.stock_news_em(symbol=code6), "个股新闻")
        col_ts = next((c for c in ("发布时间", "时间") if c in raw.columns), None)
        col_title = next((c for c in ("新闻标题", "标题") if c in raw.columns), None)
        col_text = next((c for c in ("新闻内容", "内容") if c in raw.columns), None)
        if col_ts and col_title:
            for _, row in raw.iterrows():
                ts = _parse_ts(row[col_ts])
                if ts:
                    items.append(NewsItem(ts=ts, title=str(row[col_title]).strip(),
                                          text=str(row[col_text] or "")[:600] if col_text else "",
                                          src="news"))
        log(f"      新闻接口返回 {len(raw)} 条（全部，未过滤）")
    except Exception as exc:
        log(f"      [warn] 个股新闻接口不可用：{type(exc).__name__}: {exc}")

    # 2) 公告（按日查询，窗口最多横跨 2 个自然日）
    if with_notices:
        days = sorted({d.date() for d in (win_start, win_end)})
        for day in days:
            ds = day.strftime("%Y%m%d")
            try:
                raw = _with_proxy_fallback(
                    lambda: ak.stock_notice_report(symbol="全部", date=ds), f"公告{ds}")
                if raw is None or raw.empty:
                    continue
                col_code = "代码" if "代码" in raw.columns else None
                col_title = next((c for c in ("公告标题", "标题") if c in raw.columns), None)
                col_date = next((c for c in ("公告日期", "日期") if c in raw.columns), None)
                if not (col_code and col_title):
                    continue
                sub = raw[raw[col_code].astype(str).str.zfill(6) == code6]
                for _, row in sub.iterrows():
                    ts = _parse_ts(row[col_date]) if col_date else None
                    ts = ts or datetime.combine(day, datetime.min.time()).replace(hour=8)
                    items.append(NewsItem(ts=ts, title=str(row[col_title]).strip(), src="notice"))
            except Exception as exc:
                log(f"      [warn] 公告接口 {ds} 不可用：{type(exc).__name__}: {exc}")

    # 3) 研报（可选，接口字段不稳定，失败就当没有）
    if with_reports:
        try:
            raw = _with_proxy_fallback(lambda: ak.stock_research_report_em(symbol=code6), "研报")
            col_ts = next((c for c in ("日期", "发布时间") if c in raw.columns), None)
            col_title = next((c for c in ("报告名称", "标题") if c in raw.columns), None)
            if col_ts and col_title:
                for _, row in raw.iterrows():
                    ts = _parse_ts(row[col_ts])
                    if ts:
                        items.append(NewsItem(ts=ts, title=str(row[col_title]).strip(), src="report"))
        except Exception as exc:
            log(f"      [warn] 研报接口不可用：{type(exc).__name__}: {exc}")

    # 去重 + 时间窗过滤（这一步是无未来函数的关键）
    seen, kept = set(), []
    for it in sorted(items, key=lambda x: x.ts, reverse=True):
        if not (win_start < it.ts <= win_end):
            continue
        if it.key in seen:
            continue
        seen.add(it.key)
        kept.append(it)
    return kept


# ------------------------------ 情绪打分 --------------------------------------
_sent_cache: dict[str, float] | None = None


def _cache_load() -> dict[str, float]:
    global _sent_cache
    if _sent_cache is None:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as fh:
                _sent_cache = json.load(fh)
        except Exception:
            _sent_cache = {}
    return _sent_cache


def _cache_save() -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(_sent_cache or {}, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _ollama_alive() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _extract_number(text: str) -> float | None:
    """从模型输出里抠一个 [-1,1] 的数：先找 -1.0~1.0 的小数，再找整数。"""
    m = re.search(r"[-+]?\d*\.\d+", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    m = re.search(r"[-+]?\d+", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


SENT_PROMPT = (
    "你是资深A股金融分析师。下面是一条关于「{name}」的新闻/公告标题与摘要。\n"
    "请评估它对该股票未来 1-3 个交易日的**情绪影响**，给出一个数字：\n"
    " -1.0 = 极度利空（如立案调查、业绩暴雷、大额减持）\n"
    "  0.0 = 中性（无实质影响、例行公告、与公司关联很弱）\n"
    "+1.0 = 极度利好（如业绩大幅超预期、重大中标、大额回购）\n"
    "只输出这个数字本身（保留一位小数），不要输出任何解释或标点。\n\n"
    "标题：{title}\n摘要：{text}\n"
)


def score_ollama(items: Iterable[NewsItem], model: str, name: str) -> tuple[dict[str, float], int]:
    """逐条打分，带磁盘缓存（同一标题+时间不重复问模型，保证可复现）。"""
    cache = _cache_load()
    scores: dict[str, float] = {}
    hits = 0
    todo = [it for it in items if it.key not in cache]
    hits = len(list(items)) - len(todo)
    for idx, it in enumerate(todo, 1):
        prompt = SENT_PROMPT.format(name=name, title=it.title, text=(it.text or "（无摘要）")[:500])
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "top_p": 1, "seed": 42, "num_predict": 8},
        }).encode("utf-8")
        try:
            req = urllib.request.Request(OLLAMA_URL, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
            val = _extract_number(str(body.get("response", "")))
            if val is None:
                log(f"      [warn] 第{idx}条模型没给出数字：{body.get('response')!r}")
                continue
            val = max(-1.0, min(1.0, val))
            cache[it.key] = round(val, 3)
            scores[it.key] = round(val, 3)
            log(f"      [{idx}/{len(todo)}] {val:+.1f}  {it.title[:40]}")
        except Exception as exc:
            log(f"      [warn] 第{idx}条打分失败：{type(exc).__name__}: {exc}")
            break
    for it in items:
        if it.key in cache and it.key not in scores:
            scores[it.key] = cache[it.key]
    _cache_save()
    return scores, hits


def score_heuristic(items: Iterable[NewsItem]) -> dict[str, float]:
    """词典兜底口径：正负词计数差 / 总数，裁剪到 [-1,1]。"""
    scores: dict[str, float] = {}
    for it in items:
        text = f"{it.title}{it.text}"
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)
        if pos == 0 and neg == 0:
            val = 0.0
        else:
            val = (pos - neg) / (pos + neg)
        scores[it.key] = round(max(-1.0, min(1.0, val)), 3)
    return scores


def detect_hard_negative(items: Iterable[NewsItem]) -> list[str]:
    """关键词硬否决。返回命中的「标题 + 关键词」，供 reasons 说明。"""
    hits: list[str] = []
    for it in items:
        whole = f"{it.title}{it.text}"
        if any(neg in whole for neg in NEGATION_WORDS):
            continue  # 澄清/辟谣类，不算
        for kw in HARD_NEGATIVE_KEYWORDS:
            if kw in whole:
                hits.append(f"「{it.title[:40]}」含关键词「{kw}」")
                break
    return hits


def llm_veto(items: list[NewsItem], model: str) -> list[str]:
    """让模型二次复核：是否构成"重大利空"。返回命中说明。"""
    if not items:
        return []
    bullets = "\n".join(f"- {it.title[:120]}" for it in items[:12])
    prompt = (
        "下面是某A股公司在一个交易日内的新闻/公告标题。请判断其中是否有**重大利空**：\n"
        "减持、处罚、立案调查、业绩暴雷、重大诉讼、退市风险六类。\n"
        "只输出 JSON：{\"veto\": true/false, \"reason\": \"一句话，命中哪条标题的哪类风险\"}\n\n"
        f"{bullets}\n"
    )
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 96},
    }).encode("utf-8")
    try:
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        raw = str(body.get("response", ""))
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group()) if m else {}
        if obj.get("veto") is True:
            return [f"LLM 复核判定重大利空：{obj.get('reason') or '（未给出理由）'}"]
    except Exception as exc:
        log(f"      [warn] LLM 复核失败（按未命中处理）：{type(exc).__name__}: {exc}")
    return []


# ==============================================================================
# 四、决策（严格按优先级串起来）
# ==============================================================================
def decide(*, symbol: str, trade_date: pd.Timestamp, holding: bool,
           tech: dict, items: list[NewsItem], sentiment: float | None,
           sentiment_source: str, hard_neg: list[str],
           sent_min: float, target_weight: float, size_note: str,
           data_report: dict, rebalance: bool = False,
           current_weight: float | None = None,
           rebalance_min: float = REBALANCE_MIN,
           cost_price: float | None = None,
           stop_loss: float = STOP_LOSS) -> dict:
    """
    判定顺序：风控 > 重大利空 > 技术(趋势+情绪) > 仓位(KER) > 再平衡(可选)。
    返回 JSON：需求文档里的 12 个字段 + 追加的 position_size / data_quality / rebalance_delta
    （都在末尾，老的解析器按 key 取值不受影响）。

    --rebalance 打开且给了 --current-weight 时，持仓状态下会按「当前权重 vs KER 目标权重」
    的差额给出 加仓/减仓 动作；差额小于 rebalance_min 则不动（避免手续费吃掉优势）。
    """
    reasons: list[str] = []
    risk_trigger = False
    hard_negative = bool(hard_neg)
    weight = float(max(0.0, min(1.0, target_weight)))

    # ---------- 0. 数据可用性 ----------
    if not tech["has_price"]:
        reasons.append(f"{trade_date:%Y-%m-%d} 没有行情（非交易日或数据未更新），按规则不交易")
        reasons.append("数据不足 → 观望")
        return _pack(symbol, trade_date, tech, sentiment, hard_negative, False,
                     "观望", "不操作", 0.0, reasons, data_report)

    if data_report.get("output_rows", 0) < MIN_BARS:
        reasons.append(f"清洗后只有 {data_report.get('output_rows', 0)} 根 K 线（< {MIN_BARS}）"
                       "，均线/KER 不可靠 → 数据不足，观望")
        return _pack(symbol, trade_date, tech, sentiment, hard_negative, False,
                     "观望", "不操作", 0.0, reasons, data_report)

    close, ma20 = tech["close"], tech["ma20"]
    ma5, golden = tech["ma5"], tech["golden_cross"]
    ker = tech["ker14"]

    # ---------- 1. 技术面陈述 ----------
    reasons.append(
        f"收盘 {close:.2f}，MA5={_fmt(ma5)}，MA20={_fmt(ma20)}"
        f"（{'MA5 在 MA20 上方' if tech['trend_above'] else 'MA5 在 MA20 下方'}）"
    )
    if golden:
        reasons.append(
            f"金叉成立：MA5_T {ma5:.3f} > MA20_T {ma20:.3f}，且 MA5_T-1 "
            f"{tech['ma5_prev']:.3f} <= MA20_T-1 {tech['ma20_prev']:.3f}"
        )
    elif tech["trend_above"]:
        reasons.append("趋势延续：MA5 已在 MA20 上方，但不是当日新金叉 → 不构成新开仓信号")
    else:
        reasons.append("无金叉：MA5 未上穿 MA20")

    # ---------- 2. 仓位（KER 只在这里起作用） ----------
    reasons.append(f"仓位测算：{size_note}")

    # ---------- 3. 情绪 ----------
    if sentiment is None:
        reasons.append("情绪分不可用（窗口内无新闻，或模型/接口不可用）→ 情绪条件不通过")
    else:
        w_start, w_end = news_window(trade_date)
        reasons.append(
            f"情绪分 S = {sentiment:+.3f}（口径：{sentiment_source}，窗口 "
            f"{w_start:%m-%d %H:%M} ~ {w_end:%m-%d %H:%M} 左开右闭，共 {len(items)} 条）→ "
            f"{'配合' if sentiment > sent_min else '不配合'}（阈值 {sent_min}）"
        )
    if hard_negative:
        reasons.append("重大利空硬否决：" + "；".join(hard_neg))
        reasons.append("情绪条件被一票否决（即使分数为正也不算配合）")

    # ---------- 4. 风控兜底（最高优先级，先判） ----------
    # 4a. 铁律止损：跌破成本价 3% → 无条件清仓（不看均线、不看情绪）
    if holding and cost_price and stop_loss > 0:
        stop_line = float(cost_price) * (1.0 - float(stop_loss))
        if close <= stop_line:
            reasons.append(
                f"【止损】持仓成本 {cost_price:.2f}，止损线 {stop_line:.2f}"
                f"（成本 −{stop_loss:.0%}），收盘 {close:.2f} 已跌破 → 无条件清仓 100%，"
                "不许扛单"
            )
            return _pack(symbol, trade_date, tech, sentiment, hard_negative, True,
                         "清仓", "清仓", 0.0, reasons, data_report)

    # 4b. 趋势风控：收盘跌破 MA20 → 清仓
    if holding and ma20 is not None and close < ma20:
        risk_trigger = True
        reasons.append(
            f"【风控】持仓中且收盘 {close:.2f} < MA20 {ma20:.3f} → 强制清仓 100%，"
            "其余条件一律不再考虑"
        )
        return _pack(symbol, trade_date, tech, sentiment, True, True,
                     "清仓", "清仓", 0.0, reasons, data_report)

    # ---------- 5. 持仓：不破位就持有（可选按 KER 做再平衡） ----------
    if holding:
        reasons.append("持仓中且未跌破 MA20 → 继续持有，不触发清仓")

        # 再平衡：只有显式 --rebalance 且给了当前权重才动，否则只给建议
        delta: float | None = None
        action = "持有"
        if not rebalance:
            if weight > 0:
                reasons.append(f"按 KER 测算的目标仓位是 {weight:.1%}；"
                               "未开 --rebalance，仅作建议，不产生调仓动作")
        elif current_weight is None:
            reasons.append("[warn] 开了 --rebalance 但没给 --current-weight → 无法算差额，"
                           "本次不做再平衡")
        else:
            cur = float(max(0.0, min(1.0, current_weight)))
            delta = weight - cur
            reasons.append(f"再平衡：当前权重 {cur:.1%}，KER 目标权重 {weight:.1%}，"
                           f"差额 {delta:+.1%}（最小调整幅度 {rebalance_min:.0%}）")
            if abs(delta) < rebalance_min:
                reasons.append("差额小于阈值 → 调仓收益盖不住摩擦成本，本次不动")
            elif delta > 0 and cost_price and float(close) <= float(cost_price):
                reasons.append(
                    f"目标是加仓，但当前浮亏（收盘 {close:.2f} ≤ 成本 {cost_price:.2f}）→ "
                    "铁律：不给亏损单加码，本次不动")
                delta = 0.0
            elif delta > 0:
                action = "加仓"
                reasons.append(f"低于目标 → 建议加仓 {delta:.1%}（加仓后权重 {weight:.1%}）")
            else:
                action = "减仓"
                reasons.append(f"高于目标 → 建议减仓 {-delta:.1%}（减仓后权重 {weight:.1%}）")

        return _pack(symbol, trade_date, tech, sentiment, False, False,
                     "持有", action, weight, reasons, data_report, delta)

    # ---------- 6. 空仓：趋势 + 情绪 + 仓位>0 三者齐备才开仓 ----------
    trend_pass = bool(golden)
    sent_pass = bool(sentiment is not None and sentiment > sent_min and not hard_negative)
    size_pass = weight > 0

    reasons.append(f"开仓条件：trend={trend_pass} / sentiment={sent_pass} / 目标仓位={weight:.1%}")

    if trend_pass and sent_pass and size_pass:
        reasons.append(f"全部满足 → 买入信号，按 {weight:.1%} 仓位开仓（KER 只决定仓位，不设门槛）")
        return _pack(symbol, trade_date, tech, sentiment, False, False,
                     "买入", "开仓", weight, reasons, data_report)

    fail = []
    if not trend_pass:
        fail.append("趋势（无当日金叉）")
    if not sent_pass:
        fail.append("情绪")
    if not size_pass:
        fail.append(f"仓位（KER { _fmt(ker) } 对应的目标仓位为 0）")
    reasons.append(f"未同时满足（缺：{'、'.join(fail)}）→ 观望，不操作")
    return _pack(symbol, trade_date, tech, sentiment, False, False,
                 "观望", "不操作", 0.0, reasons, data_report)


def _fmt(x: Any) -> str:
    return "null" if x is None else f"{x:.3f}"


def _pack(symbol: str, trade_date: pd.Timestamp, tech: dict, sentiment: float | None,
          hard_negative: bool, risk_trigger: bool, signal: str, action: str,
          position_size: float, reasons: list[str], data_report: dict | None = None,
          rebalance_delta: float | None = None) -> dict:
    """输出 JSON：需求文档的字段在前，新增字段追加在后。"""
    return {
        "symbol": symbol,
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "ma5": _round(tech["ma5"]),
        "ma20": _round(tech["ma20"]),
        "golden_cross": bool(tech["golden_cross"]),
        "ker14": _round(tech["ker14"], 4),
        "sentiment_score": _round(sentiment, 3),
        "hard_negative_news": bool(hard_negative),
        "risk_trigger": bool(risk_trigger),
        "signal": signal,
        "position_action": action,
        "reasons": reasons,
        "position_size": _round(position_size, 4),
        "data_quality": data_report or {},
        "rebalance_delta": _round(rebalance_delta, 4),
    }


# ==============================================================================
# 五、未来函数自检
# ==============================================================================
def audit_no_lookahead(df: pd.DataFrame, trade_date: pd.Timestamp) -> bool:
    """
    把 T 之后的所有收盘价改成随机数（+80%），再算一次技术指标与决策主体。
    只要两者一致，就说明 T 的信号没有偷偷用到未来数据。
    """
    base = compute_tech(df, trade_date)

    tampered = df.copy()
    future_mask = tampered["date"] > trade_date
    n_future = int(future_mask.sum())
    if n_future == 0:
        log("      [audit] T 之后没有数据，本次无法做篡改检验（视为通过）")
        return True
    rng = np.random.default_rng(7)
    tampered.loc[future_mask, "close"] = tampered.loc[future_mask, "close"] * rng.uniform(1.5, 2.0, n_future)
    alt = compute_tech(tampered, trade_date)

    keys = ("ma5", "ma20", "ma5_prev", "ma20_prev", "golden_cross", "ker14", "close")
    diffs = [k for k in keys if (base[k] is None) != (alt[k] is None)
             or (base[k] is not None and abs(float(base[k]) - float(alt[k])) > 1e-9)]
    if diffs:
        log(f"      [audit] ✗ 发现未来函数：篡改 T 之后的 {n_future} 根 K 线后，{diffs} 发生变化！")
        return False
    log(f"      [audit] ✓ 通过：篡改 T 之后 {n_future} 根 K 线，T 日指标/信号完全不变")
    return True


def _load_wide_frame(code6: str, trade_date: pd.Timestamp) -> pd.DataFrame | None:
    """
    给 --audit 用的「宽行情」：找本地所有 data_<code>_*.csv，挑一份**延伸到 T 之后**、
    行数最多的，用来做篡改检验。找不到就返回 None（自检会退化说明无法检验）。
    """
    best: pd.DataFrame | None = None
    for fn_name in os.listdir(HERE):
        if not (fn_name.startswith(f"data_{code6}_") and fn_name.endswith(".csv")):
            continue
        try:
            df = pd.read_csv(os.path.join(HERE, fn_name))
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            continue
        if df.empty or df["date"].max() <= trade_date:
            continue
        if best is None or len(df) > len(best):
            best = df
    return best


# ==============================================================================
# 六、命令行入口
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="量化交易策略信号判定助手（只用 T 日及之前的数据，输出 JSON）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python signal_assistant.py --symbol 000001 --position flat",
    )
    p.add_argument("--symbol", default="000001", help="股票代码，默认 000001（平安银行）")
    p.add_argument("--date", default=None, help="交易日 T，YYYY-MM-DD；默认最近交易日")
    p.add_argument("--position", choices=["flat", "long"], default="flat",
                   help="当前持仓状态：flat=空仓（默认），long=持仓中")
    p.add_argument("--rebalance", action="store_true",
                   help="持仓时按「KER 目标权重 vs 当前权重」的差额给出 加仓/减仓 动作")
    p.add_argument("--current-weight", type=float, default=None,
                   help="当前持仓权重 0~1（配合 --rebalance 用；不给就只能给建议、不动作）")
    p.add_argument("--rebalance-min", type=float, default=REBALANCE_MIN,
                   help=f"再平衡最小调整幅度，默认 {REBALANCE_MIN:.0%}（小于此值不动）")
    p.add_argument("--ker-lo", type=float, default=KER_LO,
                   help=f"linear 模式：KER 到此值目标仓位为 0，默认 {KER_LO}")
    p.add_argument("--ker-hi", type=float, default=KER_HI,
                   help=f"linear 模式：KER 到此值目标仓位拉满，默认 {KER_HI}")
    p.add_argument("--max-weight", type=float, default=MAX_WEIGHT,
                   help=f"单票最大仓位系数（0~1），默认 {MAX_WEIGHT}")
    p.add_argument("--size-mode", choices=["linear", "tier", "none", "iron"], default=SIZE_MODE,
                   help="仓位模式：linear=按 KER 线性插值（默认）；tier=四档；iron=三铁律"
                        "（震荡 20%% / 趋势 80%% / 浮亏降到 40%%）；none=固定满仓")
    p.add_argument("--cost", type=float, default=None,
                   help="持仓成本价（配合 --position long 使用；用于止损与浮盈加仓判定）")
    p.add_argument("--stop-loss", type=float, default=STOP_LOSS,
                   help=f"止损线：收盘跌破成本价此比例即清仓，默认 {STOP_LOSS:.0%}（0=关闭）")
    p.add_argument("--sent-min", type=float, default=SENT_MIN, help=f"情绪阈值，默认 {SENT_MIN}")
    p.add_argument("--model", default="qwen2.5:7b", help="Ollama 模型名")
    p.add_argument("--sentiment-source", choices=["auto", "ollama", "heuristic", "none"],
                   default="auto", help="情绪来源：auto=有模型用模型、没有就用词典")
    p.add_argument("--sentiment-mode", choices=["score", "event"], default="score",
                   help="score = 让模型直接打 -1~1 分（旧口径）；event = LLM 只抽事件标签，alpha 用先验权重表算")
    p.add_argument("--news-file", default=None, help="离线新闻 JSONL（回测/调试用）")
    p.add_argument("--event-model", default="qwen2.5:7b", help="事件抽取用的本地模型")
    p.add_argument("--no-event-llm", action="store_true",
                   help="事件模式只用规则抽取（离线可复现，不依赖 Ollama）")
    p.add_argument("--no-notices", action="store_true", help="不查公告")
    p.add_argument("--with-reports", action="store_true", help="额外查研报（接口不稳，可选）")
    p.add_argument("--llm-veto", action="store_true", help="用 LLM 复核重大利空（默认只用关键词）")
    p.add_argument("--audit", action="store_true", help="未来函数自检（篡改 T 之后的数据）")
    p.add_argument("--out", default=None, help="把 JSON 另存到文件")
    p.add_argument("--quiet", action="store_true", help="不打印 stderr 诊断信息")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        global log
        log = lambda *a, **k: None  # noqa: E731

    t0 = time.time()
    code6 = _to_code6(args.symbol)
    log("=" * 68)
    log(f"  信号判定：{code6}  持仓={args.position}  仓位模式={args.size_mode}"
        f"（KER {args.ker_lo}~{args.ker_hi} → 0~{args.max_weight:.0%}）  情绪阈值={args.sent_min}")
    log("=" * 68)

    # ---- 1. 确定交易日 T ----
    try:
        want = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    except Exception as exc:
        log(f"[error] 日期格式不对：{exc}")
        return 2

    df = fetch_prices(code6, want)
    if df is None or df.empty:
        log("[error] 行情数据拿不到（网络不通且无本地缓存）")
        empty = _pack(code6, want, {"has_price": False, "ma5": None, "ma20": None,
                                    "golden_cross": False, "ker14": None},
                      None, False, False, "观望", "不操作", 0.0,
                      ["行情数据不可用：接口失败且无本地缓存 → 数据不足，不交易"], {})
        print(json.dumps(empty, ensure_ascii=False, indent=2))
        return 2

    # ---- 1.5 数据清洗（必须在任何指标之前） ----
    df, qreport = clean_prices(df)
    log(f"      清洗：{qreport['input_rows']} 行 → {qreport['output_rows']} 行 | "
        f"重复 {qreport['dropped_duplicates']} · 坏 close {qreport['dropped_invalid_close']} · "
        f"占位行 {qreport['dropped_placeholder_rows']} · OHLC 修正 {qreport['ohlc_fixed']} · "
        f"毛刺修复 {qreport['spikes_fixed']}")
    for warn_text in qreport.get("warnings", []):
        log(f"      [清洗] {warn_text}")
    if df.empty:
        log("[error] 清洗后没剩数据")
        return 2

    # 允许 T 是周末/节假日：自动回退到最近一个交易日，但必须明确告知
    trade_date = want
    if trade_date not in set(df["date"]):
        earlier = df.loc[df["date"] <= want, "date"]
        if earlier.empty:
            log(f"[error] 数据里没有 {want:%Y-%m-%d} 及之前的行情（区间太早）")
            return 2
        trade_date = earlier.max()
        log(f"      [note] {want:%Y-%m-%d} 不是交易日，回退到最近交易日 {trade_date:%Y-%m-%d}")
        # 数据陈旧检查：回退超过 10 个自然日，说明停牌或数据没更新
        if (want - trade_date).days > 10:
            log(f"      [warn] 最近行情距今 {want:%Y-%m-%d} 已 {(want - trade_date).days} 天，可能停牌")

    # ---- 2. 技术指标 ----
    tech = compute_tech(df, trade_date)
    log(f"      技术面：bars={tech['bars']} close={_fmt(tech['close'])} "
        f"ma5={_fmt(tech['ma5'])} ma20={_fmt(tech['ma20'])} "
        f"金叉={tech['golden_cross']} ker14={_fmt(tech['ker14'])}")

    # ---- 3. 未来函数自检 ----
    if args.audit:
        wide = _load_wide_frame(code6, trade_date)
        if wide is None:
            log("      [audit] 本地没有延伸到 T 之后的行情，改用联网拉宽区间...")
            wide = fetch_prices(code6, pd.Timestamp.today().normalize())
        if wide is not None and not wide.empty and wide["date"].max() > trade_date:
            wide, _ = clean_prices(wide)
        if wide is None or wide.empty or wide["date"].max() <= trade_date:
            log("      [audit] 拿不到 T 之后的数据，本次无法做篡改检验（跳过）")
        elif not audit_no_lookahead(wide, trade_date):
            print(json.dumps({"error": "audit_failed", "detail": "检测到未来函数"},
                             ensure_ascii=False))
            return 3

    # ---- 4. 新闻与情绪 ----
    win_start, win_end = news_window(trade_date)
    log(f"      情绪窗口：{win_start:%Y-%m-%d %H:%M} ~ {win_end:%Y-%m-%d %H:%M}（左开右闭）")

    items: list[NewsItem] = []
    if args.news_file:
        items = [it for it in _load_news_file(args.news_file) if win_start < it.ts <= win_end]
        log(f"      离线新闻文件命中窗口内 {len(items)} 条")
    elif args.sentiment_source != "none":
        items = fetch_live_news(code6, win_start, win_end,
                                with_notices=not args.no_notices,
                                with_reports=args.with_reports)
        log(f"      窗口内可用新闻/公告：{len(items)} 条")

    sentiment: float | None = None
    source_desc = "none"
    event_detail: dict | None = None
    if items and args.sentiment_source != "none":
        use_ollama = args.sentiment_source == "ollama" or (
            args.sentiment_source == "auto" and _ollama_alive())
        if args.sentiment_mode == "event":
            # ---- 事件化情绪：LLM 只抽标签，alpha 由先验权重表算出 ----
            import event_sentiment as es
            event_detail = es.analyze(
                items,
                use_llm=(use_ollama and not args.no_event_llm),
                model=args.event_model,
                now=win_end)
            sentiment = event_detail.get("alpha")
            source_desc = (f"event:{event_detail['mode']}（{event_detail['n_events']} 个事件，"
                           f"丢弃非法 {event_detail['dropped']}）")
            log(f"      事件：{event_detail['note']}")
            for ev in event_detail.get("events", [])[:5]:
                log(f"        · {ev['name']}（强度 {ev['intensity']}，贡献 {ev['contribution']:+.3f}）"
                    f" 依据「{ev['evidence']}」")
        elif use_ollama:
            scores, cached = score_ollama(items, args.model, code6)
            vals = [scores[it.key] for it in items if it.key in scores]
            if vals:
                sentiment = float(np.mean(vals))
                source_desc = f"ollama:{args.model}（{len(vals)} 条等权平均，{cached} 条命中缓存）"
            else:
                sentiment = None
                source_desc = "ollama 调用失败"
        elif args.sentiment_source in ("heuristic", "auto"):
            scores = score_heuristic(items)
            vals = [scores[it.key] for it in items]
            sentiment = float(np.mean(vals))
            source_desc = "heuristic 词典（正负词计数差）"
        log(f"      情绪分 = {_fmt(sentiment)}  来源：{source_desc}")
    elif args.sentiment_source == "none":
        log("      情绪：按参数跳过（--sentiment-source none）")
    else:
        log("      情绪：窗口内没有任何新闻/公告 → S 为 null（如实输出数据不足）")

    # ---- 5. 重大利空 ----
    hard_hits = detect_hard_negative(items) if items else []
    if args.llm_veto and items and _ollama_alive():
        hard_hits += llm_veto(items, args.model)
    if hard_hits:
        log(f"      [veto] 命中重大利空 {len(hard_hits)} 条：{hard_hits[0]}")

    # ---- 6. 仓位测算（KER 只在这里起作用） ----
    weight, size_note = size_from_ker(tech["ker14"], args.size_mode,
                                      args.ker_lo, args.ker_hi, args.max_weight,
                                      holding=(args.position == "long"),
                                      cost_price=args.cost, close=tech["close"])
    log(f"      仓位：{weight:.1%}   {size_note}")

    # ---- 7. 决策 ----
    result = decide(
        symbol=code6, trade_date=trade_date, holding=(args.position == "long"),
        tech=tech, items=items, sentiment=sentiment, sentiment_source=source_desc,
        hard_neg=hard_hits, sent_min=args.sent_min,
        target_weight=weight, size_note=size_note, data_report=qreport,
        rebalance=args.rebalance, current_weight=args.current_weight,
        rebalance_min=args.rebalance_min,
        cost_price=args.cost, stop_loss=args.stop_loss,
    )

    # ---- 8. 输出 ----
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            log(f"      已保存到 {args.out}")
        except Exception as exc:
            log(f"      [warn] 保存失败：{exc}")
    log(f"      耗时 {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

