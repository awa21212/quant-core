"""数据层：行情抓取、缓存、清洗、指标。

从 core_satellite.py 抽出，行为保持一致。"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

import portfolio as pf
from cs_regime import BEAR_POOL, BEAR_POOL_ALL, BEAR_MA


LOG = lambda m: print(m, file=__import__("sys").stderr)

CORE_INDEX = "sh000300"
CORE_NAME = "沪深300"

# 行业/主题 ETF 池（替代「挑个股」的卫星腿）：只做动量轮动，不做个股基本面
# 选择标准：2022 年前上市、日均成交额 > 5000 万、行业互不重叠
ETF_UNIVERSE = {
    "512880": ("券商ETF", "非银金融"),
    "512800": ("银行ETF", "银行"),
    "512170": ("医疗ETF", "医药生物"),
    "512660": ("军工ETF", "国防军工"),
    "512760": ("芯片ETF", "电子"),
    "515030": ("新能源车ETF", "电力设备"),
    "515790": ("光伏ETF", "电力设备"),
    "512200": ("房地产ETF", "房地产"),
    "512400": ("有色ETF", "有色金属"),
    "515220": ("煤炭ETF", "煤炭"),
    "159928": ("消费ETF", "食品饮料"),
    "512980": ("传媒ETF", "传媒"),
    # —— 扩到 20 只：补宽基、防御与海外，降低“纯行业轮动”的集中度 ——
    "510300": ("沪深300ETF", "宽基"),
    "159915": ("创业板ETF", "成长"),
    "588000": ("科创50ETF", "科技成长"),
    "510880": ("红利ETF", "红利"),
    "518880": ("黄金ETF", "黄金"),
    "513050": ("中概互联ETF", "港股互联网"),
    "513100": ("纳指ETF", "海外科技"),
    "512690": ("酒ETF", "食品饮料"),
}
ETF_TOP = 3          # 每次持有动量最强的几只
ETF_MOM = 20         # 动量回看天数（20 个交易日 ≈ 1 个月）
# 扩容前的 12 只原始池（--etf-pool 12 时使用，便于对照）
BROAD_ETF = {"510300", "159915", "588000"}
# 15 只密集池 = 20 只去掉「结构性重复/不适合动量」的 5 只：
#   3 只宽基（与核心 50% 沪深300 重复）+ 红利（慢趋势，20 日动量多为假信号）+ 酒（与消费高度重叠）
DEFAULT_REGULAR = {
    "000001": ("平安银行", "银行"),
    "000858": ("五粮液", "食品饮料"), "002304": ("洋河股份", "食品饮料"),
    "600887": ("伊利股份", "食品饮料"),
    "300750": ("宁德时代", "电力设备"), "601012": ("隆基绿能", "电力设备"),
    "002594": ("比亚迪", "汽车"),
    "600030": ("中信证券", "非银金融"), "601318": ("中国平安", "非银金融"),
    "002415": ("海康威视", "电子"),
    "601899": ("紫金矿业", "有色金属"),
    "000651": ("格力电器", "家用电器"),
    "601668": ("中国建筑", "建筑装饰"),
    "600009": ("上海机场", "交通运输"),
    "002230": ("科大讯飞", "计算机"),
}

# 5 只「长期股」：买入持有，不做择时（压舱石，但仍在卫星账本里）
# 注意：刻意避开茅台这类「一手就要 14 万」的票——100 万本金下它占 14%，
#       塞不进 5% 的长期仓位里，硬买会破坏结构。
DEFAULT_LONGTERM = {
    "600036": ("招商银行", "银行"),
    "000333": ("美的集团", "家用电器"),
    "601088": ("中国神华", "煤炭"),
    "600276": ("恒瑞医药", "医药生物"),
    "600900": ("长江电力", "公用事业"),
}

# 合并后的行业映射（长期股优先，避免同一只票两套名字）
INDUSTRY_MAP = {**{c: v[1] for c, v in DEFAULT_REGULAR.items()},
                **{c: v[1] for c, v in DEFAULT_LONGTERM.items()}}
NAME_MAP = {**{c: v[0] for c, v in DEFAULT_REGULAR.items()},
            **{c: v[0] for c, v in DEFAULT_LONGTERM.items()}}


# ============================================================================
# 二、取数（失败会退避重试，并把系统代理清掉）
# ============================================================================
def _last_trading_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """「现在」往回退到最近一个已经收盘的交易日（15:30 前算昨天，周末回退）。

    只排除周末、不认节假日：节假日那几天多问一次数据源，代价远小于「缓存卡住不更新」。
    """
    now = now or pd.Timestamp.now()
    day = now.normalize()
    if now.hour * 60 + now.minute < 15 * 60 + 30:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def _is_stale(last: pd.Timestamp, grace_days: int = 0) -> bool:
    """缓存最后一根 K 线是不是没跟上最近交易日。"""
    return pd.Timestamp(last).normalize() < _last_trading_day() - pd.Timedelta(days=grace_days)


def load_index(start: str, end: str, path: str = "index_sh000300.csv") -> pd.Series:
    """沪深300 收盘价（先用本地缓存；缓存停在旧交易日就增补）。

    2026-11-24 修：原先只要 CSV 存在就永远不再联网，指数卡在 11-20 之后，
    裁判/模拟盘/日报全部拿着同一根旧 K 线出「夏普 -0.00」。
    """
    df = None
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
    need_download = df is None or df.empty or _is_stale(df["date"].max())
    if need_download:
        import akshare as ak
        raw = _ak_retry(lambda: ak.stock_zh_index_daily(symbol=CORE_INDEX), "沪深300 指数")
        raw = raw[["date", "close"]].copy()
        raw["date"] = pd.to_datetime(raw["date"])
        if df is not None and not df.empty:            # 增量合并，别把历史丢了
            before = len(df)
            df = (pd.concat([df, raw], ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date"))
            added = len(df) - before
            LOG(f"[index] 缓存停在 {pd.to_datetime(df['date']).max().date()} 之前，"
                f"补 {added} 根 K 线 → {path}")
        else:
            df = raw.sort_values("date")
            LOG(f"[index] 下载 {CORE_INDEX} → {path}（{len(df)} 行）")
        df.to_csv(path, index=False)
    s = df.set_index("date")["close"].sort_index()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s.astype(float)


def _clear_proxy_env() -> None:
    """系统里常残留一个已经退出的代理（127.0.0.1:7892），requests 会直接 ProxyError。"""
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    os.environ.setdefault("NO_PROXY", "*")


def _ak_retry(fn, label: str):
    """先直连；失败就清代理再试一次（实测顺序反过来更省时间）。"""
    _clear_proxy_env()
    try:
        return fn()
    except Exception as exc:
        LOG(f"[warn] {label} 首次请求失败（{type(exc).__name__}），重置连接后重试…")
        time.sleep(1.0)
        return fn()


def load_stocks(codes: list[str], end: str, adjust: str = "hfq",
                start: str | None = None) -> dict[str, pd.DataFrame]:
    """逐只取数；取不到的票直接剔除（不因为一只票挂掉整条流水线）。

    默认用后复权（hfq）：回测动辄回到 2015 年，前复权会把高分红股的历史价减成负数。
    """
    # 回看天数由回测起点决定：早先写死 420 天，跑 2020–2025 时长期股只有最后一年多，
    # 前面几年全被裁掉、静默变成「长期股 0 只」。
    lookback = 420
    if start:
        lookback = max(420, (pd.Timestamp(end) - pd.Timestamp(start)).days + 420)
    out: dict[str, pd.DataFrame] = {}
    for c in codes:
        try:
            df = _ak_retry(lambda c=c: pf.load_one(c, pd.Timestamp(end), adjust=adjust,
                                                   lookback_days=lookback),
                           f"个股 {c}")
        except Exception as e:                                   # noqa: BLE001
            LOG(f"[data] {c} 取数失败：{e}")
            continue
        if df is None or df.empty:
            LOG(f"[data] {c} 空数据，跳过")
            continue
        # 后复权价是「上市以来累计乘数」，一只 20 年高分红股的价格可能被放大到 3000 元/股，
        # 按它算整手就买不起了（恒瑞一手 30 万）。用首日真实价做锚点整体缩放：
        # 收益率不变（同一常数乘子），但绝对价格回到现实，整手/资金约束才有意义。
        if adjust == "hfq" and not df.empty:
            try:
                raw = _ak_retry(lambda c=c: pf.load_one(c, pd.Timestamp(end), adjust="",
                                                        lookback_days=lookback), f"个股真实价 {c}")
                common = df.index.intersection(raw.index)
                if len(common) and float(df.loc[common[0], "close"]) > 0:
                    k = float(raw.loc[common[0], "close"]) / float(df.loc[common[0], "close"])
                    if 0 < k < 10:
                        for col in ("open", "high", "low", "close"):
                            df[col] = df[col] * k
                        LOG(f"[data] {c} 后复权价按首日真实价缩放 x{k:.4f}"
                            f"（{common[0].date()} {raw.loc[common[0], 'close']:.2f} 元）")
            except Exception as e:                               # noqa: BLE001
                LOG(f"[data] {c} 真实价缩放失败（保持后复权）：{e}")
        out[c] = df
    return out


def _fix_splits(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """ETF 份额折算（1:2 拆分/合并）在不复权数据里会显示成「单日 −50%」。

    新浪的 ETF 日线是不复权的，遇到折算日会把历史价「腰斩」——这不是真的暴跌，
    而是份额翻倍、价格减半。这里用「单日涨跌超过 25%」当作折算日，把折算日之前
    的 OHLC 全部按比例缩放到折算后的价位上，让收益率曲线连续。
    """
    c = df["close"].astype(float).to_numpy()
    n = len(c)
    if n < 2:
        return df
    mult = np.ones(n)
    cum = 1.0
    events: list[tuple[str, float]] = []
    for i in range(n - 1, 0, -1):
        if c[i - 1] > 0 and abs(c[i] / c[i - 1] - 1.0) > 0.25:
            ratio = c[i] / c[i - 1]          # 折算比例（拆分≈0.5，合并≈2）
            cum *= ratio
            events.append((str(df.index[i].date()), round(ratio, 4)))
        mult[i - 1] = cum
    if not events:
        return df
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype(float) * mult
    LOG(f"[etf] {code} 检测到 {len(events)} 次份额折算并已修正：{events[::-1]}")
    return df


def load_etf(code: str, start: str, end: str, mom: int, trend_ma: int = 0) -> pd.DataFrame:
    """ETF 日线（新浪源，不复权——ETF 分红很少，不复权影响可忽略）+ 与个股一致的指标列。"""
    path = f"etf_{code}.csv"
    # 需要的前置窗口：早于 start 约 400 天，否则动量/年线在区间起点全是 NaN
    need_from = pd.Timestamp(start) - pd.Timedelta(days=400)
    need_to = pd.Timestamp(end)
    df = None
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        if df is None or df.empty:
            df = None
        else:
            lo, hi = df["date"].min(), df["date"].max()
            # 原先用「落后 10 天以内就算覆盖」的宽松判据：短假能容忍，
            # 但一旦停更 1~9 天，缓存就被当新鲜货，ETF 价格冻住 → 权益曲线平掉、夏普归零。
            if lo > need_from or _is_stale(hi):
                LOG(f"[etf] {code} 缓存不覆盖 {need_from.date()}~{need_to.date()}"
                    f"（{lo.date()}~{hi.date()}）→ 重新下载")
                df = None
    if df is None or df.empty:
        import akshare as ak
        prefix = "sh" if code.startswith(("5", "6")) else "sz"
        # 一次拉全量（从回测起点前 400 天拉起），这样换年份回测也不用重新下载，
        # 动量指标有足够前置窗口；2015 回测需要 2014 年的数据，所以不能写死 2016。
        beg = str(need_from.date()).replace("-", "")
        raw = _ak_retry(
            lambda cc=f"{prefix}{code}": ak.stock_zh_a_daily(
                symbol=cc, start_date=beg, end_date="20261231"),
            f"ETF {code}")
        df = raw[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(path, index=False)
        LOG(f"[etf] 下载 {code} → {path}（{len(df)} 行）")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").set_index("date")
    if os.environ.get("ETF_NO_SPLIT_FIX") != "1":
        df = _fix_splits(df, code)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["mom"] = df["close"] / df["close"].shift(mom) - 1.0
    df["vol20"] = df["close"].pct_change().rolling(20).std()
    df["above_ma20"] = df["close"] >= df["ma20"]
    # 长期趋势过滤（trend_ma=250 即年线）：只在长期均线上方才允许入场
    if trend_ma and trend_ma > 0:
        df["above_trend"] = df["close"] >= df["close"].rolling(int(trend_ma)).mean()
    else:
        df["above_trend"] = True
    df["golden"] = False
    df["golden_n"] = False
    df["trend_up"] = df["ma5"] > df["ma20"]
    # 指标算完再按回测区间切片（保证区间起点就有动量值）
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df


def _safe_load_etf(code: str, start: str, end: str, mom: int, trend_ma: int = 0):
    """
    加载单只 ETF；上市晚于回测起点（比如 2019 年才上市的芯片ETF）或无数据时返回 None。
    为什么要这个包装：早年回测（2015）里二十只 ETF 有一大半还不存在，
    不能因为一只票取不到数就让整条流水线崩掉。
    """
    try:
        df = load_etf(code, start, end, mom, trend_ma)
    except Exception as exc:
        LOG(f"[etf] {code} 取数失败，跳过：{exc}")
        return None
    if df is None or df.empty or len(df) < 30:
        n = 0 if df is None else len(df)
        LOG(f"[etf] {code} 区间内数据不足（{n} 行，可能上市晚于 {start}），跳过")
        return None
    return df


def prepare(cfg: Config):
    """对齐到指数交易日历，算出「连续 N 日金叉」等派生列。"""
    idx = load_index(cfg.start, cfg.end)
    if idx.empty:
        raise SystemExit(f"指数 {CORE_INDEX} 在 {cfg.start}~{cfg.end} 无数据")
    dates = list(idx.index)

    etf_mode = cfg.sleeve == "etf"
    if etf_mode and getattr(cfg, "etf_pool", "20") == "12":
        cfg.etf = {c: v for c, v in cfg.etf.items() if c in ETF_POOL12}
    elif etf_mode and getattr(cfg, "etf_pool", "20") == "17":
        cfg.etf = {c: v for c, v in cfg.etf.items() if c not in BROAD_ETF}
    elif etf_mode and getattr(cfg, "etf_pool", "20") == "15":
        cfg.etf = {c: v for c, v in cfg.etf.items() if c in ETF_POOL15}
    sleeve_codes = list(cfg.etf) if etf_mode else list(cfg.regular)
    all_codes = sleeve_codes + [c for c in cfg.longterm if c not in sleeve_codes]

    # V3.0：开启状态切换时，必须把防守资产也加载进来（即便它们不在进攻池里）
    bear_pool = BEAR_POOL_ALL if getattr(cfg, "bear_pool", "bond_gold") == "all" else BEAR_POOL
    bear_codes = [c for c in bear_pool if cfg.regime != "off"]
    if etf_mode:
        load_list = [c for c in sleeve_codes if c not in cfg.longterm]
        load_list += [c for c in bear_codes if c not in load_list and c not in cfg.longterm]
        raw = {}
        for c in load_list:
            got = _safe_load_etf(c, cfg.start, cfg.end, cfg.etf_mom,
                                 getattr(cfg, "trend_ma", 0))
            if got is not None:
                raw[c] = got
        raw.update(load_stocks([c for c in cfg.longterm], cfg.end, start=cfg.start))
    else:
        raw = load_stocks(all_codes, cfg.end, start=cfg.start)
        if bear_codes:
            # 个股版 + 状态切换：防守资产本来不在股票池里，得单独按 ETF 口径加载
            for c in bear_codes:
                got = _safe_load_etf(c, cfg.start, cfg.end, cfg.etf_mom, 0)
                if got is not None:
                    raw[c] = got

    aligned: dict[str, pd.DataFrame] = {}
    for code, df in raw.items():
        d = df.reindex(dates)
        # 保留「未 ffill」的原始收盘：用于判定停牌（当日无成交）与一字板
        if "close" in d.columns:
            d["raw_close"] = d["close"]
        d = d.ffill()
        # 「连续 N 日确认」的正确含义：金叉发生在 N-1 天前，且到今天为止 ma5>ma20 一直成立。
        # 注意：金叉当天 MA5 才上穿，第二天不可能再「金叉」一次，所以不能直接对 golden 做 rolling。
        up = (d["ma5"] > d["ma20"]) & d["ma5"].notna() & d["ma20"].notna()
        trend_held = up.rolling(cfg.confirm, min_periods=cfg.confirm).sum() == cfg.confirm
        golden_prev = d["golden"].fillna(False).shift(max(0, cfg.confirm - 1)).fillna(False).astype(bool)
        d["golden_n"] = golden_prev & trend_held
        d["trend_up"] = up
        if code in bear_codes:
            # 防御资产自己的趋势闸门（默认 MA60）：比 MA20 抗噪，债券 ETF 尤其需要
            d["bear_ma"] = d["close"].rolling(int(cfg.bear_ma)).mean()
            d["above_bear"] = d["close"] >= d["bear_ma"]
        aligned[code] = d
    return idx, dates, aligned
