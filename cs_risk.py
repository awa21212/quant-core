"""风控层：费用、整手、回撤熔断、全局宏观减仓目标。

约定：
- 所有「手数」都走 `lot()`，向下取整到 100 股
- 卖出费用：佣金（最低 5 元）+ 印花税（指数/ETF 免）
- 熔断是**状态机**：回撤触及 `trigger` 打开，回到 `recover` 以内才关
- 宏观减仓是**目标制**（按原始份额算目标），不是「按比例卖」，避免反复穿线
  把底仓一刀刀削没
"""
from __future__ import annotations

COMMISSION = 0.00025      # 万 2.5（可用 --commission 覆盖）
COMMISSION_MIN = 5.0      # 单笔最低 5 元（可用 --commission-min 覆盖）
STAMP = 0.0005            # 印花税，现行 A 股卖出单边 0.05%（2023-08-28 起）；ETF/指数免
TRANSFER = 0.00001        # 过户费 0.001%（沪深 A 股双向；ETF 免）
SLIPPAGE = 0.0002         # 基础滑点 0.02%（单边）
LOT = 100

DD_TRIGGER = 0.10
DD_FACTOR = 0.30
DD_RECOVER = 0.05

CUT_MAP = {"hold": 0.0, "half": 0.5, "zero": 1.0}   # 值 = 需要减掉的比例

_STAMP = [STAMP]
_COMMISSION = [COMMISSION]
_COMMISSION_MIN = [COMMISSION_MIN]
_TRANSFER = [TRANSFER]
_IMPACT = [0.0]           # 冲击成本：在基础滑点之上再加的单边 bps


def set_fees(commission=None, commission_min=None, transfer=None,
             impact=None, stamp=None) -> None:
    """运行前统一注入费用参数（引擎 setup 调用一次）。"""
    if commission is not None:
        _COMMISSION[0] = float(commission)
    if commission_min is not None:
        _COMMISSION_MIN[0] = float(commission_min)
    if transfer is not None:
        _TRANSFER[0] = float(transfer)
    if impact is not None:
        _IMPACT[0] = float(impact)
    if stamp is not None:
        _STAMP[0] = float(stamp)


def slip_buy() -> float:
    """买入执行价 = 收盘 ×(1 + slip_buy)。"""
    return SLIPPAGE + _IMPACT[0]


def slip_sell() -> float:
    """卖出执行价 = 收盘 ×(1 − slip_sell)。"""
    return SLIPPAGE + _IMPACT[0]


def buy_fee(amount: float, on: bool, stock: bool = False) -> float:
    """买入费用：佣金（最低 5 元）+ 过户费（仅 A 股股票）。"""
    if not on:
        return 0.0
    fee = max(_COMMISSION_MIN[0], amount * _COMMISSION[0])
    if stock:
        fee += amount * _TRANSFER[0]
    return fee


def sell_fee(amount: float, on: bool, etf: bool = False) -> float:
    """卖出费用：佣金（最低 5 元）+ 印花税 + 过户费（后两项 ETF/指数免）。"""
    if not on:
        return 0.0
    fee = max(_COMMISSION_MIN[0], amount * _COMMISSION[0])
    if not etf:
        fee += amount * _STAMP[0] + amount * _TRANSFER[0]
    return fee


def lot(shares: float) -> int:
    return int(shares // LOT) * LOT


def breaker_step(on: bool, dd: float, trigger: float = DD_TRIGGER,
                 recover: float = DD_RECOVER) -> tuple[bool, str | None]:
    """熔断状态机：返回 (新状态, 事件名)。事件名 ∈ {触发熔断, 解除熔断, None}。"""
    if not on and dd <= -abs(trigger):
        return True, "触发熔断"
    if on and dd >= -abs(recover):
        return False, "解除熔断"
    return on, None


def cut_ratio(mode: str) -> float:
    """hold/half/zero → 减仓比例（0 / 0.5 / 1）。非法值按 half 处理。"""
    return CUT_MAP.get(str(mode).lower(), 0.5)
