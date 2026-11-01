"""金额边界统一校验，退款累计使用整数分，避免浮点数累加误差。"""
from __future__ import annotations

from decimal import Decimal, DecimalException


def to_cents(value) -> int:
    """接受非负、有限且最多两位小数的金额，返回 SQLite 可保存的整数分。"""
    if isinstance(value, bool):
        raise ValueError("金额不能使用布尔值。")
    try:
        amount = Decimal(str(value))
        if not amount.is_finite():
            raise ValueError("金额必须是有限数值。")
        cents = amount * 100
        if cents != cents.to_integral_value():
            raise ValueError("金额最多支持两位小数。")
        if cents < 0 or cents > 2**63 - 1:
            raise ValueError("金额超出允许范围。")
        return int(cents)
    except (DecimalException, TypeError) as exc:
        raise ValueError("金额必须是有效数值。") from exc


def from_cents(cents: int) -> float:
    """仅用于接口和界面展示，余额计算仍使用整数分。"""
    return float(Decimal(cents) / 100)
