"""完整变换日志（证书）中的初等操作。

只允许三类整数幺模初等操作：

* 交换：:class:`RowSwap` / :class:`ColSwap`
* 变号：:class:`RowSign` / :class:`ColSign`
* 加整数倍：:class:`RowAdd` / :class:`ColAdd`

约定（0 基下标）：

* ``RowAdd(t, s, k)``  表示  第 t 行 <- 第 t 行 + k * 第 s 行；
* ``ColAdd(t, s, k)``  表示  第 t 列 <- 第 t 列 + k * 第 s 列。

对应的初等矩阵行列式分别为 -1、-1、+1，故累计变换矩阵的行列式必为 ±1。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RowSwap",
    "RowSign",
    "RowAdd",
    "ColSwap",
    "ColSign",
    "ColAdd",
]


def _check_index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是 int，得到 {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} 不能为负，得到 {value}")
    return value


@dataclass(frozen=True, slots=True)
class RowSwap:
    i: int
    j: int

    def __post_init__(self) -> None:
        _check_index("i", self.i)
        _check_index("j", self.j)
        if self.i == self.j:
            raise ValueError("交换操作要求两个不同下标")


@dataclass(frozen=True, slots=True)
class RowSign:
    """第 i 行整体乘 -1。"""

    i: int

    def __post_init__(self) -> None:
        _check_index("i", self.i)


@dataclass(frozen=True, slots=True)
class RowAdd:
    target: int
    source: int
    multiple: int

    def __post_init__(self) -> None:
        _check_index("target", self.target)
        _check_index("source", self.source)
        if isinstance(self.multiple, bool) or not isinstance(self.multiple, int):
            raise TypeError("multiple 必须是 int")
        if self.target == self.source:
            raise ValueError("加整数倍操作要求两个不同下标")
        if self.multiple == 0:
            raise ValueError("multiple 为 0 的操作是空操作，不应记入日志")


@dataclass(frozen=True, slots=True)
class ColSwap:
    i: int
    j: int

    def __post_init__(self) -> None:
        _check_index("i", self.i)
        _check_index("j", self.j)
        if self.i == self.j:
            raise ValueError("交换操作要求两个不同下标")


@dataclass(frozen=True, slots=True)
class ColSign:
    """第 i 列整体乘 -1。"""

    i: int

    def __post_init__(self) -> None:
        _check_index("i", self.i)


@dataclass(frozen=True, slots=True)
class ColAdd:
    target: int
    source: int
    multiple: int

    def __post_init__(self) -> None:
        _check_index("target", self.target)
        _check_index("source", self.source)
        if isinstance(self.multiple, bool) or not isinstance(self.multiple, int):
            raise TypeError("multiple 必须是 int")
        if self.target == self.source:
            raise ValueError("加整数倍操作要求两个不同下标")
        if self.multiple == 0:
            raise ValueError("multiple 为 0 的操作是空操作，不应记入日志")
