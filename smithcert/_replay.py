"""变换日志的独立重放器。

给定原始矩阵和 ``smith`` 返回的完整操作日志，仅通过把每条初等操作实例化为
初等矩阵并做精确矩阵乘法，重建 U、D、V：

    行操作 op：U <- E * U
    列操作 op：V <- V * F
    最后      D = U * A * V

它不调用消元算法本身，可作为证书的独立核验路径。
"""

from __future__ import annotations

from collections.abc import Sequence

from ._basic import identity, matmul
from ._ops import (
    ColAdd,
    ColSign,
    ColSwap,
    RowAdd,
    RowSign,
    RowSwap,
)


def _row_elem(n: int, op: object) -> list[list[int]]:
    E = identity(n)
    if isinstance(op, RowSwap):
        E[op.i], E[op.j] = E[op.j], E[op.i]
    elif isinstance(op, RowSign):
        E[op.i][op.i] = -1
    elif isinstance(op, RowAdd):
        # 第 target 行 <- 第 target 行 + multiple * 第 source 行
        E[op.target][op.source] = op.multiple
    else:  # pragma: no cover - 由调用方保证
        raise TypeError(f"不是行初等操作：{op!r}")
    return E


def _col_elem(n: int, op: object) -> list[list[int]]:
    F = identity(n)
    if isinstance(op, ColSwap):
        for row in F:
            row[op.i], row[op.j] = row[op.j], row[op.i]
    elif isinstance(op, ColSign):
        F[op.i][op.i] = -1
    elif isinstance(op, ColAdd):
        # 第 target 列 <- 第 target 列 + multiple * 第 source 列
        F[op.source][op.target] = op.multiple
    else:  # pragma: no cover - 由调用方保证
        raise TypeError(f"不是列初等操作：{op!r}")
    return F


def replay_certificate(
    A: Sequence[Sequence[int]], operations: Sequence[object]
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """重放日志，返回 (U, D, V)，满足 D = U A V。"""
    m = len(A)
    n = len(A[0])
    U = identity(m)
    V = identity(n)
    A0 = [list(row) for row in A]
    for op in operations:
        if isinstance(op, (RowSwap, RowSign, RowAdd)):
            U = matmul(_row_elem(m, op), U)
        elif isinstance(op, (ColSwap, ColSign, ColAdd)):
            V = matmul(V, _col_elem(n, op))
        else:
            raise TypeError(f"日志中存在未知操作类型：{op!r}")
    D = matmul(matmul(U, A0), V)
    return U, D, V
