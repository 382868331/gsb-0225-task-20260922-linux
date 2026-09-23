"""输入校验与精确整数矩阵运算。

本模块全程使用 Python 任意精度整数（``int``），不使用任何浮点运算，
因此不存在舍入误差。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ._errors import InputValidationError

MAX_DIM = 8
MAX_ABS_ENTRY = 10_000


def _locate(prefix: str, i: int, j: int | None = None) -> str:
    if j is None:
        return f"{prefix}[{i}]"
    return f"{prefix}[{i}][{j}]"


def _reject_numeric(value: object, where: str) -> None:
    """对单个标量执行整数契约检查；非法时抛 InputValidationError。"""
    if isinstance(value, bool):
        # bool 是 int 的子类，必须在 int 检查之前显式拒绝。
        raise InputValidationError(f"{where} 不能是 bool（需要 int），得到 {value!r}")
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise InputValidationError(f"{where} 不能是 NaN")
        if math.isinf(value):
            raise InputValidationError(f"{where} 不能是 Infinity 或 -Infinity")
        raise InputValidationError(
            f"{where} 必须是 int，不接受 float（得到 {value!r}）；本模块只做精确整数运算"
        )
    raise InputValidationError(
        f"{where} 必须是 int，得到 {type(value).__name__}（{value!r}）"
    )


def validate_matrix(
    A: object,
    name: str = "A",
    *,
    max_dim: int = MAX_DIM,
    max_abs: int = MAX_ABS_ENTRY,
) -> list[list[int]]:
    """校验整数矩阵，返回独立的可变副本（list[list[int]]）。

    契约：1 <= 行数、列数 <= max_dim；元素为 int（拒绝 bool、NaN、Infinity、
    float 及其他非整数类型）；|元素| <= max_abs；各行等长。
    """
    if not isinstance(A, Sequence) or isinstance(A, (str, bytes, bytearray)):
        raise InputValidationError(
            f"{name} 必须是行的 list/tuple（非空矩阵），得到 {type(A).__name__}"
        )
    if len(A) == 0:
        raise InputValidationError(f"{name} 不能为空矩阵（至少 1 行）")
    if len(A) > max_dim:
        raise InputValidationError(
            f"{name} 有 {len(A)} 行，超过上限 {max_dim}"
        )

    rows: list[list[int]] = []
    width: int | None = None
    for i, row in enumerate(A):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise InputValidationError(
                f"{_locate(name, i)} 必须是行 list/tuple，得到 {type(row).__name__}"
            )
        if len(row) == 0:
            raise InputValidationError(f"{_locate(name, i)} 为空行（至少 1 列）")
        if width is None:
            width = len(row)
            if width > max_dim:
                raise InputValidationError(
                    f"{_locate(name, i)} 有 {width} 列，超过上限 {max_dim}"
                )
        elif len(row) != width:
            raise InputValidationError(
                f"{_locate(name, i)} 长度为 {len(row)}，与第 0 行的 {width} 不一致（不接受参差矩阵）"
            )
        out_row: list[int] = []
        for j, value in enumerate(row):
            _reject_numeric(value, _locate(name, i, j))
            assert isinstance(value, int)
            if abs(value) > max_abs:
                raise InputValidationError(
                    f"{_locate(name, i, j)} = {value}，绝对值超过上限 {max_abs}"
                )
            out_row.append(value)
        rows.append(out_row)
    return rows


def validate_vector(b: object, length: int, name: str = "b") -> list[int]:
    """校验整数向量，长度必须为 length，返回独立副本。

    元素类型契约与矩阵相同；元素绝对值**不**设上限（Python 大整数精确运算）。
    """
    if not isinstance(b, Sequence) or isinstance(b, (str, bytes, bytearray)):
        raise InputValidationError(
            f"{name} 必须是 list/tuple，得到 {type(b).__name__}"
        )
    if len(b) != length:
        raise InputValidationError(
            f"{name} 长度为 {len(b)}，但矩阵有 {length} 行"
        )
    out: list[int] = []
    for i, value in enumerate(b):
        _reject_numeric(value, _locate(name, i))
        assert isinstance(value, int)
        out.append(value)
    return out


# --------------------------------------------------------------------- 精确运算


def identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """整数矩阵乘法 C = A * B（精确，无浮点）。"""
    m, k, n = len(A), len(B), len(B[0])
    Bt = [[B[i][j] for i in range(k)] for j in range(n)]
    return [
        [sum(a * b for a, b in zip(A[i], Bt[j])) for j in range(n)]
        for i in range(m)
    ]


def matvec(A: list[list[int]], x: list[int]) -> list[int]:
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def transpose(A: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*A)]


def determinant(M: Sequence[Sequence[int]]) -> int:
    """Bareiss 分数无关消元求行列式，结果精确为整数。"""
    n = len(M)
    A = [list(row) for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        pivot = k
        while pivot < n and A[pivot][k] == 0:
            pivot += 1
        if pivot == n:
            return 0
        if pivot != k:
            A[k], A[pivot] = A[pivot], A[k]
            sign = -sign
        p = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            row_i = A[i]
            row_k = A[k]
            for j in range(k + 1, n):
                row_i[j] = (p * row_i[j] - aik * row_k[j]) // prev
            row_i[k] = 0
        prev = p
    return sign * A[n - 1][n - 1]
