"""smithcert —— 整数矩阵 Smith 标准形证书（标准库，纯整数精确计算）。

公开接口：

* :func:`smith` —— Smith 标准形 + 完整初等变换日志证书；
* :func:`integer_solve` —— A x = b 整数可解判定、一个特解、整数零空间基；
* :func:`replay_certificate` —— 独立重放日志重建 U、D、V。

输入边界：矩阵至多 8×8，元素绝对值 <= 10000，必须是 ``int``
（拒绝 bool、float、NaN、Infinity）。右端向量为等长整数序列。
"""

from __future__ import annotations

from ._basic import (
    MAX_ABS_ENTRY,
    MAX_DIM,
    determinant,
    matmul,
    matvec,
    transpose,
    validate_matrix,
)
from ._errors import BudgetExhaustedError, InputValidationError, SmithCertError
from ._ops import (
    ColAdd,
    ColSign,
    ColSwap,
    RowAdd,
    RowSign,
    RowSwap,
)
from ._replay import replay_certificate
from ._smith import DEFAULT_BUDGET, SmithResult, smith_normal_form
from ._solve import IntegerSolveResult, integer_solve

__all__ = [
    "smith",
    "integer_solve",
    "replay_certificate",
    "SmithResult",
    "IntegerSolveResult",
    "RowSwap",
    "RowSign",
    "RowAdd",
    "ColSwap",
    "ColSign",
    "ColAdd",
    "SmithCertError",
    "InputValidationError",
    "BudgetExhaustedError",
    "DEFAULT_BUDGET",
    "MAX_DIM",
    "MAX_ABS_ENTRY",
    "determinant",
    "matmul",
    "matvec",
    "transpose",
    "validate_matrix",
]

__version__ = "1.0.0"


def smith(A: object, *, budget: int = DEFAULT_BUDGET) -> SmithResult:
    """计算整数矩阵 A 的 Smith 标准形证书。

    返回 :class:`SmithResult`，其中整数幺模矩阵 U、V 与标准形 D 满足
    精确恒等式 ``U * A * V == D``，``operations`` 为完整初等变换日志。

    :raises InputValidationError: 输入不满足契约（含可定位下标）。
    :raises BudgetExhaustedError: 变换次数超过预算（非“无解”证明）。
    """
    M = validate_matrix(A, "A")
    return smith_normal_form(M, budget=budget)
