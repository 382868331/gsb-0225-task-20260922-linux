"""smithcert: 整数矩阵 Smith 标准形证书与整数线性方程组求解。

仅依赖 Python 标准库，全程精确整数运算。
"""

from .errors import (
    BudgetExhaustedError,
    DimensionMismatchError,
    IntegerRequiredError,
    InvalidMatrixError,
    MatrixLimitError,
    MatrixShapeError,
    NonFiniteNumberError,
    SmithCertError,
)
from .solve import (
    BOUNDED_STATUS_BUDGET_EXHAUSTED,
    BOUNDED_STATUS_FEASIBLE,
    BOUNDED_STATUS_INFEASIBLE,
    BOUNDED_STATUS_OPTIMAL,
    BOUNDED_STATUS_UNSUPPORTED,
    BoundedSolveResult,
    IntegerSolveResult,
    Obstruction,
    solve_bounded_integer,
    solve_integer,
)
from .transform import Transform, replay
from .smith import SmithResult, smith

MAX_ROWS = 8
MAX_COLS = 8
MAX_ABS_ENTRY = 10_000
DEFAULT_BUDGET = 100_000

__all__ = [
    "smith",
    "solve_integer",
    "replay",
    "Transform",
    "SmithResult",
    "IntegerSolveResult",
    "BoundedSolveResult",
    "Obstruction",
    "solve_bounded_integer",
    "BOUNDED_STATUS_OPTIMAL",
    "BOUNDED_STATUS_FEASIBLE",
    "BOUNDED_STATUS_INFEASIBLE",
    "BOUNDED_STATUS_BUDGET_EXHAUSTED",
    "BOUNDED_STATUS_UNSUPPORTED",
    "SmithCertError",
    "InvalidMatrixError",
    "MatrixShapeError",
    "DimensionMismatchError",
    "MatrixLimitError",
    "IntegerRequiredError",
    "NonFiniteNumberError",
    "BudgetExhaustedError",
    "MAX_ROWS",
    "MAX_COLS",
    "MAX_ABS_ENTRY",
    "DEFAULT_BUDGET",
]
