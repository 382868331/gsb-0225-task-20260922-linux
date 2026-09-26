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
from .solve import Obstruction, IntegerSolveResult, solve_integer
from .bounded import (
    BoundedSolveResult,
    BUDGET_EXHAUSTED,
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNSUPPORTED,
    solve_bounded_integer,
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
    "Obstruction",
    "BoundedSolveResult",
    "solve_bounded_integer",
    "OPTIMAL",
    "FEASIBLE",
    "INFEASIBLE",
    "UNSUPPORTED",
    "BUDGET_EXHAUSTED",
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
