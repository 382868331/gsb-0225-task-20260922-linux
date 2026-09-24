"""smithcert: 整数矩阵 Smith 标准形证书与整数线性方程组求解。

仅依赖 Python 标准库，全程精确整数运算。
"""

from .errors import (
    BoundaryProductError,
    BudgetExhaustedError,
    ChainComplexError,
    DimensionMismatchError,
    IntegerRequiredError,
    InvalidMatrixError,
    MatrixLimitError,
    MatrixShapeError,
    NonFiniteNumberError,
    SmithCertError,
)
from .homology import HomologyGroup, HomologyResult, homology
from .solve import Obstruction, IntegerSolveResult, solve_integer
from .transform import Transform, replay
from .smith import SmithResult, smith

MAX_ROWS = 8
MAX_COLS = 8
MAX_ABS_ENTRY = 10_000
MAX_CHAIN_DIM = 2
DEFAULT_BUDGET = 100_000

__all__ = [
    "smith",
    "solve_integer",
    "homology",
    "replay",
    "Transform",
    "SmithResult",
    "IntegerSolveResult",
    "Obstruction",
    "HomologyGroup",
    "HomologyResult",
    "SmithCertError",
    "InvalidMatrixError",
    "MatrixShapeError",
    "DimensionMismatchError",
    "MatrixLimitError",
    "IntegerRequiredError",
    "NonFiniteNumberError",
    "BudgetExhaustedError",
    "ChainComplexError",
    "BoundaryProductError",
    "MAX_ROWS",
    "MAX_COLS",
    "MAX_ABS_ENTRY",
    "MAX_CHAIN_DIM",
    "DEFAULT_BUDGET",
]
