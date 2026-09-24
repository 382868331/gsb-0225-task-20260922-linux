"""smithcert 的公开错误类型。

所有异常都带可定位信息（矩阵名、下标或参数名），非法输入一律抛错，
不做静默纠正。
"""


class SmithCertError(Exception):
    """本包所有异常的基类。"""


class InvalidMatrixError(SmithCertError):
    """矩阵输入非法的基类。"""


class MatrixShapeError(InvalidMatrixError):
    """不是非空 list[list[int]]、行长度不一致等形状错误。"""


class DimensionMismatchError(InvalidMatrixError):
    """如 b 的长度与 A 的行数不一致。"""


class MatrixLimitError(InvalidMatrixError):
    """超过公开接口上限（8×8、元素绝对值 10000）。"""


class IntegerRequiredError(InvalidMatrixError, TypeError):
    """需要精确 int 的位置收到 bool / float / 其他类型。

    按契约 bool 即使是 int 子类也一律拒绝；float 即使值为 2.0 也拒绝
    （本库只做精确整数运算，不接受隐式舍入）。
    """


class NonFiniteNumberError(InvalidMatrixError, ValueError):
    """数值输入为 NaN 或 ±Infinity。"""


class NotAChainComplexError(SmithCertError):
    """相邻边缘算子的乘积不为零：输入不是链复形。

    链复形要求 ∂_k · ∂_{k+1} = 0（精确整数相等）。违例时抛此异常，
    信息含乘积矩阵中第一个非零元的位置与数值，便于定位。
    """


class BudgetExhaustedError(SmithCertError, RuntimeError):
    """初等变换步数预算耗尽（搜索未能结束）。

    与“证明无解”严格区分：整数方程组无解时 solve_integer 返回
    consistent=False 及余数证据，绝不抛此异常。在公开上限内正常输入
    不应触发；触发只表示搜索未完成，不代表任何数学结论。
    """
