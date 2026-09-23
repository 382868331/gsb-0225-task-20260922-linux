"""Smith 标准形证书模块的错误类型。

所有异常都继承自 :class:`SmithCertError`，调用方既可以按具体异常处理，
也可以统一捕获。
"""


class SmithCertError(Exception):
    """本模块所有公开异常的公共基类。"""


class InputValidationError(ValueError, SmithCertError):
    """输入不满足公开契约（形状、取值范围、NaN/Infinity 等）。

    错误信息始终携带可定位的位置（例如 ``A[1][2]``）。
    """


class BudgetExhaustedError(RuntimeError, SmithCertError):
    """初等变换的搜索预算耗尽。

    这是“算法没能在预算内算完”，**不**表示数学上无解。整数线性方程组
    无解由 :func:`smithcert.integer_solve` 的结果字段（含余数/障碍证书）
    表达，二者严格区分。
    """
