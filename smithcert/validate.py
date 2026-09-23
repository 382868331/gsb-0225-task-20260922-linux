"""输入校验：形状、整数性、有限性、接口上限。

校验失败即抛带定位信息的异常，绝不就地修改用户对象。
"""

from math import isfinite

from .errors import (
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    NonFiniteNumberError,
)


def _check_int(value, where):
    """要求精确 int：拒绝 bool（bool 是 int 子类也拒）、拒绝 float/其他。

    对 float 还要单独区分 NaN/Infinity（按契约要明确报“非有限数”）。
    """
    if isinstance(value, bool):
        raise IntegerRequiredError(f"{where}: bool 不是合法整数参数（拒绝 True/False）")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise NonFiniteNumberError(
                f"{where}: 数值必须有限，收到 {value!r}（拒绝 NaN/Infinity）"
            )
        raise IntegerRequiredError(
            f"{where}: 需要精确 int，收到 float {value!r}（不做隐式取整）"
        )
    # 拒绝 Fraction、complex、str 等，杜绝隐式转换。
    raise IntegerRequiredError(
        f"{where}: 需要精确 int，收到 {type(value).__name__} 类型的 {value!r}"
    )


def validate_matrix(A, *, name="A", max_rows=None, max_cols=None,
                    max_abs=None, min_rows=1, min_cols=1):
    """校验整数矩阵，返回 (rows, cols)。不复制、不修改 A。

    规则：
      - A 必须是非空 list，每行是非空 list；
      - 各行等长；
      - 每个元素精确 int（拒绝 bool）；非 int 的 float 若为 NaN/Inf
        报 NonFiniteNumberError，否则报 IntegerRequiredError；
      - |元素| <= max_abs；行数/列数不超过给定上限。
    """
    if not isinstance(A, list):
        raise MatrixShapeError(
            f"{name}: 矩阵必须是 list[list[int]]，收到 {type(A).__name__}"
        )
    if len(A) < min_rows:
        raise MatrixShapeError(f"{name}: 矩阵至少要有 {min_rows} 行，收到 {len(A)} 行")
    if max_rows is not None and len(A) > max_rows:
        raise MatrixLimitError(
            f"{name}: 行数 {len(A)} 超过上限 {max_rows}"
        )
    cols = None
    for i, row in enumerate(A):
        if not isinstance(row, list):
            raise MatrixShapeError(
                f"{name}[{i}]: 每一行必须是 list，收到 {type(row).__name__}"
            )
        if cols is None:
            cols = len(row)
            if cols < min_cols:
                raise MatrixShapeError(
                    f"{name}: 矩阵至少要有 {min_cols} 列，收到 {cols} 列"
                )
            if max_cols is not None and cols > max_cols:
                raise MatrixLimitError(
                    f"{name}: 列数 {cols} 超过上限 {max_cols}"
                )
        elif len(row) != cols:
            raise MatrixShapeError(
                f"{name}[{i}]: 行长度 {len(row)} 与首行长度 {cols} 不一致"
            )
        for j, x in enumerate(row):
            _check_int(x, f"{name}[{i}][{j}]")
            if max_abs is not None and abs(x) > max_abs:
                raise MatrixLimitError(
                    f"{name}[{i}][{j}]: 元素 {x} 的绝对值超过上限 {max_abs}"
                )
    return len(A), cols


def validate_vector(b, n, *, name="b", max_abs=None):
    """校验长度为 n 的整数列向量，返回 None。"""
    if not isinstance(b, list):
        raise MatrixShapeError(
            f"{name}: 向量必须是 list[int]，收到 {type(b).__name__}"
        )
    if len(b) != n:
        raise DimensionMismatchError(
            f"{name}: 向量长度 {len(b)} 与 A 的行数 {n} 不一致"
        )
    for i, x in enumerate(b):
        _check_int(x, f"{name}[{i}]")
        if max_abs is not None and abs(x) > max_abs:
            raise MatrixLimitError(
                f"{name}[{i}]: 元素 {x} 的绝对值超过上限 {max_abs}"
            )
