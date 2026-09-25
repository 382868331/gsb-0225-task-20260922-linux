"""带模同余条件的整数线性方程组求解（solve_mixed）。

同时要求
    A x = b            （普通整数等式，可缺省）
    C_i · x ≡ d_i (mod m_i)   （每行一条同余，m_i 为正整数）

方法（不另起炉灶，直接复用本包整数求解内核）：
  对每条 m_i > 1 的同余引入独立整数辅助变量 t_i，写成
      C_i · x - m_i * t_i = d_i
  与 A x = b 拼成增广整数线性方程组 Ã x̃ = b̃（x̃ = (x, t)），
  调用 solve_integer 求解；m_i = 1 的同余对任意整数 x 恒成立，
  作为无约束条件直接丢弃（不引入辅助变量）。

  增广系统的特解前 n 个坐标即原变量特解；增广零空间基投影到前 n 个
  坐标得到原变量解差的**生成集**——投影可能冗余（生成集而非基），
  故再对投影生成集做一次整数规范化：以生成向量为行的矩阵 G 求
  Smith 证书 U G V = D，rowZ(G) = rowZ(U G)（U 幺模），取 U G 的
  非零行即为该投影格的一组 Z-基。这不是“删掉辅助坐标留下冗余向量”，
  而是规范化后的基，张成的格恰为全部原变量解差。

  不可解时返回增广系统的可检查整数障碍（w·Ã 各分量被 modulus 整除
  而 w·b̃ 不然，或 w·Ã = 0 而 w·b̃ ≠ 0），w 为增广证书 U 的一行，
  可脱离本模块按算术独立复核。

  全程精确整数运算，无浮点、无有理解取整。
"""

from dataclasses import dataclass, field

from .errors import (
    DimensionMismatchError,
    MatrixLimitError,
    MatrixShapeError,
    ModulusError,
)
from .smith import smith, _DEFAULT_BUDGET
from .solve import IntegerSolveResult, Obstruction, solve_integer
from .validate import validate_matrix, validate_vector, _check_int

_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000


@dataclass(frozen=True)
class MixedSolveResult:
    """solve_mixed 的结果。

    consistent: True 时
      particular: 原变量特解 x0（长度 n 的整数向量）；
      nullspace_basis: 原变量解差格的一组 Z-基（投影后已整数规范化，
        每个向量长度 n）。全部原变量解为
            x = particular + Σ z_j * basis_j （z_j ∈ Z）。
    consistent: False 时 particular 为 None、nullspace_basis 为空，
      obstruction 为增广系统 Ã x̃ = b̃ 的可独立核验整数障碍
      （w·Ã 与 w·b̃ 的整除/零行证据）。

    n_vars: 原变量个数 n；n_aux: 实际引入的辅助变量个数
      （等于 m_i > 1 的同余条数）。
    dropped: 因 m_i = 1 而作为无约束丢弃的同余下标（对 C 的行号）。
    augmented_A / augmented_b: 实际求解的增广系统（供复核）。
    solve_result: 增广系统的完整 IntegerSolveResult，含 Smith 证书
      U, D, V 与全变换日志 transforms（证书链保留在增广系统上，
      可用 replay(augmented_A, transforms) 独立重放核验）。
    """

    consistent: bool
    particular: "list | None"
    nullspace_basis: list
    obstruction: "Obstruction | None"
    n_vars: int
    n_aux: int
    dropped: tuple
    augmented_A: list
    augmented_b: list
    solve_result: "IntegerSolveResult | None"
    transforms: tuple = field(default_factory=tuple)


def _validate_congruences(C, d, m, n):
    """校验 C/d/m 的形状与整数性，返回 C 的行数 q。"""
    q, cn = validate_matrix(
        C, name="C", max_rows=_MAX_ROWS, max_cols=_MAX_COLS, max_abs=_MAX_ABS,
    )
    if cn != n:
        raise DimensionMismatchError(
            f"C: 列数 {cn} 与变量个数 n={n} 不一致"
        )
    validate_vector(d, q, name="d", max_abs=_MAX_ABS)
    if not isinstance(m, list):
        raise MatrixShapeError(
            f"m: 模数列表必须是 list[int]，收到 {type(m).__name__}"
        )
    if len(m) != q:
        raise DimensionMismatchError(
            f"m: 长度 {len(m)} 与 C 的行数 {q} 不一致"
        )
    for i, mi in enumerate(m):
        _check_int(mi, f"m[{i}]")
        if mi < 1:
            raise ModulusError(
                f"m[{i}]: 模数必须为正整数，收到 {mi}"
            )
        if mi > _MAX_ABS:
            raise MatrixLimitError(
                f"m[{i}]: 模数 {mi} 的绝对值超过上限 {_MAX_ABS}"
            )
    return q


def _matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def _normalize_lattice(generators, n, budget):
    """把投影生成集规范化为 Z-基。

    generators: 整数向量列表（长度 n），张成格 L。
    返回 L 的一组 Z-基（向量列表）。空生成集返回空基。
    原理：G 以生成向量为行，smith(G) 得 U G V = D，U 幺模故
    rowZ(G) = rowZ(U G)，U G 的非零行即 L 的基。
    """
    if not generators:
        return []
    G = [list(g) for g in generators]
    cert = smith(G, budget=budget)
    U, rank = cert.U, cert.rank
    UG = [[sum(U[i][k] * G[k][j] for k in range(len(G)))
           for j in range(n)] for i in range(len(G))]
    basis = [UG[i] for i in range(rank)]
    for z in basis:  # 防御性核验：规范化不得产生零基向量
        if all(v == 0 for v in z):
            raise AssertionError("内部错误：规范化产生零基向量")
    return basis


def solve_mixed(A, b, C, d, m, *, budget=_DEFAULT_BUDGET):
    """求解 A x = b 且 C_i·x ≡ d_i (mod m_i) 的整数解。

    参数:
      A, b: 普通等式（A 为 p×n 的 list[list[int]] 或 None 表示无等式；
            b 相应为长度 p 的 list[int] 或 None）；
      C, d, m: 同余条件（C 为 q×n，d、m 长度 q；m_i 为正整数，
            m_i = 1 视为无约束）；C 为 None 表示无同余；
      budget: Smith 消元步数上限（耗尽抛 BudgetExhaustedError，
            表示搜索未完成，与无解结论无关）。

    约束: 增广系统行数 p + q'、列数 n + q' 均不得超过 8
      （q' 为 m_i > 1 的同余条数），元素与模数绝对值 ≤ 10000。

    返回 MixedSolveResult。非法输入抛 InvalidMatrixError 子类，
    模数非正抛 ModulusError。全程精确整数运算。
    """
    # ---- 确定变量个数 n 并校验等式部分 ----
    if A is None:
        if b is not None:
            raise DimensionMismatchError("A 为 None 时 b 必须为 None")
        A_rows = []
        n = None
    else:
        p, n = validate_matrix(
            A, name="A", max_rows=_MAX_ROWS, max_cols=_MAX_COLS,
            max_abs=_MAX_ABS,
        )
        validate_vector(b, p, name="b", max_abs=_MAX_ABS)
        A_rows = [list(row) for row in A]
        b = list(b)

    if C is None:
        if d is not None or m is not None:
            raise DimensionMismatchError("C 为 None 时 d、m 必须为 None")
        C_rows, d_list, m_list = [], [], []
    else:
        if n is None:
            # 无普通等式时，变量个数由 C 的列数确定。
            _, n = validate_matrix(
                C, name="C", max_rows=_MAX_ROWS, max_cols=_MAX_COLS,
                max_abs=_MAX_ABS,
            )
        C_rows = [list(row) for row in C]
        _validate_congruences(C, d, m, n)
        d_list, m_list = list(d), list(m)

    if n is None:
        raise MatrixShapeError("A 与 C 不能同时为 None（无法确定变量个数）")

    # ---- 组装增广系统：每条 m_i > 1 的同余引入独立辅助变量 t_i ----
    kept = [i for i in range(len(C_rows)) if m_list[i] > 1]
    dropped = tuple(i for i in range(len(C_rows)) if m_list[i] == 1)
    n_aux = len(kept)
    aug_rows = len(A_rows) + n_aux
    aug_cols = n + n_aux
    if aug_rows > _MAX_ROWS:
        raise MatrixLimitError(
            f"增广系统行数 {aug_rows}（等式 {len(A_rows)} + 同余 {n_aux}）"
            f"超过上限 {_MAX_ROWS}"
        )
    if aug_cols > _MAX_COLS:
        raise MatrixLimitError(
            f"增广系统列数 {aug_cols}（变量 {n} + 辅助 {n_aux}）"
            f"超过上限 {_MAX_COLS}"
        )

    aug_A = [row[:] + [0] * n_aux for row in A_rows]
    aug_b = list(b) if A is not None else []
    for k, i in enumerate(kept):
        row = C_rows[i][:] + [0] * n_aux
        row[n + k] = -m_list[i]
        aug_A.append(row)
        aug_b.append(d_list[i])

    if not aug_A:  # 既无等式也无有效同余：所有整数向量都是解
        return MixedSolveResult(
            consistent=True, particular=[0] * n,
            nullspace_basis=[[1 if j == r else 0 for j in range(n)]
                             for r in range(n)],
            obstruction=None, n_vars=n, n_aux=0, dropped=dropped,
            augmented_A=[], augmented_b=[], solve_result=None,
            transforms=(),
        )

    # ---- 调用现有整数求解内核 ----
    res = solve_integer(aug_A, aug_b, budget=budget)

    if not res.consistent:
        return MixedSolveResult(
            consistent=False, particular=None, nullspace_basis=[],
            obstruction=res.obstruction, n_vars=n, n_aux=n_aux,
            dropped=dropped, augmented_A=aug_A, augmented_b=aug_b,
            solve_result=res, transforms=tuple(res.transforms),
        )

    # ---- 投影回原变量空间并整数规范化 ----
    particular = res.particular[:n]
    generators = [z[:n] for z in res.nullspace_basis]
    basis = _normalize_lattice(generators, n, budget)

    # 精确回代核验：特解与每个基向量都满足原约束。
    if A is not None and _matvec(A, particular) != list(b):
        raise AssertionError("内部错误：特解不满足 A x = b")
    for i in kept:
        lhs = sum(c * x for c, x in zip(C_rows[i], particular))
        if (lhs - d_list[i]) % m_list[i] != 0:
            raise AssertionError("内部错误：特解不满足同余条件")
    for z in basis:
        if A is not None and any(v != 0 for v in _matvec(A, z)):
            raise AssertionError("内部错误：基向量不在 A 的零空间中")
        for i in kept:
            lhs = sum(c * x for c, x in zip(C_rows[i], z))
            if lhs % m_list[i] != 0:
                raise AssertionError("内部错误：基向量不满足齐次同余")

    return MixedSolveResult(
        consistent=True, particular=particular, nullspace_basis=basis,
        obstruction=None, n_vars=n, n_aux=n_aux, dropped=dropped,
        augmented_A=aug_A, augmented_b=aug_b,
        solve_result=res, transforms=tuple(res.transforms),
    )

