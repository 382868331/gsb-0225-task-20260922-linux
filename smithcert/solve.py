"""整数线性方程组 A x = b 的可解判定、特解与整数零空间基。

直接使用 Smith 证书，不经过任何有理运算/取整：
  U A V = D（D 为 Smith 标准形）。令 c = U b、y = V^{-1} x，则
  A x = b  ⇔  D y = c，且 x = V y（V 幺模，整数双射）。

  - 对 i < rank：d_i y_i = c_i，有整数解当且仅当 d_i | c_i；
  - 对 rank ≤ i < m：D 的第 i 行为零，要求 c_i = 0；
  - 自由变量为 y_j (j ≥ rank)，零空间基即 V 的第 rank..n-1 列。

无解时返回可独立验算的整数“障碍向量” w（U 的某一行）：
  - 整除障碍：wA 的每个分量都是 d 的倍数，而 w·b 除以 d 余非零；
  - 零行障碍：wA = 0，而 w·b ≠ 0。
"""

from dataclasses import dataclass, field

from .smith import smith, _DEFAULT_BUDGET
from .validate import validate_matrix, validate_vector

_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000


@dataclass(frozen=True)
class Obstruction:
    """无解的整数证据（可脱离本模块独立按算术核验）。

    kind:
      "zero_row"     —— w*A == 0 但 w*b = wb != 0；
      "divisibility" —— w*A 的每个分量都是 modulus 的倍数，
                        但 wb 除以 modulus 的余数 remainder != 0。
    index: 触发障碍的 D 行号（即 U 的行号，仅作定位）。
    w:     障碍行向量（U 的一行，整数）。
    wA:    w*A（整数行向量，便于直接核验）。
    wb:    w*b（整数）。
    modulus, remainder: 整除障碍的模数与非零余数；zero_row 时为 1, wb。
    """

    kind: str
    index: int
    w: list
    wA: list
    wb: int
    modulus: int
    remainder: int


@dataclass(frozen=True)
class IntegerSolveResult:
    """solve_integer 的结果。

    consistent: True 时 particular 为一个整数特解，nullspace_basis 为
    整数零空间的一组基（每个元素是长度 n 的整数向量）；通解为
        x = particular + Σ z_j * basis_j （z_j ∈ Z）。
    consistent: False 时 particular 为 None、nullspace_basis 为空，
    obstruction 带可独立验算的证据。
    rank/invariants/U/D/V/transforms: 所用 Smith 证书的全部信息。
    """

    consistent: bool
    particular: "list | None"
    nullspace_basis: list
    obstruction: "Obstruction | None"
    rank: int
    invariants: list
    U: list
    D: list
    V: list
    c: list
    transforms: list = field(default_factory=tuple)


def _matvec(A, x):
    """整数矩阵 × 整数列向量。"""
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def _row_mat(w, A):
    """整数行向量 w × 整数矩阵 A。"""
    n = len(A[0])
    return [sum(w[i] * A[i][j] for i in range(len(A))) for j in range(n)]


def solve_integer(A, b, *, budget=_DEFAULT_BUDGET):
    """判定 A x = b 是否有整数解，求一个特解与整数零空间基。

    参数:
      A: list[list[int]]，m×n，m,n ≤ 8，元素绝对值 ≤ 10000；
      b: list[int]，长度 m，元素绝对值 ≤ 10000；
      budget: Smith 消元步数上限（耗尽抛 BudgetExhaustedError，
      那是“搜索未完成”，不是无解结论）。

    返回 IntegerSolveResult。输入形状/类型非法抛 InvalidMatrixError
    子类（信息含定位）；本函数无任何浮点运算。
    """
    m, n = validate_matrix(
        A, name="A",
        max_rows=_MAX_ROWS, max_cols=_MAX_COLS, max_abs=_MAX_ABS,
    )
    validate_vector(b, m, name="b", max_abs=_MAX_ABS)

    cert = smith(A, budget=budget)
    U, D, V = cert.U, cert.D, cert.V
    rank = cert.rank
    invariants = cert.invariants

    c = _matvec(U, b)  # 变换后的右端项

    # ---- 可解性：逐行检查 d_i | c_i 与零行 c_i = 0 ----
    obstruction = None
    for i in range(m):
        if i < rank:
            d = invariants[i]
            rem = c[i] % d  # d > 0，余数在 [0, d)
            if rem != 0:
                w = U[i][:]
                wA = _row_mat(w, A)
                obstruction = Obstruction(
                    kind="divisibility", index=i, w=w, wA=wA,
                    wb=c[i], modulus=d, remainder=rem,
                )
                break
        elif c[i] != 0:
            w = U[i][:]
            wA = _row_mat(w, A)
            # 正常情况下 wA 必须全 0（D 的第 i 行为零）；若不成立说明
            # 证书有内部问题，显式报错而不是静默。
            if any(x != 0 for x in wA):
                raise AssertionError("内部错误：零行障碍向量与 D 不一致")
            obstruction = Obstruction(
                kind="zero_row", index=i, w=w, wA=wA,
                wb=c[i], modulus=1, remainder=c[i],
            )
            break

    if obstruction is not None:
        return IntegerSolveResult(
            consistent=False, particular=None, nullspace_basis=[],
            obstruction=obstruction, rank=rank, invariants=invariants,
            U=U, D=D, V=V, c=c, transforms=cert.transforms,
        )

    # ---- 特解：y_i = c_i/d_i (i<rank)，其余 0；x = V y ----
    y = [0] * n
    for i in range(rank):
        if c[i] % invariants[i] != 0:  # 防御性检查，前面已保证
            raise AssertionError("内部错误：整除判断与特解构造矛盾")
        y[i] = c[i] // invariants[i]
    x0 = [sum(V[r][j] * y[j] for j in range(n)) for r in range(n)]

    # 精确回代核验 A x0 == b；不成立即内部错误。
    if _matvec(A, x0) != list(b):
        raise AssertionError("内部错误：构造的特解不满足 A x = b")

    # ---- 整数零空间基：V 的第 rank..n-1 列 ----
    basis = [[V[r][j] for r in range(n)] for j in range(rank, n)]
    # 每个基向量必须确实在核中（精确核验）。
    for z in basis:
        if any(v != 0 for v in _matvec(A, z)):
            raise AssertionError("内部错误：零空间基向量核验失败")

    return IntegerSolveResult(
        consistent=True, particular=x0, nullspace_basis=basis,
        obstruction=None, rank=rank, invariants=invariants,
        U=U, D=D, V=V, c=c, transforms=cert.transforms,
    )
