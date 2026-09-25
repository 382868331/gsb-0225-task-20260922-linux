"""整数线性方程组 + 模同余条件的混合求解（solve_mixed）。

问题：求整数向量 x（长度 n）同时满足
    A x = b                       （普通整数等式，可为 0 行）
    C_i x ≡ d_i  (mod m_i)        （每行一条同余，m_i 为正整数）

方法（不另搭求解内核）：
  每条 m_i > 1 的同余引入一个独立整数辅助变量 t_i，写成等式
      C_i x - m_i t_i = d_i
  与原等式拼成增广系统 A_aug x_hat = b_aug（x_hat = (x, t)），直接调用
  现有整数求解内核 solve_integer。m_i = 1 的同余对任何整数 x 恒成立，
  作为无约束丢弃，不引入辅助变量。

投影回原变量空间：
  特解取 x_hat0 的前 n 个坐标。增广零空间基向量的前 n 个坐标构成原
  变量解差格的生成集——投影是线性满射，生成集的像生成像格，不会遗漏；
  但可能冗余（维数高于投影格秩），故对投影生成集再做一次整数规范化：
  以生成向量为行组成矩阵求 Smith 证书，U2*G 的非零行即投影格的一组
  Z-基（U2 幺模，行格不变；非零行线性无关且生成同一行格）。不删除
  坐标直接当基，也不做任何有理运算/取整。

不可解时透传增广系统的整数障碍证据（w*A_aug 全被模数整除而
w*b_aug 余非零，或 w*A_aug=0 而 w*b_aug≠0），可脱离本模块按算术核验。
"""

from dataclasses import dataclass, field

from .errors import (
    DimensionMismatchError,
    MatrixLimitError,
    MatrixShapeError,
)
from .smith import smith, _DEFAULT_BUDGET
from .solve import Obstruction, solve_integer, _matvec
from .validate import validate_matrix, validate_vector, _check_int

_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000


@dataclass(frozen=True)
class MixedSolveResult:
    """solve_mixed 的结果。

    consistent: True 时 particular 为原变量空间的一个整数特解（长度 n），
    basis 为原变量解差格的一组 Z-基（每个向量长度 n）；全部解为
        x = particular + Σ z_j basis_j （z_j ∈ Z）。
    consistent: False 时 particular 为 None、basis 为空，obstruction 为
    增广系统上的可独立验算整数障碍证据。

    证书字段（保留增广变换链）：
      n_vars / n_aux: 原变量数 / 实际引入的辅助变量数；
      active: 被保留的同余下标（m_i > 1 者，按原顺序）；
      A_aug, b_aug: 实际求解的增广系统；
      U, D, V, c, transforms: 增广系统 Smith/求解证书
        （U*A_aug*V=D，transforms 为完整初等变换日志，可 replay 重放）；
      norm_transforms: 投影生成集规范化（第二次 Smith）的变换日志。
    """

    consistent: bool
    particular: "list | None"
    basis: list
    obstruction: "Obstruction | None"
    n_vars: int
    n_aux: int
    active: tuple
    A_aug: list
    b_aug: list
    U: list
    D: list
    V: list
    c: list
    transforms: tuple = field(default_factory=tuple)
    norm_transforms: tuple = field(default_factory=tuple)


def _validate_optional_matrix(A, *, name):
    """校验允许为空的矩阵块，返回 (rows, cols_or_None)。

    空 list 表示“没有该组条件”，行数为 0、列数未知（None）。
    """
    if not isinstance(A, list):
        raise MatrixShapeError(
            f"{name}: 矩阵必须是 list[list[int]]，收到 {type(A).__name__}"
        )
    if len(A) == 0:
        return 0, None
    rows, cols = validate_matrix(
        A, name=name,
        max_rows=_MAX_ROWS, max_cols=_MAX_COLS, max_abs=_MAX_ABS,
    )
    return rows, cols


def _validate_moduli(m, p):
    """校验模数列表：长度 p，每项为正整数（拒绝 bool/float），≤ 上限。"""
    if not isinstance(m, list):
        raise MatrixShapeError(
            f"m: 模数必须是 list[int]，收到 {type(m).__name__}"
        )
    if len(m) != p:
        raise DimensionMismatchError(
            f"m: 模数个数 {len(m)} 与 C 的行数 {p} 不一致"
        )
    for i, mi in enumerate(m):
        _check_int(mi, f"m[{i}]")
        if mi <= 0:
            raise ValueError(
                f"m[{i}]: 模数必须为正整数，收到 {mi}"
            )
        if mi > _MAX_ABS:
            raise MatrixLimitError(
                f"m[{i}]: 模数 {mi} 超过上限 {_MAX_ABS}"
            )


def _normalize_lattice(gens, n, budget):
    """对投影生成集做整数规范化，返回 (basis, norm_transforms)。

    gens 为生成同一 Z-子格的整数向量（长度 n，可能冗余）。以其为行组成
    矩阵 G 求 Smith 证书 U2*G*V2=D2；U2 幺模 ⇒ U2*G 与 G 行格相同，其
    非零行线性无关，构成该格的一组 Z-基。
    """
    G = [g[:] for g in gens]
    cert = smith(G, budget=budget)
    U2 = cert.U
    P = [[sum(U2[i][t] * G[t][j] for t in range(len(G)))
          for j in range(n)] for i in range(len(G))]
    basis = [row for row in P if any(v != 0 for v in row)]
    return basis, cert.transforms


def solve_mixed(A, b, C, d, m, *, budget=_DEFAULT_BUDGET):
    """求解 A x = b 且 C_i x ≡ d_i (mod m_i) 的整数解。

    参数:
      A: list[list[int]] 或 []（无普通等式），行数 ≤ 8；
      b: list[int]，长度等于 A 的行数；
      C: list[list[int]] 或 []（无同余条件），每行长度 n；
      d: list[int]，长度等于 C 的行数；
      m: list[int]，长度等于 C 的行数，每个模数为正整数；
         m_i = 1 视为无约束（不引入辅助变量）；
      budget: Smith 消元步数上限（耗尽抛 BudgetExhaustedError，只表示
      搜索未完成，不是无解结论）。

    增广系统（等式行 + 每个保留同余一行；原变量 + 每个保留同余一个辅助
    变量）的行数、列数均不得超过 8，否则抛 MatrixLimitError。
    元素/右端/模数绝对值上限 10000。返回 MixedSolveResult。
    """
    m0, n_a = _validate_optional_matrix(A, name="A")
    validate_vector(b, m0, name="b", max_abs=_MAX_ABS)
    p, n_c = _validate_optional_matrix(C, name="C")
    validate_vector(d, p, name="d", max_abs=_MAX_ABS)
    _validate_moduli(m, p)

    if n_a is None and n_c is None:
        raise MatrixShapeError(
            "A 与 C 不能同时为空：无法确定变量个数 n"
        )
    n = n_a if n_a is not None else n_c
    if n_a is not None and n_c is not None and n_a != n_c:
        raise DimensionMismatchError(
            f"C: 列数 {n_c} 与 A 的列数 {n_a} 不一致"
        )

    # m_i = 1 的同余恒成立，作为无约束丢弃。
    active = tuple(i for i in range(p) if m[i] > 1)
    n_aux = len(active)

    rows_aug = m0 + n_aux
    cols_aug = n + n_aux
    if rows_aug > _MAX_ROWS:
        raise MatrixLimitError(
            f"增广系统行数 {rows_aug}（等式 {m0} 行 + 保留同余 {n_aux} 条）"
            f"超过上限 {_MAX_ROWS}"
        )
    if cols_aug > _MAX_COLS:
        raise MatrixLimitError(
            f"增广系统列数 {cols_aug}（原变量 {n} 个 + 辅助变量 {n_aux} 个）"
            f"超过上限 {_MAX_COLS}"
        )

    # 组装增广系统 A_aug x_hat = b_aug，x_hat = (x, t)。
    A_aug = [row[:] + [0] * n_aux for row in A]
    b_aug = list(b)
    for k, i in enumerate(active):
        row = C[i][:] + [0] * n_aux
        row[n + k] = -m[i]
        A_aug.append(row)
        b_aug.append(d[i])
    if not A_aug:  # 既无等式也无保留同余：任意整数向量都是解
        A_aug = [[0] * n]
        b_aug = [0]

    res = solve_integer(A_aug, b_aug, budget=budget)

    if not res.consistent:
        return MixedSolveResult(
            consistent=False, particular=None, basis=[],
            obstruction=res.obstruction,
            n_vars=n, n_aux=n_aux, active=active,
            A_aug=A_aug, b_aug=b_aug,
            U=res.U, D=res.D, V=res.V, c=res.c,
            transforms=res.transforms, norm_transforms=(),
        )

    # 投影回原变量空间。
    x0 = res.particular[:n]
    gens = [z[:n] for z in res.nullspace_basis]
    if gens:
        basis, norm_transforms = _normalize_lattice(gens, n, budget)
    else:
        basis, norm_transforms = [], ()

    # 精确核验（全部整数运算，不满足即内部错误，不静默）。
    if m0 and _matvec(A, x0) != list(b):
        raise AssertionError("内部错误：混合系统特解不满足 A x = b")
    for i in range(p):
        lhs = sum(C[i][j] * x0[j] for j in range(n))
        if (lhs - d[i]) % m[i] != 0:
            raise AssertionError("内部错误：特解不满足同余条件")
    for z in basis:
        if m0 and any(v != 0 for v in _matvec(A, z)):
            raise AssertionError("内部错误：基向量不满足 A z = 0")
        for i in range(p):
            if sum(C[i][j] * z[j] for j in range(n)) % m[i] != 0:
                raise AssertionError("内部错误：基向量不满足齐次同余")

    return MixedSolveResult(
        consistent=True, particular=x0, basis=basis,
        obstruction=None,
        n_vars=n, n_aux=n_aux, active=active,
        A_aug=A_aug, b_aug=b_aug,
        U=res.U, D=res.D, V=res.V, c=res.c,
        transforms=res.transforms, norm_transforms=norm_transforms,
    )
