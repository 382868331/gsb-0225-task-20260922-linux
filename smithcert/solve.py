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
from fractions import Fraction
from itertools import combinations
from math import isfinite

from .errors import (
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixShapeError,
    NonFiniteNumberError,
)
from .smith import smith, _DEFAULT_BUDGET
from .validate import validate_matrix, validate_vector

_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000

BOUNDED_STATUS_OPTIMAL = "optimal"
BOUNDED_STATUS_FEASIBLE = "feasible"
BOUNDED_STATUS_INFEASIBLE = "infeasible"
BOUNDED_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
BOUNDED_STATUS_UNSUPPORTED = "unsupported"


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


@dataclass(frozen=True)
class BoundedSolveResult:
    """有限整数上下界内 A x=b 的求解结果。

    status:
      optimal                 已证明 L1 最小且并列时字典序最小；
      feasible                找到可行解，但预算内未证明最优；
      infeasible              已证明界内无整数可行解；
      budget_exhausted        预算耗尽且尚未找到可行解；
      unsupported             输入含非有限上下界，本题不求解无界问题。
    solution 为最优/当前可行解；其余状态为 None。objective 为 L1 范数。
    coefficients 是该解在通解 x=particular+Σzⱼbasisⱼ 中的整数系数。
    equation_check 与 bounds_check 保存证书式精确核验结果。
    """

    status: str
    feasible: bool
    optimal: bool
    solution: "list | None"
    objective: "int | None"
    lower: list
    upper: list
    Ax: "list | None"
    b: list
    equation_check: "bool | None"
    bounds_check: "bool | None"
    particular: "list | None"
    nullspace_basis: list
    coefficients: "list | None"
    obstruction: "Obstruction | None"
    nodes_visited: int
    budget: int
    reason: "str | None" = None


def _matvec(A, x):
    """整数矩阵 × 整数列向量。"""
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def _row_mat(w, A):
    """整数行向量 w × 整数矩阵 A。"""
    n = len(A[0])
    return [sum(w[i] * A[i][j] for i in range(len(A))) for j in range(n)]


def _validate_bound_vector(values, n, name):
    """校验上下界；None 与 ±∞表示无界，会由上层返回 unsupported。"""
    if not isinstance(values, list):
        raise MatrixShapeError(
            f"{name}: 上下界必须是 list，收到 {type(values).__name__}"
        )
    if len(values) != n:
        raise DimensionMismatchError(
            f"{name}: 长度 {len(values)} 与变量数 {n} 不一致"
        )
    normalized = []
    unbounded = False
    for i, value in enumerate(values):
        if value is None:
            normalized.append(None)
            unbounded = True
            continue
        if isinstance(value, bool):
            raise IntegerRequiredError(
                f"{name}[{i}]: bool 不是合法整数界（拒绝 True/False）"
            )
        if isinstance(value, int):
            normalized.append(value)
            continue
        if isinstance(value, float):
            if value != value:
                raise NonFiniteNumberError(
                    f"{name}[{i}]: 整数界不能是 NaN；无界请显式使用 None 或 ±Infinity"
                )
            if not isfinite(value):
                normalized.append(None)
                unbounded = True
                continue
            raise IntegerRequiredError(
                f"{name}[{i}]: 整数界不能是有限 float {value!r}"
            )
        raise IntegerRequiredError(
            f"{name}[{i}]: 整数界必须是 int/None，收到 "
            f"{type(value).__name__}: {value!r}"
        )
    return normalized, unbounded


def _bounded_unsupported(lower, upper, b, budget, reason):
    return BoundedSolveResult(
        status=BOUNDED_STATUS_UNSUPPORTED,
        feasible=False, optimal=False, solution=None, objective=None,
        lower=lower, upper=upper, Ax=None, b=b,
        equation_check=None, bounds_check=None,
        particular=None, nullspace_basis=[], coefficients=None,
        obstruction=None, nodes_visited=0, budget=budget, reason=reason,
    )


def _bounded_result(status, solution, objective, lower, upper, Ax, b,
                    particular, basis, coefficients, obstruction,
                    nodes_visited, budget):
    feasible = status in (
        BOUNDED_STATUS_OPTIMAL, BOUNDED_STATUS_FEASIBLE,
    )
    equation_check = Ax == b if feasible else None
    bounds_check = (
        all(lo <= v <= hi for v, lo, hi in zip(solution, lower, upper))
        if feasible else None
    )
    return BoundedSolveResult(
        status=status, feasible=feasible,
        optimal=status == BOUNDED_STATUS_OPTIMAL,
        solution=solution, objective=objective, lower=lower, upper=upper,
        Ax=Ax, b=b, equation_check=equation_check,
        bounds_check=bounds_check, particular=particular,
        nullspace_basis=basis, coefficients=coefficients,
        obstruction=obstruction, nodes_visited=nodes_visited,
        budget=budget,
    )


class _SearchBudget(Exception):
    """内部分支预算信号，捕获后转换成公开状态，不当作数学结论。"""


def _solve_square_fraction(M, rhs):
    n = len(M)
    X = [row[:] + [rhs[i]] for i, row in enumerate(M)]
    for col in range(n):
        pivot = next((r for r in range(col, n)
                      if X[r][col] != 0), None)
        if pivot is None:
            return None
        X[col], X[pivot] = X[pivot], X[col]
        piv = X[col][col]
        for j in range(col + 1, n + 1):
            X[col][j] /= piv
        X[col][col] = Fraction(1)
        for r in range(n):
            if r == col or X[r][col] == 0:
                continue
            factor = X[r][col]
            for j in range(col, n + 1):
                X[r][j] -= factor * X[col][j]
    return [X[i][n] for i in range(n)]


def _real_projection_bounds(x0, basis, lower, upper):
    """枚举有界多面体顶点，求每个零空间系数的精确有理投影界。"""
    k = len(basis)
    constraints = []
    for i, p in enumerate(x0):
        row = [Fraction(z[i]) for z in basis]
        constraints.append((row, Fraction(upper[i] - p)))
        constraints.append(([(-v) for v in row], Fraction(p - lower[i])))

    minima = [None] * k
    maxima = [None] * k
    for indices in combinations(range(len(constraints)), k):
        M = [constraints[t][0][:] for t in indices]
        rhs = [constraints[t][1] for t in indices]
        z = _solve_square_fraction(M, rhs)
        if z is None:
            continue
        if all(sum(a * v for a, v in zip(row, z)) <= bound
               for row, bound in constraints):
            for j, value in enumerate(z):
                minima[j] = value if minima[j] is None else min(minima[j], value)
                maxima[j] = value if maxima[j] is None else max(maxima[j], value)
    return minima, maxima


def _check_domains(x0, basis, lower, upper, domains, remaining):
    """检查一个分支盒，并计入搜索节点预算。

    初始精确实投影已经是不漏解的有限盒。多变量线性等式的一般区间传播若
    独立套用其他变量极值，会把相关约束误判为空；这里只保持精确投影盒，
    完整性由分支穷尽与单点 Ax=b、边界检查保证。
    """
    remaining[0] -= 1
    if remaining[0] < 0:
        raise _SearchBudget()
    if any(lo > hi for lo, hi in domains):
        return None, remaining[0]
    return domains, remaining[0]


def _compose(x0, basis, z):
    return [p + sum(coeff * vector[i] for coeff, vector in zip(z, basis))
            for i, p in enumerate(x0)]


def _objective_lower_bound(x0, basis, domains):
    total = Fraction(0)
    lex_lower = []
    lex_upper = []
    for i, p in enumerate(x0):
        minimum = maximum = p
        for j, z in enumerate(basis):
            coefficient = z[i]
            if coefficient > 0:
                minimum += coefficient * domains[j][0]
                maximum += coefficient * domains[j][1]
            elif coefficient < 0:
                minimum += coefficient * domains[j][1]
                maximum += coefficient * domains[j][0]
        if minimum <= 0 <= maximum:
            abs_lower = Fraction(0)
        else:
            abs_lower = min(abs(minimum), abs(maximum))
        total += abs_lower
        lex_lower.append(minimum)
        lex_upper.append(maximum)
    return -((-total.numerator) // total.denominator), lex_lower, lex_upper


def _search_bounded(sol, lower, upper, budget):
    x0 = sol.particular
    basis = sol.nullspace_basis
    k = len(basis)
    nodes_visited = 0
    remaining = [budget]

    if k == 0:
        if all(lo <= v <= hi for v, lo, hi in zip(x0, lower, upper)):
            return (BOUNDED_STATUS_OPTIMAL, x0,
                    sum(abs(v) for v in x0), [], 1)
        return (BOUNDED_STATUS_INFEASIBLE, None, None, None, 1)

    minima, maxima = _real_projection_bounds(x0, basis, lower, upper)
    if any(value is None for value in minima + maxima):
        return (BOUNDED_STATUS_INFEASIBLE, None, None, None,
                nodes_visited)

    domains = []
    for lo, hi in zip(minima, maxima):
        integer_low = -((-lo.numerator) // lo.denominator)
        integer_high = hi.numerator // hi.denominator
        if integer_low > integer_high:
            return (BOUNDED_STATUS_INFEASIBLE, None, None, None,
                    nodes_visited)
        domains.append((integer_low, integer_high))

    stack = [domains]
    best_x = None
    best_obj = None
    best_z = None

    while stack:
        nodes_visited += 1
        if nodes_visited > budget:
            status = (BOUNDED_STATUS_FEASIBLE if best_x is not None
                      else BOUNDED_STATUS_BUDGET_EXHAUSTED)
            return (status, best_x, best_obj, best_z, nodes_visited)

        current = stack.pop()
        try:
            propagated, remaining[0] = _check_domains(
                x0, basis, lower, upper, current, remaining,
            )
        except _SearchBudget:
            status = (BOUNDED_STATUS_FEASIBLE if best_x is not None
                      else BOUNDED_STATUS_BUDGET_EXHAUSTED)
            return (status, best_x, best_obj, best_z, nodes_visited)
        if propagated is None:
            continue

        if all(lo == hi for lo, hi in propagated):
            z = [lo for lo, _ in propagated]
            x = _compose(x0, basis, z)
            if all(lo <= v <= hi for v, lo, hi in zip(x, lower, upper)):
                objective = sum(abs(v) for v in x)
                if (best_x is None or objective < best_obj
                        or (objective == best_obj and tuple(x) < tuple(best_x))):
                    best_x, best_obj, best_z = x, objective, z
            continue

        lb, lex_lower, _lex_upper = _objective_lower_bound(
            x0, basis, propagated,
        )
        if best_obj is not None:
            if lb > best_obj or (lb == best_obj
                                 and tuple(lex_lower) >= tuple(best_x)):
                continue

        widths = [hi - lo for lo, hi in propagated]
        branch = max(range(k), key=lambda j: widths[j])
        lo, hi = propagated[branch]
        middle = lo + (hi - lo) // 2
        left = propagated[:]
        right = propagated[:]
        left[branch] = (lo, middle)
        right[branch] = (middle + 1, hi)
        stack.append(right)
        stack.append(left)

    if best_x is None:
        return (BOUNDED_STATUS_INFEASIBLE, None, None, None,
                nodes_visited)
    return (BOUNDED_STATUS_OPTIMAL, best_x, best_obj, best_z,
            nodes_visited)


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


def solve_bounded_integer(A, b, lower, upper, *, budget=_DEFAULT_BUDGET):
    """在逐变量有限整数盒中最小化 ||x||_1，并列时取字典序最小解。

    lower/upper 必须是长度 n 的 list[int]。None 或 ±float("inf") 表示
    该方向无界；本题明确不把无界问题伪装成可行/不可行，而是返回
    status="unsupported"。预算只计有界分支节点；耗尽时区分
    "feasible"（已有可行解但未证最优）与 "budget_exhausted"（无当前解）。
    """
    m, n = validate_matrix(
        [row[:] for row in A], name="A",
        max_rows=_MAX_ROWS, max_cols=6, max_abs=_MAX_ABS,
    )
    validate_vector(b, m, name="b", max_abs=_MAX_ABS)
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget 必须是 int，收到 {type(budget).__name__}")
    if budget <= 0:
        raise ValueError(f"budget 必须为正整数，收到 {budget}")

    checked_lower, lower_unbounded = _validate_bound_vector(
        lower, n, "lower",
    )
    checked_upper, upper_unbounded = _validate_bound_vector(
        upper, n, "upper",
    )
    b_copy = list(b)
    if lower_unbounded or upper_unbounded:
        return _bounded_unsupported(
            checked_lower, checked_upper, b_copy, budget,
            "存在非有限整数界；无界输入在本题明确 unsupported",
        )

    reversed_bounds = [
        i for i, (lo, hi) in enumerate(zip(checked_lower, checked_upper))
        if lo > hi
    ]
    if reversed_bounds:
        return _bounded_result(
            BOUNDED_STATUS_INFEASIBLE, None, None,
            checked_lower, checked_upper, None, b_copy,
            None, [], None, None, 0, budget,
        )

    integer_solution = solve_integer(A, b_copy)
    if not integer_solution.consistent:
        return _bounded_result(
            BOUNDED_STATUS_INFEASIBLE, None, None,
            checked_lower, checked_upper, None, b_copy,
            None, [], None, integer_solution.obstruction, 0, budget,
        )

    status, solution, objective, coefficients, nodes = _search_bounded(
        integer_solution, checked_lower, checked_upper, budget,
    )
    Ax = _matvec(A, solution) if solution is not None else None
    return _bounded_result(
        status, solution, objective, checked_lower, checked_upper, Ax,
        b_copy, integer_solution.particular,
        integer_solution.nullspace_basis, coefficients,
        None, nodes, budget,
    )
