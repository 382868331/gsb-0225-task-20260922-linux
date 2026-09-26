"""有限整数上下界内的最优整数解。

先由 solve_integer() 取得整数可解性、特解 p 与整数零空间 Z-基 B。
所有整数解恰为 x = p + B z；本模块在这个仿射整数格与有限盒
L_i <= x_i <= U_i 的交集中寻找最优解。

搜索不对 z 猜测一个小系数盒，而是在原始变量 x 的有限域上做：
  1. 逐线性方程的区间投影收缩（只取数学上可证明的整数上下界）；
  2. 对剩余有限整数区间做不交的二叉划分；
  3. 用 L1 范数下界和字典序下界剪枝。
每个分支节点的域都是有限域，二叉划分互不遗漏，因此完整结束时可证明
最优或不可行；预算耗尽只返回未完成状态，绝不冒充不可行。
"""

from dataclasses import dataclass

from .errors import DimensionMismatchError, MatrixShapeError
from .solve import solve_integer, _matvec
from .validate import validate_matrix, validate_vector, _check_int

_MAX_ROWS = 8
_MAX_VARS = 6
_MAX_ABS = 10_000
_DEFAULT_BUDGET = 100_000

OPTIMAL = "optimal"
FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
UNSUPPORTED = "unsupported"
BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class BoundedSolveResult:
    """有限界整数求解结果。"""

    status: str
    solution: "list | None"
    l1_norm: "int | None"
    particular: "list | None"
    nullspace_basis: list
    residual: "list | None"
    bounds_check: "list | None"
    obstruction: object
    nodes_evaluated: int
    budget_used: int
    budget_total: int
    reason: str = ""


class _BudgetStop(Exception):
    """内部预算控制信号，不在公开 API 中暴露为异常。"""


def _ceil_div(a, b):
    """b > 0 时的精确整数 ceil(a / b)。"""
    return -((-a) // b)


def validate_bounded_matrix(A):
    return validate_matrix(
        A, name="A", max_rows=_MAX_ROWS, max_cols=_MAX_VARS, max_abs=_MAX_ABS
    )


def _validate_bounds(bounds, n):
    if bounds is None:
        return None, "bounds=None 表示无界问题；本题有界求解只支持有限上下界"
    if not isinstance(bounds, list):
        raise MatrixShapeError(
            f"bounds: 必须是 list[(lower, upper)]，收到 {type(bounds).__name__}"
        )
    if len(bounds) != n:
        raise DimensionMismatchError(
            f"bounds: 界数 {len(bounds)} 与变量数 {n} 不一致"
        )

    finite = []
    unbounded = False
    for i, item in enumerate(bounds):
        if item is None:
            unbounded = True
            continue
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise MatrixShapeError(
                f"bounds[{i}]: 必须是长度为 2 的 (lower, upper)，收到 {item!r}"
            )
        lower, upper = item
        if lower is None or upper is None:
            unbounded = True
            continue
        _check_int(lower, f"bounds[{i}][0]")
        _check_int(upper, f"bounds[{i}][1]")
        if lower > upper:
            raise MatrixShapeError(
                f"bounds[{i}]: 下界 {lower} 大于上界 {upper}"
            )
        finite.append((lower, upper))

    if unbounded:
        return None, "至少一个变量没有有限整数下界或上界；无界输入返回 unsupported"
    return finite, ""


def _project_rows(A, b, domains, remaining):
    """用每行 Ax=b 的整数区间投影反复收缩各变量域。"""
    n = len(domains)
    while True:
        changed = False
        remaining[0] -= 1
        if remaining[0] < 0:
            raise _BudgetStop

        for row, rhs in zip(A, b):
            for j, coef in enumerate(row):
                if coef == 0:
                    continue
                low_sum = 0
                high_sum = 0
                for k, a in enumerate(row):
                    if k == j:
                        continue
                    lo, hi = domains[k]
                    if a >= 0:
                        low_sum += a * lo
                        high_sum += a * hi
                    else:
                        low_sum += a * hi
                        high_sum += a * lo

                numerator_low = rhs - high_sum
                numerator_high = rhs - low_sum
                if coef > 0:
                    new_low = _ceil_div(numerator_low, coef)
                    new_high = numerator_high // coef
                else:
                    positive = -coef
                    new_low = _ceil_div(-numerator_high, positive)
                    new_high = (-numerator_low) // positive

                old_low, old_high = domains[j]
                new_low = max(old_low, new_low)
                new_high = min(old_high, new_high)
                if new_low > new_high:
                    return None
                if (new_low, new_high) != (old_low, old_high):
                    domains[j] = (new_low, new_high)
                    changed = True
        if not changed:
            return domains


def solve_bounded_integer(A, b, bounds, *, budget=_DEFAULT_BUDGET):
    """在有限逐变量界内求 L1 最小、再字典序最小的整数解。"""
    m, n = validate_bounded_matrix(A)
    validate_vector(b, m, name="b", max_abs=_MAX_ABS)
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget 必须是 int，收到 {type(budget).__name__}")
    if budget <= 0:
        raise ValueError(f"budget 必须为正整数，收到 {budget}")

    finite_bounds, unsupported_reason = _validate_bounds(bounds, n)
    if finite_bounds is None:
        return BoundedSolveResult(
            status=UNSUPPORTED,
            solution=None,
            l1_norm=None,
            particular=None,
            nullspace_basis=[],
            residual=None,
            bounds_check=None,
            obstruction=None,
            nodes_evaluated=0,
            budget_used=0,
            budget_total=budget,
            reason=unsupported_reason,
        )

    # 入参 budget 只度量有界分支搜索；Smith 证书使用其公开默认预算，
    # 避免小规模矩阵的构造变换耗尽留给完备性证明的预算。
    base = solve_integer(A, b)
    common = dict(
        particular=base.particular,
        nullspace_basis=base.nullspace_basis,
        obstruction=base.obstruction,
        budget_total=budget,
    )
    if not base.consistent:
        return BoundedSolveResult(
            status=INFEASIBLE,
            solution=None,
            l1_norm=None,
            residual=None,
            bounds_check=None,
            nodes_evaluated=0,
            budget_used=0,
            reason="无界整数方程组本身已由 Smith 障碍证书证明无解",
            **common,
        )

    domains0 = list(finite_bounds)
    best = None

    def within_bounds(x):
        return all(lo <= v <= hi for v, (lo, hi) in zip(x, finite_bounds))

    if within_bounds(base.particular):
        best = base.particular[:]

    remaining = [budget]
    nodes = 0

    def objective_key(x):
        return sum(abs(v) for v in x), tuple(x)

    def l1_lower(domains):
        total = 0
        for lo, hi in domains:
            if lo <= 0 <= hi:
                continue
            total += min(abs(lo), abs(hi))
        return total

    def can_beat(domains):
        if best is None:
            return True
        lb = l1_lower(domains)
        best_l1 = sum(abs(v) for v in best)
        if lb > best_l1:
            return False
        if lb < best_l1:
            return True
        componentwise_low = tuple(lo for lo, _ in domains)
        return componentwise_low < tuple(best)

    def search(initial_domains):
        nonlocal best, nodes
        stack = [initial_domains]
        while stack:
            remaining[0] -= 1
            if remaining[0] < 0:
                raise _BudgetStop
            nodes += 1

            domains = _project_rows(A, b, stack.pop(), remaining)
            if domains is None or not can_beat(domains):
                continue

            if all(lo == hi for lo, hi in domains):
                x = [lo for lo, _ in domains]
                if _matvec(A, x) == list(b) and (
                    best is None or objective_key(x) < objective_key(best)
                ):
                    best = x
                continue

            j = max(range(n), key=lambda idx: domains[idx][1] - domains[idx][0])
            lo, hi = domains[j]
            mid = (lo + hi) // 2
            upper_domains = domains[:]
            lower_domains = domains[:]
            upper_domains[j] = (mid + 1, hi)
            lower_domains[j] = (lo, mid)
            # 栈 LIFO：先压右半，再压左半，使左半优先被检查。
            stack.append(upper_domains)
            stack.append(lower_domains)

    completed = False
    try:
        search(domains0)
        completed = True
    except _BudgetStop:
        completed = False

    used = budget - max(remaining[0], 0)
    if best is not None and not completed:
        status = FEASIBLE
        reason = "找到可行解，但分支搜索预算耗尽，尚未证明全局最优"
    elif best is None and not completed:
        status = BUDGET_EXHAUSTED
        reason = "分支搜索预算耗尽，且当前尚未找到可行解；不表示无解"
    elif best is None:
        status = INFEASIBLE
        reason = "有限域区间收缩与完整分支划分均未留下可行整数点"
    else:
        status = OPTIMAL
        reason = "分支搜索完整结束；L1 范数与字典序剪枝均为精确整数界"

    solution = best[:] if best is not None else None
    residual = _matvec(A, solution) if solution is not None else None
    bounds_check = (
        [lo <= v <= hi for v, (lo, hi) in zip(solution, finite_bounds)]
        if solution is not None
        else None
    )
    l1 = sum(abs(v) for v in best) if best is not None else None
    return BoundedSolveResult(
        status=status,
        solution=solution,
        l1_norm=l1,
        residual=residual,
        bounds_check=bounds_check,
        nodes_evaluated=nodes,
        budget_used=used,
        reason=reason,
        **common,
    )
