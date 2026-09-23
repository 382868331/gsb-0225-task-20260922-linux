"""整数矩阵 Smith 标准形（带初等变换证书）。

算法（整数消元 / pivot chasing，Kannan–Bachem 风格）：

维护工作矩阵 M 与行/列变换日志。处理第 k 个枢轴：
  1. 在子矩阵 M[k:, k:] 中取一个非零元，用行/列交换搬到 (k,k) 并转正。
  2. 反复扫描第 k 列、第 k 行（子块范围）：
       - 某元素不能被当前枢轴 d 整除时，用带余除法做一次“加整数倍”，
         把余数换到枢轴位置。余数绝对值严格变小，故必终止；
       - 全部整除后，用“加整数倍”把该行、该列其余元素清零。
  3. 关键的整除链步骤：扫描剩余子块 M[k+1:, k+1:]。若存在 M[r][c]
     不被 d 整除，把第 c 列加回第 k 列（此时 M[k][k] 仍为 d），令
     M[r][k]=M[r][c]，回到步骤 2——枢轴将再次严格变小。
     直至 d 整除整个剩余子块。归纳可得 d_k | d_{k+1}。
  4. 进入下一枢轴；子块全零则结束。

枢轴按 k=0,1,... 顺序放置，非零不变因子自然在最前、零在最后。
全程只做：交换、变号、加整数倍——累积的 U、V 是整数幺模阵，
U*A*V = D，det(U)=det(V)=±1。所有运算为 Python 大整数精确运算，
无浮点、无“有理结果取整”。
"""

from dataclasses import dataclass, field

from .errors import BudgetExhaustedError
from .transform import Transform, replay, checked_k
from .validate import validate_matrix

_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000
_DEFAULT_BUDGET = 100_000


@dataclass(frozen=True)
class SmithResult:
    """smith() 的结果。

    U: m×m 整数幺模阵；D: m×n 对角阵；V: n×n 整数幺模阵；U*A*V == D。
    invariants: 非零对角不变因子（升序，满足整除链）。
    rank: 秩。transforms: 全变换日志（可独立 replay 核验）。
    """

    U: list
    D: list
    V: list
    invariants: list
    rank: int
    transforms: list = field(default_factory=tuple)


class _Engine:
    """对工作矩阵 M 施加初等变换并同步记录日志。"""

    def __init__(self, M, m, n, budget):
        self.M = M
        self.m = m
        self.n = n
        self.log = []
        self.remaining = budget

    def _tick(self):
        if self.remaining <= 0:
            raise BudgetExhaustedError(
                "Smith 消元初等变换步数预算耗尽：搜索未完成（这不是“无解”结论；"
                "方程组无解由 solve_integer 返回 consistent=False 及余数证据）"
            )
        self.remaining -= 1

    def _record(self, t):
        self._tick()
        self.log.append(t)

    # ---- 行变换（核验时等价地左乘到 U）----
    def row_swap(self, a, b):
        if a == b:
            return
        self.M[a], self.M[b] = self.M[b], self.M[a]
        self._record(Transform("row", "swap", a, b))

    def row_negate(self, a):
        row = self.M[a]
        for c in range(self.n):
            row[c] = -row[c]
        self._record(Transform("row", "negate", a))

    def row_add(self, target, source, k):
        """target 行 += k * source 行。"""
        k = checked_k(k)
        if k == 0:
            return
        tr, sr = self.M[target], self.M[source]
        for c in range(self.n):
            tr[c] += k * sr[c]
        self._record(Transform("row", "add", target, source, k))

    # ---- 列变换（核验时在 V 上做对应逆操作）----
    def col_swap(self, a, b):
        if a == b:
            return
        for r in range(self.m):
            self.M[r][a], self.M[r][b] = self.M[r][b], self.M[r][a]
        self._record(Transform("column", "swap", a, b))

    def col_negate(self, a):
        for r in range(self.m):
            self.M[r][a] = -self.M[r][a]
        self._record(Transform("column", "negate", a))

    def col_add(self, target, source, k):
        """target 列 += k * source 列。"""
        k = checked_k(k)
        if k == 0:
            return
        for r in range(self.m):
            self.M[r][target] += k * self.M[r][source]
        self._record(Transform("column", "add", target, source, k))


def _find_nonzero(M, k, m, n):
    """在子块 M[k:m, k:n] 找一个非零元；全零返回 None。"""
    for r in range(k, m):
        row = M[r]
        for c in range(k, n):
            if row[c] != 0:
                return r, c
    return None


def _reduce_pivot(eng, k, m, n):
    """把 (k,k) 消成“整除整个剩余子块”的正枢轴，并清零其所在行/列。

    返回时保证：
      M[k][k] = d > 0；
      M[k][c] = 0 (c != k, c >= k)；M[r][k] = 0 (r != k, r >= k)；
      d | M[r][c] 对所有 r > k、c > k。
    """
    M = eng.M
    while True:
        pivot = M[k][k]

        # 阶段 A：扫第 k 列（r>k），不整除则把余数换到枢轴位。
        progressed = False
        for r in range(k + 1, m):
            x = M[r][k]
            if x == 0:
                continue
            q, rem = divmod(x, pivot)  # pivot>0，故 0 <= rem < pivot
            if rem == 0:
                continue
            eng.row_add(r, k, -q)
            eng.row_swap(r, k)
            progressed = True
            break
        if progressed:
            continue

        # 阶段 B：扫第 k 行（c>k）。
        for c in range(k + 1, n):
            x = M[k][c]
            if x == 0:
                continue
            q, rem = divmod(x, pivot)
            if rem == 0:
                continue
            eng.col_add(c, k, -q)
            eng.col_swap(c, k)
            progressed = True
            break
        if progressed:
            continue

        # 阶段 C：pivot 整除第 k 行、第 k 列，先清零。
        for r in range(k + 1, m):
            x = M[r][k]
            if x != 0:
                eng.row_add(r, k, -(x // pivot))
        for c in range(k + 1, n):
            x = M[k][c]
            if x != 0:
                eng.col_add(c, k, -(x // pivot))

        # 阶段 D：pivot 必须整除剩余子块的每个元素，否则整除链不成立。
        # 把违禁元素所在列加回第 k 列，重回 A/B，枢轴会严格变小。
        bad = None
        for r in range(k + 1, m):
            row = M[r]
            for c in range(k + 1, n):
                if row[c] % pivot != 0:
                    bad = (r, c)
                    break
            if bad is not None:
                break
        if bad is not None:
            _, c = bad
            eng.col_add(k, c, 1)  # M[k][k] 仍为 pivot（M[k][c] 已是 0）
            continue

        # 收尾自检：行/列确已清零（全整数精确运算，不满足即内部错误）。
        for r in range(k + 1, m):
            if M[r][k] != 0:
                raise AssertionError("内部错误：枢轴列清零失败")
        for c in range(k + 1, n):
            if M[k][c] != 0:
                raise AssertionError("内部错误：枢轴行清零失败")
        return


def smith(A, *, budget=_DEFAULT_BUDGET):
    """计算整数矩阵 A 的 Smith 标准形证书。

    参数:
      A: list[list[int]]，行数、列数均 ≤ 8，每个元素绝对值 ≤ 10000。
      budget: 初等变换步数上限（正 int，拒绝 bool），防止异常情形下
              无限搜索；耗尽抛 BudgetExhaustedError——只表示搜索未完成，
              与“方程组无解”语义严格分开。

    返回 SmithResult(U, D, V, invariants, rank, transforms)：
      U*A*V = D；U、V 为整数幺模阵（det=±1）；D 为 m×n 非负对角阵，
      非零对角元为正且依次整除，零行/零列在最后。

    异常:
      InvalidMatrixError 子类：形状/类型/上限非法，信息含定位下标；
      BudgetExhaustedError：步数预算耗尽。
    """
    m, n = validate_matrix(
        A, name="A",
        max_rows=_MAX_ROWS, max_cols=_MAX_COLS, max_abs=_MAX_ABS,
    )
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget 必须是 int，收到 {type(budget).__name__}")
    if budget <= 0:
        raise ValueError(f"budget 必须为正整数，收到 {budget}")

    M = [row[:] for row in A]
    eng = _Engine(M, m, n, budget)

    rank = 0
    k = 0
    lim = min(m, n)
    while k < lim:
        nz = _find_nonzero(M, k, m, n)
        if nz is None:
            break
        r, c = nz
        eng.row_swap(k, r)
        eng.col_swap(k, c)
        if M[k][k] < 0:
            eng.row_negate(k)
        _reduce_pivot(eng, k, m, n)
        rank += 1
        k += 1

    # 枢轴依 0,1,... 放置，非零元已在最前、零在最后。做最终不变量校验。
    for i in range(rank):
        if M[i][i] <= 0:
            raise AssertionError(f"内部错误：第 {i} 个对角元不是正数")
    for i in range(rank, lim):
        if M[i][i] != 0:
            raise AssertionError("内部错误：零对角元位置错误")
    invariants = [M[i][i] for i in range(rank)]
    for a, b in zip(invariants, invariants[1:]):
        if b % a != 0:
            raise AssertionError(f"内部错误：整除链被破坏 {a} ∤ {b}")

    D = [row[:] for row in M]

    # U、V 只通过“从单位阵独立重放日志”得到，不信任任何内部累积状态。
    U, M2, V = replay(A, eng.log, shape=(m, n))
    if M2 != D:
        raise AssertionError("内部错误：日志重放与消元结果不一致")

    return SmithResult(
        U=U, D=D, V=V,
        invariants=invariants, rank=rank,
        transforms=tuple(eng.log),
    )
