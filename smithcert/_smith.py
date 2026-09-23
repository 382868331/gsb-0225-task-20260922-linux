"""整数矩阵 Smith 标准形的核心算法。

算法（经典 Kanna–Sahayam/整高斯消元风格，全程整数运算）：

对每个主元位置 k：
  1. 在剩余子矩阵中找非零元，通过行/列交换移到 (k,k)；若全为 0，则秩为 k，结束。
  2. 用“截断除法 + 交换”把主元与其下方/右方的元素做欧几里得约化，
     直到主元整除同列下方、同行右方的所有元素，再把它们整批清零。
     每次余数交换后主元绝对值严格下降，因此必然终止。
  3. 主元变号为正。
  4. 若主元不整除剩余子块中的某个元素，用一次列加法把该元素带入主元列后重新约化
     （这保证了最终 d_i | d_{i+1} 的整除链）。

左乘的行操作累计进 U，右乘的列操作累计进 V，每个操作都写入完整日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._basic import identity
from ._errors import BudgetExhaustedError
from ._ops import (
    ColAdd,
    ColSign,
    ColSwap,
    RowAdd,
    RowSign,
    RowSwap,
)

DEFAULT_BUDGET = 100_000


@dataclass(frozen=True, slots=True)
class SmithResult:
    """``smith(A)`` 的结果。

    * ``U``：m×m 整数幺模矩阵（行列式 ±1）；
    * ``D``：m×n Smith 标准形，非负对角，零在最后，非零不变因子满足整除链；
    * ``V``：n×n 整数幺模矩阵；
    * 恒等式 ``U * A * V == D``（精确整数相等）；
    * ``operations``：按执行顺序排列的完整初等变换日志（可独立重放）；
    * ``rank``：A 的秩；
    * ``invariant_factors``：正不变因子 ``[d_1, ..., d_r]``，d_i | d_{i+1}。
    """

    U: list[list[int]]
    D: list[list[int]]
    V: list[list[int]]
    operations: tuple[object, ...]
    rank: int
    invariant_factors: tuple[int, ...]
    transformations_used: int = field(default=0)


def _trunc_divmod(a: int, p: int) -> tuple[int, int]:
    """截断到零方向的除法：q 向 0 取整，r = a - q*p，满足 |r| < |p|。"""
    q = abs(a) // abs(p)
    if (a < 0) != (p < 0):
        q = -q
    return q, a - q * p


class _Engine:
    """工作矩阵 W 与累计变换矩阵 U、V 的就地消元引擎。"""

    def __init__(self, W: list[list[int]], budget: int) -> None:
        self.W = W
        self.m = len(W)
        self.n = len(W[0])
        self.U = identity(self.m)
        self.V = identity(self.n)
        self.log: list[object] = []
        self.budget = budget
        self.used = 0

    def _tick(self) -> None:
        self.used += 1
        if self.used > self.budget:
            raise BudgetExhaustedError(
                f"初等变换次数超过预算 {self.budget}；这是搜索预算耗尽，"
                "不表示数学上无解（可增大 budget 重试）"
            )

    # -- 行操作（同时作用于 W 与 U；U 是行初等矩阵的乘积） -------------

    def row_swap(self, i: int, j: int) -> None:
        self._tick()
        self.W[i], self.W[j] = self.W[j], self.W[i]
        self.U[i], self.U[j] = self.U[j], self.U[i]
        self.log.append(RowSwap(i, j))

    def row_sign(self, i: int) -> None:
        self._tick()
        self.W[i] = [-x for x in self.W[i]]
        self.U[i] = [-x for x in self.U[i]]
        self.log.append(RowSign(i))

    def row_add(self, target: int, source: int, k: int) -> None:
        if k == 0:
            return
        self._tick()
        Wt, Ws = self.W[target], self.W[source]
        self.W[target] = [a + k * b for a, b in zip(Wt, Ws)]
        Ut, Us = self.U[target], self.U[source]
        self.U[target] = [a + k * b for a, b in zip(Ut, Us)]
        self.log.append(RowAdd(target, source, k))

    # -- 列操作（同时作用于 W 与 V；V 是列初等矩阵的乘积） -------------

    def col_swap(self, i: int, j: int) -> None:
        self._tick()
        for row in self.W:
            row[i], row[j] = row[j], row[i]
        for row in self.V:
            row[i], row[j] = row[j], row[i]
        self.log.append(ColSwap(i, j))

    def col_sign(self, i: int) -> None:
        self._tick()
        for row in self.W:
            row[i] = -row[i]
        for row in self.V:
            row[i] = -row[i]
        self.log.append(ColSign(i))

    def col_add(self, target: int, source: int, k: int) -> None:
        if k == 0:
            return
        self._tick()
        for row in self.W:
            row[target] += k * row[source]
        for row in self.V:
            row[target] += k * row[source]
        self.log.append(ColAdd(target, source, k))

    # -- 主算法 ---------------------------------------------------------

    def run(self) -> int:
        m, n, W = self.m, self.n, self.W
        k = 0
        while k < min(m, n):
            # 1) 在子矩阵 W[k:, k:] 中找一个非零主元
            pi = pj = -1
            for i in range(k, m):
                for j in range(k, n):
                    if W[i][j] != 0:
                        pi, pj = i, j
                        break
                if pi != -1:
                    break
            if pi == -1:
                break  # 剩余子块全零，秩为 k
            if pi != k:
                self.row_swap(k, pi)
            if pj != k:
                self.col_swap(k, pj)

            # 2) 欧几里得约化，直到十字位置全部清零
            while True:
                bounced = False

                # 主元列下方
                for i in range(k + 1, m):
                    a = W[i][k]
                    if a == 0:
                        continue
                    q, r = _trunc_divmod(a, W[k][k])
                    if r != 0:
                        self.row_add(i, k, -q)  # a -> r，|r| < |主元|
                        self.row_swap(i, k)
                        bounced = True
                        break
                    if q != 0:
                        self.row_add(i, k, -q)  # 整除，直接清零
                if bounced:
                    continue

                # 主元行右方
                for j in range(k + 1, n):
                    a = W[k][j]
                    if a == 0:
                        continue
                    q, r = _trunc_divmod(a, W[k][k])
                    if r != 0:
                        self.col_add(j, k, -q)
                        self.col_swap(j, k)
                        bounced = True
                        break
                    if q != 0:
                        self.col_add(j, k, -q)
                if bounced:
                    continue

                # 3) 主元取正号
                if W[k][k] < 0:
                    self.col_sign(k)
                pivot = W[k][k]

                # 4) 整除链修复：pivot 必须整除剩余子块的每个元素
                bad_i = bad_j = -1
                for i in range(k + 1, m):
                    for j in range(k + 1, n):
                        if W[i][j] % pivot != 0:
                            bad_i, bad_j = i, j
                            break
                    if bad_i != -1:
                        break
                if bad_i != -1:
                    # 第 k 列 <- 第 k 列 + 第 bad_j 列：
                    # 主元不变（W[k][bad_j]==0），而 W[bad_i][k] 带上不可整除元素，
                    # 重新进入欧几里得约化，主元绝对值严格减小。
                    self.col_add(k, bad_j, 1)
                    continue

                k += 1
                break

        return k


def smith_normal_form(
    A: list[list[int]], *, budget: int = DEFAULT_BUDGET
) -> SmithResult:
    """对已经过输入校验的整数矩阵副本计算 Smith 标准形证书。

    通常应通过 :func:`smithcert.smith` 调用（含输入校验）。
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget 必须是 int")
    if budget <= 0:
        raise ValueError("budget 必须为正整数")

    W = [row[:] for row in A]
    m, n = len(W), len(W[0])
    engine = _Engine(W, budget)
    rank = engine.run()

    # 防御性不变量检查：消元结束时 W 必须已对角化（不应失败）。
    for i in range(m):
        for j in range(n):
            if i != j and W[i][j] != 0:
                raise AssertionError(
                    f"内部错误：消元结束却存在非对角非零元 W[{i}][{j}]={W[i][j]}"
                )
    factors = tuple(W[i][i] for i in range(rank))

    D = [[0] * n for _ in range(m)]
    for i, d in enumerate(factors):
        D[i][i] = d

    return SmithResult(
        U=engine.U,
        D=D,
        V=engine.V,
        operations=tuple(engine.log),
        rank=rank,
        invariant_factors=factors,
        transformations_used=engine.used,
    )
