"""测试专用独立参考实现（不调用 smithcert 的任何核心算法）。

期望值全部独立产生：
  - 有理数高斯消元求秩；
  - Fraction 行列式 + 枚举子式求行列式因子 Δ_k 与不变因子 s_k=Δ_k/Δ_{k-1}；
  - 整数可解性独立判定定理：Ax=b 有整数解当且仅当对每个 k，A 的
    k 阶行列式因子等于增广矩阵 [A|b] 的 k 阶行列式因子；
  - 整数矩阵乘法、行列式、变换重放均在本文件独立实现。

这些代码只服务于测试，属于“独立参考”，不导入 smith、solve 模块。
"""

from fractions import Fraction
from itertools import combinations
from math import gcd


def matmul(A, B):
    """精确整数矩阵乘法。"""
    p, q = len(A), len(B[0])
    k = len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(q)]
            for i in range(p)]


def matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def row_mat(w, A):
    """行向量 w × 矩阵 A（独立实现）。"""
    n = len(A[0])
    return [sum(w[i] * A[i][j] for i in range(len(A))) for j in range(n)]


def det_fraction(M):
    """用 Fraction 高斯消元独立计算行列式（可处理零主元、矩形外的方阵）。"""
    n = len(M)
    X = [[Fraction(v) for v in row] for row in M]
    sign = 1
    for col in range(n):
        pivot = next((r for r in range(col, n) if X[r][col] != 0), None)
        if pivot is None:
            return 0
        if pivot != col:
            X[col], X[pivot] = X[pivot], X[col]
            sign = -sign
        piv = X[col][col]
        for r in range(col + 1, n):
            if X[r][col] != 0:
                factor = X[r][col] / piv
                for c in range(col, n):
                    X[r][c] -= factor * X[col][c]
    d = Fraction(sign)
    for i in range(n):
        d *= X[i][i]
    assert d.denominator == 1
    return d.numerator


def rational_rank(A):
    """有理数域秩（独立高斯消元）。"""
    if not A:
        return 0
    m, n = len(A), len(A[0])
    X = [[Fraction(v) for v in row] for row in A]
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, m) if X[i][col] != 0), None)
        if pivot is None:
            continue
        X[r], X[pivot] = X[pivot], X[r]
        piv = X[r][col]
        for c in range(col, n):
            X[r][c] /= piv
        for i in range(m):
            if i != r and X[i][col] != 0:
                f = X[i][col]
                for c in range(col, n):
                    X[i][c] -= f * X[r][c]
        r += 1
    return r


def _minor(A, rows, cols):
    return [[A[i][j] for j in cols] for i in rows]


def determinantal_divisors(A):
    """返回 [Δ_0, Δ_1, ..., Δ_p]，Δ_k=所有 k 阶子式绝对值的 gcd。"""
    m, n = len(A), len(A[0])
    out = [1]
    for k in range(1, min(m, n) + 1):
        g = 0
        for rows in combinations(range(m), k):
            for cols in combinations(range(n), k):
                g = gcd(g, abs(det_fraction(_minor(A, rows, cols))))
                if g == 1:
                    break
            if g == 1:
                break
        out.append(g)
    return out


def expected_invariants(A):
    """独立期望的非零 Smith 不变因子 s_k = Δ_k / Δ_{k-1}（Δ_k≠0 段）。"""
    ds = determinantal_divisors(A)
    rank = max((k for k, d in enumerate(ds) if d != 0), default=0)
    return [ds[k] // ds[k - 1] for k in range(1, rank + 1)]


def expected_integer_consistent(A, b):
    """独立判定 Ax=b 是否有整数解。

    判据（Smith/行列式因子定理）：
      (1) 有理数域秩不增：rank_Q(A) = rank_Q([A|b])；
      (2) 对 k=1..r（r=rank A），A 与增广矩阵的 k 阶行列式因子相等
          （所有 k 阶子式绝对值的 gcd 相同）。
    """
    m, n = len(A), len(A[0])
    Ab = [row[:] + [b[i]] for i, row in enumerate(A)]
    r = rational_rank(A)
    if rational_rank(Ab) != r:
        return False
    dA = determinantal_divisors(A)
    dAb = determinantal_divisors(Ab)
    return all(dA[k] == dAb[k] for k in range(1, r + 1))


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def independent_replay(A, transforms):
    """测试侧独立重放：从 A 与单位阵出发，逐条施加日志中的初等变换。

    与 smithcert.transform.replay 是两份独立代码，用于交叉核验。
    返回 (U, M, V)。
    """
    m, n = len(A), len(A[0])
    M = [row[:] for row in A]
    U, V = identity(m), identity(n)

    def row_op(X, t):
        if t.kind == "swap":
            X[t.i], X[t.j] = X[t.j], X[t.i]
        elif t.kind == "negate":
            X[t.i] = [-v for v in X[t.i]]
        elif t.kind == "add":
            X[t.i] = [a + t.k * b for a, b in zip(X[t.i], X[t.j])]
        else:
            raise AssertionError(f"非法变换类型 {t.kind}")

    def col_op(X, t):
        if t.kind == "swap":
            for r in range(len(X)):
                X[r][t.i], X[r][t.j] = X[r][t.j], X[r][t.i]
        elif t.kind == "negate":
            for r in range(len(X)):
                X[r][t.i] = -X[r][t.i]
        elif t.kind == "add":
            for r in range(len(X)):
                X[r][t.i] += t.k * X[r][t.j]
        else:
            raise AssertionError(f"非法变换类型 {t.kind}")

    allowed = {"swap", "negate", "add"}
    for t in transforms:
        assert t.side in ("row", "column"), f"非法变换侧 {t.side}"
        assert t.kind in allowed, f"非法变换类型 {t.kind}"
        if t.kind == "add":
            assert isinstance(t.k, int) and not isinstance(t.k, bool)
        if t.side == "row":
            row_op(M, t)
            row_op(U, t)
        else:
            # 同一个列操作按序连乘到 V：V <- V P（不取逆）。
            col_op(M, t)
            col_op(V, t)
    return U, M, V


def fraction_nullspace(A):
    """独立的有理数零空间基（Fraction RREF），返回长度 n 的 Fraction 向量表。

    只供测试完备性核验，与被测实现无关。
    """
    m, n = len(A), len(A[0])
    X = [[Fraction(v) for v in row] for row in A]
    pivot_cols = []
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, m) if X[i][col] != 0), None)
        if pivot is None:
            continue
        X[r], X[pivot] = X[pivot], X[r]
        piv = X[r][col]
        for c in range(n):
            X[r][c] /= piv
        for i in range(m):
            if i != r and X[i][col] != 0:
                f = X[i][col]
                for c in range(n):
                    X[i][c] -= f * X[r][c]
        pivot_cols.append(col)
        r += 1
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        v = [Fraction(0) for _ in range(n)]
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivot_cols):
            v[pc] = -X[i][fc]
        basis.append(v)
    return basis


def maximal_minors_gcd_one(vectors):
    """n×k 整数矩阵 B 的像格在 Z^n 中是否饱和：所有 k 阶子式 gcd 是否为 1。

    若整数向量列构成某饱和整数格（如整数方程组的核 ker_Z(A)）在 Q 上的
    一组基，且它们都落在该格内，则子式 gcd=1 当且仅当它们就是该格的
    整数基（Z-基）。
    """
    k = len(vectors)
    n = len(vectors[0])
    if k == 0:
        return True
    g = 0
    for rows in combinations(range(n), k):
        sub = [[vectors[j][i] for i in rows] for j in range(k)]
        g = gcd(g, abs(det_fraction(sub)))
        if g == 1:
            return True
    return False


def fraction_independent(vectors, n):
    """若干整数向量在 Q 上是否线性无关。"""
    if not vectors:
        return True
    mat = [[Fraction(v) for v in vec] for vec in vectors]
    r = rational_rank(mat)
    return r == len(vectors) and len(vectors) <= n


def expected_homology(boundaries):
    """独立计算链复形各阶 (自由秩, 挠系数列表)。

    自由秩：b_k = dim C_k − rank_Q(∂_k) − rank_Q(∂_{k+1})，秩用
    Fraction 高斯消元独立求得（Betti 数只依赖有理秩）。
    挠系数：∂_{k+1} 的大于 1 的 Smith 不变因子，由枚举全部子式的
    行列式因子 Δ_k（gcd）独立求得 s_k = Δ_k/Δ_{k-1}。
    不调用 smithcert 的任何核心代码。
    """
    dim = len(boundaries)
    ranks = [rational_rank(B) for B in boundaries]
    out = []
    for k in range(dim + 1):
        # dim C_k：k=0 取 ∂₁ 的行数，否则取 ∂_k 的列数
        n_k = len(boundaries[0]) if k == 0 else len(boundaries[k - 1][0])
        free = (n_k - (ranks[k - 1] if k >= 1 else 0)
                - (ranks[k] if k < dim else 0))
        torsion = []
        if k < dim:
            torsion = [d for d in expected_invariants(boundaries[k]) if d > 1]
        out.append((free, torsion))
    return out
