"""测试专用的独立参考工具（不调用 smithcert 的任何核心实现）。

* 精确整数矩阵乘法 / Bareiss 行列式；
* 行列式因子 oracle：用所有 k 阶子式的 gcd 独立确定 Smith 不变因子；
* 基于 fractions.Fraction 的高斯消元（独立的秩 / 线性无关判定）。
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations


def ref_matmul(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
        for i in range(m)
    ]


def ref_matvec(A, x):
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def ref_det(M):
    """独立 Bareiss 行列式（与被测包内实现分开编写）。"""
    n = len(M)
    if n == 0:
        return 1
    A = [list(map(int, row)) for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None:
            return 0
        if p != k:
            A[k], A[p] = A[p], A[k]
            sign = -sign
        pivot = A[k][k]
        for i in range(k + 1, n):
            ai = A[i][k]
            for j in range(k + 1, n):
                A[i][j] = (pivot * A[i][j] - ai * A[k][j]) // prev
            A[i][k] = 0
        prev = pivot
    return sign * A[n - 1][n - 1]


def ref_minor_gcds(A):
    """返回 [Δ_0, Δ_1, ..., Δ_min(m,n)]，Δ_k = 所有 k 阶子式绝对值的 gcd。

    Δ_0 = 1。这是 Smith 标准形的独立刻画：d_k = Δ_k / Δ_{k-1}。
    """
    m, n = len(A), len(A[0])
    deltas = [1]
    for k in range(1, min(m, n) + 1):
        g = 0
        for rows in combinations(range(m), k):
            for cols in combinations(range(n), k):
                sub = [[A[i][j] for j in cols] for i in rows]
                g = math.gcd(g, abs(ref_det(sub)))
                if g == 1:
                    break
            if g == 1:
                break
        deltas.append(g)
    return deltas


def ref_invariant_factors(A):
    """独立期望值：由行列式因子商给出的 Smith 不变因子（仅正项）。"""
    deltas = ref_minor_gcds(A)
    factors = []
    for k in range(1, len(deltas)):
        if deltas[k] == 0:
            break
        factors.append(deltas[k] // deltas[k - 1])
    return factors


def ref_rank(A):
    """fractions 高斯消元求秩（有理数，仅用于测试中独立核对）。"""
    A = [list(map(Fraction, row)) for row in A]
    m, n = len(A), len(A[0])
    r = 0
    for col in range(n):
        p = next((i for i in range(r, m) if A[i][col] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][col]
        A[r] = [x / pv for x in A[r]]
        for i in range(m):
            if i != r and A[i][col] != 0:
                f = A[i][col]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        r += 1
    return r


def ref_cols_independent(vectors):
    """一组整数向量作为列是否在 Q 上线性无关（独立高斯消元）。"""
    if not vectors:
        return True
    n = len(vectors[0])
    A = [[Fraction(v[i]) for v in vectors] for i in range(n)]
    return ref_rank(A) == len(vectors)
