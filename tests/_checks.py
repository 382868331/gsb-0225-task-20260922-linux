"""测试共享断言助手（非 test_ 前缀，不会被 discover 当作用例模块）。"""

from __future__ import annotations

import unittest
from fractions import Fraction

import _reference as ref

from smithcert import integer_solve, matvec


def rational_consistent(A, b):
    """独立 Fraction 高斯-若尔当消元：A x = b 在 Q 上是否相容。"""
    m, n = len(A), len(A[0])
    M = [[Fraction(v) for v in row] + [Fraction(v)] for row, v in zip(A, b)]
    r = 0
    for col in range(n):
        p = next((i for i in range(r, m) if M[i][col] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][col]
        M[r] = [x / pv for x in M[r]]
        for i in range(m):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * c for a, c in zip(M[i], M[r])]
        r += 1
    return all(M[i][n] == 0 for i in range(r, m))


def rational_nullspace(A):
    """独立 Fraction 消元给出的零空间基（Fraction 向量列表）。"""
    m, n = len(A), len(A[0])
    M = [[Fraction(v) for v in row] for row in A]
    r = 0
    pivot_cols: list[int] = []
    for col in range(n):
        p = next((i for i in range(r, m) if M[i][col] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][col]
        M[r] = [x / pv for x in M[r]]
        for i in range(m):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * c for a, c in zip(M[i], M[r])]
        pivot_cols.append(col)
        r += 1
    free = [j for j in range(n) if j not in pivot_cols]
    basis = []
    for fcol in free:
        v = [Fraction(0)] * n
        v[fcol] = Fraction(1)
        for ri, pc in enumerate(pivot_cols):
            v[pc] = -M[ri][fcol]
        basis.append(v)
    return basis


def in_span_Q(basis, target):
    """target 是否为 basis（整数向量）的有理线性组合。"""
    if not basis:
        return all(v == 0 for v in target)
    k = len(basis)
    n = len(target)
    M = [[Fraction(basis[c][r]) for c in range(k)] + [target[r]] for r in range(n)]
    row = 0
    for col in range(k):
        p = next((i for i in range(row, n) if M[i][col] != 0), None)
        if p is None:
            continue
        M[row], M[p] = M[p], M[row]
        pv = M[row][col]
        M[row] = [x / pv for x in M[row]]
        for i in range(n):
            if i != row and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * c for a, c in zip(M[i], M[row])]
        row += 1
    return all(M[i][k] == 0 for i in range(row, n))


def check_solvable_case(test: unittest.TestCase, A, b, nullity=None):
    res = integer_solve(A, b)
    m, n = len(A), len(A[0])
    test.assertTrue(res.solvable, msg=f"应可解：{res.obstruction}")
    test.assertIsNone(res.obstruction)
    x0 = res.particular_solution
    test.assertIsNotNone(x0)
    test.assertEqual(len(x0), n)
    test.assertTrue(all(isinstance(v, int) and not isinstance(v, bool) for v in x0))
    test.assertEqual(matvec(A, list(x0)), list(b))
    test.assertEqual(res.rank, ref.ref_rank(A))
    test.assertEqual(res.nullity, n - res.rank)
    if nullity is not None:
        test.assertEqual(len(res.null_space_basis), nullity)
    test.assertEqual(len(res.null_space_basis), n - res.rank)
    for z in res.null_space_basis:
        test.assertEqual(len(z), n)
        test.assertTrue(all(isinstance(v, int) for v in z))
        test.assertEqual(matvec(A, list(z)), [0] * m)
    test.assertTrue(ref.ref_cols_independent(list(res.null_space_basis)))
    for v in rational_nullspace(A):
        test.assertTrue(in_span_Q(list(res.null_space_basis), v))
    return res


def check_unsolvable_case(test: unittest.TestCase, A, b, rational_consistent_flag):
    res = integer_solve(A, b)
    m, n = len(A), len(A[0])
    test.assertFalse(res.solvable)
    test.assertIsNone(res.particular_solution)
    test.assertIsNotNone(res.obstruction)
    ob = res.obstruction
    test.assertEqual(rational_consistent(A, b), rational_consistent_flag)

    u = ob["certificate_row_U"]
    test.assertEqual(len(u), m)
    uA = [sum(u[i] * A[i][j] for i in range(m)) for j in range(n)]
    ub = sum(u[i] * b[i] for i in range(m))
    test.assertEqual(ub, ob["transformed_rhs_entry"])
    if ob["kind"] == "nondivisible_pivot":
        d = ob["pivot"]
        test.assertGreater(d, 0)
        test.assertTrue(all(v % d == 0 for v in uA))
        test.assertNotEqual(ub % d, 0)
        test.assertEqual(ub % d, ob["residue"])
    else:  # zero_row_nonzero_rhs
        test.assertTrue(all(x == 0 for x in uA))
        test.assertNotEqual(ub, 0)
        test.assertEqual(ob["residue"], ub)
    return res
