"""Smith 标准形证书的核心性质测试。

期望值来自独立参考（Bareiss 行列式、k 阶子式 gcd 的行列式因子 oracle），
不调用被测核心生成；U*A*V=D 等只作为必要的补充往返核验。
"""

from __future__ import annotations

import unittest

import _reference as ref

from smithcert import (
    ColAdd,
    ColSign,
    ColSwap,
    RowAdd,
    RowSign,
    RowSwap,
    determinant,
    matmul,
    replay_certificate,
    smith,
)

ALLOWED_OPS = (RowSwap, RowSign, RowAdd, ColSwap, ColSign, ColAdd)

# 覆盖：矩形、秩亏、负项、对角不满足整除链等情形。
FIXTURES = [
    [[0]],
    [[7]],
    [[-13]],
    [[2, 4], [4, 2]],
    [[-2, -4], [-6, -8]],
    [[6, 4], [0, 10]],          # 对角 (2,30) 才是 SNF；朴素对角线 6,10 无整除关系
    [[6, 0, 0], [0, 10, 0], [0, 0, 15]],  # gcd 链修复的经典例
    [[1, 2, 3], [4, 5, 6]],     # 2x3 矩形、满秩
    [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]],  # 3x4，秩 1
    [[1, 0, -1], [2, -2, 2], [0, 3, -3], [-1, 1, 0]],  # 4x3 秩亏
    [[0, 0, 0], [0, 0, 0]],
    [[-1, -2], [-3, -4], [-5, -6], [-7, -8], [-9, -10], [-11, -12]],  # 6x2
]


def assert_smith_invariants(test: unittest.TestCase, A: list[list[int]]) -> None:
    m, n = len(A), len(A[0])
    res = smith(A)
    U, D, V, ops = res.U, res.D, res.V, res.operations

    # 1) 精确恒等式 U*A*V = D
    test.assertEqual(matmul(matmul(U, A), V), D)

    # 2) 独立重放日志（只走初等矩阵乘法，不经过消元引擎）
    U2, D2, V2 = replay_certificate(A, ops)
    test.assertEqual(U2, U)
    test.assertEqual(V2, V)
    test.assertEqual(D2, D)

    # 3) 日志只含允许的六类初等操作
    for op in ops:
        test.assertIsInstance(op, ALLOWED_OPS)

    # 4) det(U) = det(V) = ±1（独立 Bareiss 实现）
    test.assertIn(ref.ref_det(U), (1, -1))
    test.assertIn(ref.ref_det(V), (1, -1))

    # 5) D 形状、非负对角、零在最后、整除链
    test.assertEqual((len(D), len(D[0])), (m, n))
    diag = [D[i][j] for i in range(m) for j in range(n) if i != j]
    test.assertTrue(all(v == 0 for v in diag), "D 必须是对角矩阵")
    pivots = [D[i][i] for i in range(min(m, n))]
    for d in pivots:
        test.assertGreaterEqual(d, 0)
    # 所有正项连续位于最前
    positives = [d for d in pivots if d > 0]
    test.assertEqual(pivots[: len(positives)], positives)
    for a, b in zip(positives, positives[1:]):
        test.assertEqual(b % a, 0, f"整除链破坏：{a} 不整除 {b}")

    # 6) 与独立 oracle（行列式因子商）一致
    test.assertEqual(tuple(positives), tuple(ref.ref_invariant_factors(A)))
    test.assertEqual(res.rank, ref.ref_rank(A))
    test.assertEqual(tuple(res.invariant_factors), tuple(positives))

    # 7) 满秩方阵时，不变因子乘积 = |det A|（独立行列式）
    if m == n and len(positives) == m:
        prod = 1
        for d in positives:
            prod *= d
        test.assertEqual(prod, abs(ref.ref_det(A)))

    # 8) 包内 Bareiss 与参考 Bareiss 对 U、V 一致（确定性交叉核对）
    test.assertEqual(determinant(U), ref.ref_det(U))
    test.assertEqual(determinant(V), ref.ref_det(V))


class SmithFixturesTest(unittest.TestCase):
    def test_all_fixtures(self) -> None:
        for A in FIXTURES:
            with self.subTest(A=A):
                assert_smith_invariants(self, A)

    def test_known_non_chain_example(self) -> None:
        # diag(6,10,15) 的 SNF 为 1, 30, 30：朴素输出不满足整除链
        res = smith([[6, 0, 0], [0, 10, 0], [0, 0, 15]])
        self.assertEqual(res.invariant_factors, (1, 30, 30))

    def test_known_2x2_example(self) -> None:
        res = smith([[6, 4], [0, 10]])
        self.assertEqual(res.invariant_factors, (2, 30))

    def test_negative_entry_sign_normalization(self) -> None:
        res = smith([[-2, -4], [-6, -8]])
        for d in res.invariant_factors:
            self.assertGreater(d, 0)
        self.assertEqual(res.invariant_factors, (2, 4))

    def test_input_not_mutated(self) -> None:
        A = [[-2, 4], [6, -8]]
        snapshot = [row[:] for row in A]
        smith(A)
        self.assertEqual(A, snapshot)

    def test_idempotent_on_diagonal_result(self) -> None:
        # 对 SNF 再做一次，仍是同样的正对角（顺序保持）
        A = [[6, 4], [0, 10]]
        D1 = smith(A).D
        res2 = smith(D1)
        self.assertEqual(res2.D, D1)


if __name__ == "__main__":
    unittest.main()
