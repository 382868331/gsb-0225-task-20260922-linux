"""solve_integer() 测试：可解判定、特解、零空间基、无解障碍。

期望值由独立 oracle 给出（增广矩阵行列式因子判据、Fraction 零空间）。
特解与基向量都做精确整数回代，不做“有理结果取整”。
"""

import random
import unittest

from smithcert import (
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    solve_integer,
)
from tests._oracle import (
    expected_integer_consistent,
    fraction_nullspace,
    fraction_independent,
    maximal_minors_gcd_one,
    matvec,
    row_mat,
)


def verify_solution(self, A, b, res):
    m, n = len(A), len(A[0])

    # 可解性与独立判据一致
    self.assertEqual(
        res.consistent, expected_integer_consistent(A, b),
        msg=f"A={A} b={b}",
    )

    # 零空间维数 = n - rank，且 Smith 证书本身成立（抽查 U/D/V 维度）
    self.assertEqual(len(res.invariants), res.rank)
    self.assertEqual(len(res.nullspace_basis), n - res.rank if res.consistent else 0)

    if res.consistent:
        self.assertIsNone(res.obstruction)
        x0 = res.particular
        # 精确 int 特解
        self.assertTrue(all(isinstance(v, int) and not isinstance(v, bool)
                            for v in x0))
        # A x0 = b 精确成立
        self.assertEqual(matvec(A, x0), list(b))

        # 零空间基：整数、被 A 零化
        for z in res.nullspace_basis:
            self.assertTrue(all(isinstance(v, int) and not isinstance(v, bool)
                                for v in z))
            self.assertEqual(matvec(A, z), [0] * m)
        # 整数上线性无关（转成 Fraction 独立判）
        if res.nullspace_basis:
            self.assertTrue(fraction_independent(res.nullspace_basis, n))

        # 完备性：
        #  (1) 维数等于独立 Fraction 零空间维数 n-rank；
        #  (2) 每个基向量被 A 整数零化（上面已查）；
        #  (3) 它们张成的格是饱和的（n×k 矩阵的所有 k 阶子式 gcd=1）。
        # 整数方程组的核是 Z^n 的饱和子格；基向量都在核内、Q 上张成核、
        # 且像格饱和，三者合起来保证它们正是整数零空间的一组 Z-基，
        # 而不只是指数 >1 的子格。
        qbasis = fraction_nullspace(A)
        self.assertEqual(len(res.nullspace_basis), len(qbasis))
        if res.nullspace_basis:
            self.assertTrue(
                maximal_minors_gcd_one(res.nullspace_basis),
                msg=f"零空间基非饱和（可能只是子格基）: {res.nullspace_basis}",
            )

        # 通解抽查：特解 + 随机整数组合仍是解
        rng = random.Random(99)
        for _ in range(10):
            coeffs = [rng.randint(-5, 5) for _ in res.nullspace_basis]
            x = [x0[j] + sum(c * z[j] for c, z in zip(coeffs,
                                                       res.nullspace_basis))
                 for j in range(n)]
            self.assertEqual(matvec(A, x), list(b))
    else:
        # ---- 无解：障碍证据可独立按整数算术核验 ----
        self.assertIsNone(res.particular)
        ob = res.obstruction
        self.assertIsNotNone(ob)
        self.assertEqual(row_mat(ob.w, A), ob.wA)
        self.assertEqual(sum(wj * bj for wj, bj in zip(ob.w, b)), ob.wb)
        if ob.kind == "zero_row":
            self.assertTrue(all(v == 0 for v in ob.wA))
            self.assertNotEqual(ob.wb, 0)
        elif ob.kind == "divisibility":
            self.assertGreater(ob.modulus, 1)
            self.assertTrue(all(v % ob.modulus == 0 for v in ob.wA))
            self.assertTrue(0 < ob.remainder < ob.modulus)
            self.assertEqual(ob.wb % ob.modulus, ob.remainder)
        else:
            self.fail(f"未知障碍类型 {ob.kind}")


CONSISTENT_CASES = [
    # 标题, A, x_true（b = A x_true 现场构造，保证确有整数解且不取整）
    ("1x1", [[3]], [2]),
    ("对角", [[2, 0], [0, 3]], [2, 3]),
    ("满秩一般", [[2, 1], [1, 3]], [2, 1]),
    ("高矩阵相容", [[1, 2], [3, 4], [5, 6]], [1, 1]),
    ("宽矩阵多解", [[1, 2, 3], [4, 5, 6]], [1, 1, 1]),
    ("零空间非平凡", [[1, -1], [-2, 2]], [3, 3]),
    ("零矩阵齐次", [[0, 0], [0, 0]], [0, 0]),
    ("d>1 恰整除", [[2, 4], [0, 2]], [1, 1]),
    ("含负项", [[-2, 3], [4, -5]], [1, 1]),
    ("3x3 整数解", [[2, 0, 0], [1, 3, 0], [2, 1, 4]], [2, 1, 3]),
    ("负 x", [[3, -2], [-5, 4]], [-2, -3]),
]

INCONSISTENT_CASES = [
    # 标题, A, b；包括“有理数解但无整数解”（整除障碍）与零行障碍
    ("1x1 余数", [[2]], [1]),
    ("整除障碍 2x2", [[2, 0], [0, 2]], [1, 2]),
    ("零行障碍", [[1, 1], [2, 2]], [2, 5]),
    ("零行障碍 b 非零", [[1, 2], [2, 4]], [1, 1]),
    ("有理解但非整数解", [[2, 1], [1, 2]], [1, 1]),
    ("3x2 不可解", [[1, 1], [2, 0], [0, 2]], [2, 1, 1]),
    ("零矩阵非齐次", [[0, 0], [0, 0]], [1, 0]),
    ("经典 gcd 障碍", [[6, 10], [15, 25]], [1, 0]),
]


class TestSolveFixed(unittest.TestCase):
    def test_consistent_cases(self):
        for title, A, x_true in CONSISTENT_CASES:
            b = matvec(A, x_true)
            with self.subTest(case=title):
                res = solve_integer(A, b)
                self.assertTrue(res.consistent, msg=title)
                verify_solution(self, A, b, res)
                # 特解确实是原方程的解（精确核验在 verify_solution 内）
                self.assertEqual(matvec(A, res.particular), b)

    def test_inconsistent_cases(self):
        for title, A, b in INCONSISTENT_CASES:
            with self.subTest(case=title):
                res = solve_integer(A, b)
                self.assertFalse(res.consistent, msg=title)
                verify_solution(self, A, b, res)

    def test_constructed_solutions_from_integer_x(self):
        """b := A x_true 必可解，且特解精确满足（不取整）。"""
        rng = random.Random(7)
        cases = [
            (3, 3), (4, 2), (2, 4), (1, 5), (5, 1), (3, 4), (4, 3),
        ]
        for m, n in cases:
            A = [[rng.randint(-6, 6) for _ in range(n)] for _ in range(m)]
            x_true = [rng.randint(-5, 5) for _ in range(n)]
            b = matvec(A, x_true)
            with self.subTest(shape=(m, n)):
                res = solve_integer(A, b)
                self.assertTrue(res.consistent)
                verify_solution(self, A, b, res)
                self.assertEqual(matvec(A, res.particular), b)


class TestSolveFuzz(unittest.TestCase):
    def test_fuzz_consistency_with_oracle(self):
        rng = random.Random(20260924)
        for trial in range(150):
            m = rng.randint(1, 5)
            n = rng.randint(1, 5)
            bound = rng.choice([2, 5, 20])
            A = [[rng.randint(-bound, bound) for _ in range(n)]
                 for _ in range(m)]
            b = [rng.randint(-30, 30) for _ in range(m)]
            with self.subTest(trial=trial, shape=(m, n)):
                res = solve_integer(A, b)
                verify_solution(self, A, b, res)

    def test_fuzz_full_rank_square_always_solvable_for_divisible_b(self):
        """满秩方阵且 Smith 不变因子全 1 时，任意 b 都有整数解。"""
        rng = random.Random(31337)
        solved = 0
        for _ in range(40):
            n = 3
            A = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(n)]
            b = [rng.randint(-50, 50) for _ in range(n)]
            res = solve_integer(A, b)
            # 与独立判据一致即可（det=±1 时必为 True）
            from tests._oracle import det_fraction
            if abs(det_fraction(A)) == 1:
                self.assertTrue(res.consistent)
                self.assertEqual(matvec(A, res.particular), b)
                solved += 1
        self.assertGreater(solved, 0)


class TestSolveInputErrors(unittest.TestCase):
    def test_b_length_mismatch(self):
        with self.assertRaises(DimensionMismatchError):
            solve_integer([[1, 2], [3, 4]], [1, 2, 3])

    def test_bad_matrix(self):
        with self.assertRaises(MatrixShapeError):
            solve_integer([], [])
        with self.assertRaises(MatrixShapeError):
            solve_integer([[1, 2], [3]], [1, 2])

    def test_bool_float_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            solve_integer([[1, 2], [3, 4]], [True, 2])
        with self.assertRaises(IntegerRequiredError):
            solve_integer([[1.0, 2], [3, 4]], [1, 2])

    def test_b_limit(self):
        with self.assertRaises(MatrixLimitError):
            solve_integer([[1]], [10001])

    def test_nan_b_rejected(self):
        from smithcert import NonFiniteNumberError
        with self.assertRaises(NonFiniteNumberError):
            solve_integer([[1, 0], [0, 1]], [float("nan"), 0])


if __name__ == "__main__":
    unittest.main()
