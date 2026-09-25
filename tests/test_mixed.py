"""solve_mixed 的测试：手算用例 + 小盒独立枚举交叉验证。

独立验证不调用被测核心求解路径：
  - 盒内全部解由纯枚举约束条件得到；
  - 枚举解与 “particular + 基向量整数组合” 生成的集合逐一比对
    （组合系数在足够大的小范围内穷举，盒小故完备）。
"""

import itertools
import unittest

from smithcert import (
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    ModulusError,
    replay,
    solve_integer,
    solve_mixed,
)


def enum_solutions(A, b, C, d, m, lo, hi):
    """独立枚举 [lo, hi]^n 中满足全部约束的整数点。"""
    n = (len(A[0]) if A else len(C[0])) if (A or C) else 0
    sols = set()
    for x in itertools.product(range(lo, hi + 1), repeat=n):
        ok = True
        if A:
            for row, bi in zip(A, b):
                if sum(a * v for a, v in zip(row, x)) != bi:
                    ok = False
                    break
        if ok and C:
            for row, di, mi in zip(C, d, m):
                if (sum(c * v for c, v in zip(row, x)) - di) % mi != 0:
                    ok = False
                    break
        if ok:
            sols.add(x)
    return sols


def generated_points(particular, basis, lo, hi, coeff=12):
    """particular + 小范围整数组合生成的点集（限制在盒内）。"""
    n = len(particular)
    pts = set()
    for zs in itertools.product(range(-coeff, coeff + 1), repeat=len(basis)):
        x = tuple(particular[j] + sum(z * g[j] for z, g in zip(zs, basis))
                  for j in range(n))
        if all(lo <= v <= hi for v in x):
            pts.add(x)
    return pts


def check_against_enumeration(self, A, b, C, d, m, lo=-6, hi=6):
    res = solve_mixed(A, b, C, d, m)
    sols = enum_solutions(A, b, C, d, m, lo, hi)
    if not res.consistent:
        self.assertEqual(sols, set())
        self.assertIsNotNone(res.obstruction)
        return res
    self.assertTrue(sols, "测试用例应在盒内有解")
    gen = generated_points(res.particular, res.nullspace_basis, lo, hi)
    self.assertEqual(sols, gen)
    return res


class TestHandComputed(unittest.TestCase):
    """手算用例：偶奇冲突、非互素模数、自由变量、无普通等式。"""

    def test_parity_conflict(self):
        # x ≡ 0 (mod 2) 且 x ≡ 1 (mod 2)：偶奇冲突，无解。
        res = solve_mixed(None, None, [[1], [1]], [0, 1], [2, 2])
        self.assertFalse(res.consistent)
        ob = res.obstruction
        self.assertEqual(ob.kind, "divisibility")
        # 障碍可独立核验：w·Ã 各分量被 2 整除，w·b̃ = 1 不然。
        self.assertTrue(all(v % ob.modulus == 0 for v in ob.wA))
        self.assertEqual(ob.wb % ob.modulus, ob.remainder)
        self.assertNotEqual(ob.remainder, 0)

    def test_noncoprime_consistent(self):
        # x ≡ 1 (mod 4), x ≡ 3 (mod 6)：gcd(4,6)=2，1≡3 (mod 2)，可解。
        # 手算：解集 = 9 + 12Z。
        res = solve_mixed(None, None, [[1], [1]], [1, 3], [4, 6])
        self.assertTrue(res.consistent)
        self.assertEqual(res.particular[0] % 12, 9 % 12)
        self.assertEqual(res.nullspace_basis, [[12]])

    def test_noncoprime_inconsistent(self):
        # x ≡ 0 (mod 4), x ≡ 1 (mod 6)：前者偶后者奇，无解。
        res = solve_mixed(None, None, [[1], [1]], [0, 1], [4, 6])
        self.assertFalse(res.consistent)
        self.assertEqual(res.obstruction.kind, "divisibility")

    def test_free_variable(self):
        # x + y ≡ 0 (mod 2)，无等式：二维满秩子格，指数 2。
        res = solve_mixed(None, None, [[1, 1]], [0], [2])
        self.assertTrue(res.consistent)
        self.assertEqual(len(res.nullspace_basis), 2)
        for z in res.nullspace_basis:
            self.assertEqual((z[0] + z[1]) % 2, 0)
        # 基必须独立且张成指数 2 的子格：行列式绝对值 = 2。
        b0, b1 = res.nullspace_basis
        det = b0[0] * b1[1] - b0[1] * b1[0]
        self.assertEqual(abs(det), 2)

    def test_no_equations(self):
        # 无普通等式（A=None），仅同余；变量个数由 C 确定。
        res = solve_mixed(None, None, [[2, 0], [0, 1]], [0, 1], [4, 3])
        self.assertTrue(res.consistent)
        x = res.particular
        self.assertEqual((2 * x[0]) % 4, 0)
        self.assertEqual(x[1] % 3, 1 % 3)

    def test_equality_and_congruence(self):
        # x + y = 4 且 x ≡ 1 (mod 3)：解 (1+3t, 3-3t)。
        res = solve_mixed([[1, 1]], [4], [[1, 0]], [1], [3])
        self.assertTrue(res.consistent)
        x = res.particular
        self.assertEqual(x[0] + x[1], 4)
        self.assertEqual(x[0] % 3, 1)
        self.assertEqual(len(res.nullspace_basis), 1)
        z = res.nullspace_basis[0]
        self.assertEqual(z[0] + z[1], 0)
        self.assertEqual(z[0] % 3, 0)
        self.assertEqual(abs(z[0]), 3)  # 最小步长

    def test_modulus_one_is_unconstrained(self):
        # m_i = 1 不引入辅助变量、不改变解集。
        res = solve_mixed([[2]], [4], [[1]], [0], [1])
        self.assertTrue(res.consistent)
        self.assertEqual(res.particular, [2])
        self.assertEqual(res.nullspace_basis, [])
        self.assertEqual(res.dropped, (0,))
        self.assertEqual(res.n_aux, 0)

    def test_projection_normalization_removes_redundancy(self):
        # x ≡ 0 (mod 2)、y 自由：投影生成集规范化后张成指数 2 子格。
        res = solve_mixed(None, None, [[1, 0]], [0], [2])
        self.assertTrue(res.consistent)
        basis = res.nullspace_basis
        self.assertEqual(len(basis), 2)
        det = basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0]
        self.assertEqual(abs(det), 2)
        sols = enum_solutions(None, None, [[1, 0]], [0], [2], -4, 4)
        gen = generated_points(res.particular, basis, -4, 4)
        self.assertEqual(sols, gen)


class TestEnumerationCrossCheck(unittest.TestCase):
    """小盒枚举独立交叉验证解集与生成格一致。"""

    def test_mixed_small_systems(self):
        cases = [
            ([[1, -1]], [0], [[1, 1]], [0], [2]),
            ([[2, 1]], [3], [[1, 0], [0, 1]], [1, 2], [2, 5]),
            (None, None, [[1, 1], [1, -1]], [1, 0], [3, 4]),
            ([[1, 2]], [5], [[1, 0]], [2], [6]),
            (None, None, [[3, 1]], [2], [7]),
        ]
        for A, b, C, d, m in cases:
            with self.subTest(A=A, C=C):
                check_against_enumeration(self, A, b, C, d, m)

    def test_inconsistent_by_enumeration(self):
        cases = [
            (None, None, [[1], [1]], [0, 1], [2, 2]),
            (None, None, [[1], [1]], [0, 1], [4, 6]),
            ([[2]], [3], None, None, None),  # 2x=3 无整数解
        ]
        for A, b, C, d, m in cases:
            with self.subTest(A=A, C=C):
                check_against_enumeration(self, A, b, C, d, m)

    def test_certificate_replay(self):
        # 增广变换链可独立重放：replay(Ã, transforms) 得到同一证书。
        res = solve_mixed([[1, 1]], [4], [[1, 0]], [1], [3])
        self.assertTrue(res.consistent)
        sr = res.solve_result
        U, M, V = replay(res.augmented_A, sr.transforms)
        self.assertEqual(U, sr.U)
        self.assertEqual(M, sr.D)
        self.assertEqual(V, sr.V)
        # 增广系统与 solve_integer 直接求解一致（数据流复用核验）。
        direct = solve_integer(res.augmented_A, res.augmented_b)
        self.assertEqual(direct.particular, sr.particular)


class TestValidation(unittest.TestCase):
    def test_modulus_must_be_positive(self):
        for bad in (0, -2):
            with self.assertRaises(ModulusError):
                solve_mixed(None, None, [[1]], [0], [bad])

    def test_modulus_type_checked(self):
        with self.assertRaises(IntegerRequiredError):
            solve_mixed(None, None, [[1]], [0], [True])
        with self.assertRaises(IntegerRequiredError):
            solve_mixed(None, None, [[1]], [0], [2.0])

    def test_augmented_limits(self):
        # 9 条有效同余 → 增广行数 9 > 8。
        with self.assertRaises(MatrixLimitError):
            solve_mixed(None, None, [[1]] * 9, [0] * 9, [2] * 9)
        # n=8 变量再加 1 个辅助变量 → 增广列数 9 > 8。
        with self.assertRaises(MatrixLimitError):
            solve_mixed(None, None, [[1] * 8], [0], [2])

    def test_dimension_mismatch(self):
        with self.assertRaises(DimensionMismatchError):
            solve_mixed([[1, 2]], [1], [[1]], [0], [2])  # C 列数 != n
        with self.assertRaises(DimensionMismatchError):
            solve_mixed(None, None, [[1]], [0, 1], [2])
        with self.assertRaises(DimensionMismatchError):
            solve_mixed(None, None, [[1]], [0], [2, 2])
        with self.assertRaises(DimensionMismatchError):
            solve_mixed(None, [1], [[1]], [0], [2])

    def test_both_none_rejected(self):
        with self.assertRaises(MatrixShapeError):
            solve_mixed(None, None, None, None, None)

    def test_input_not_mutated(self):
        A, b = [[1, 1]], [4]
        C, d, m = [[1, 0]], [1], [3]
        snapshot = ([r[:] for r in A], b[:], [r[:] for r in C], d[:], m[:])
        solve_mixed(A, b, C, d, m)
        self.assertEqual((A, b, C, d, m), snapshot)


if __name__ == "__main__":
    unittest.main()
