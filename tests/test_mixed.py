"""solve_mixed() 测试：等式 + 模同余混合系统。

手算用例（一元偶奇冲突、非互素模数、自由变量、无普通等式）+ 小盒
独立枚举对照：盒内期望解集由直接逐个检验约束得到（不调用被测核心），
实际侧用 Fraction 独立求解 z 的坐标并检验其为整数来判定格成员关系，
不调用 smithcert 的任何求解函数。
"""

import itertools
import random
import unittest
from fractions import Fraction

from smithcert import (
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    replay,
    solve_mixed,
)


def satisfies(A, b, C, d, m, x):
    """独立直接检验 x 是否满足全部等式与同余。"""
    n = len(x)
    for i in range(len(A)):
        if sum(A[i][j] * x[j] for j in range(n)) != b[i]:
            return False
    for i in range(len(C)):
        if (sum(C[i][j] * x[j] for j in range(n)) - d[i]) % m[i] != 0:
            return False
    return True


def box_solutions(A, b, C, d, m, B):
    """盒 [-B,B]^n 内全部解（逐个直接检验，不调用被测核心）。"""
    n = len(A[0]) if A else len(C[0])
    out = set()
    for x in itertools.product(range(-B, B + 1), repeat=n):
        if satisfies(A, b, C, d, m, list(x)):
            out.add(x)
    return out


def in_lattice(basis, v):
    """独立判定 v 是否属于 basis 生成的整数格。

    用 Fraction 高斯消元解 B z = v（基向量线性无关则解唯一），
    再要求每个分量都是整数。独立方法，不调用 smithcert。
    """
    n = len(v)
    k = len(basis)
    if k == 0:
        return all(t == 0 for t in v)
    M = [[Fraction(basis[j][i]) for j in range(k)] + [Fraction(v[i])]
         for i in range(n)]
    row = 0
    for col in range(k):
        p = next((r for r in range(row, n) if M[r][col] != 0), None)
        if p is None:
            return False
        M[row], M[p] = M[p], M[row]
        piv = M[row][col]
        M[row] = [x / piv for x in M[row]]
        for r in range(n):
            if r != row and M[r][col] != 0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[row])]
        row += 1
    for r in range(row, n):
        if M[r][k] != 0:
            return False
    z = [M[r][k] for r in range(k)]
    return all(t.denominator == 1 for t in z)


def check_box(self, A, b, C, d, m, B=5):
    """对照：盒内期望解集 == particular + 基生成格（独立成员判定）。"""
    res = solve_mixed([r[:] for r in A], list(b),
                      [r[:] for r in C], list(d), list(m))
    exp = box_solutions(A, b, C, d, m, B)
    if not res.consistent:
        self.assertEqual(exp, set(),
                         msg="判定不可解但盒内有解: %s" % sorted(exp)[:3])
        return res
    x0 = res.particular
    self.assertTrue(satisfies(A, b, C, d, m, x0),
                    msg="特解不满足约束: %s" % (x0,))
    for z in res.basis:
        self.assertTrue(
            satisfies(A, [0] * len(A), C, [0] * len(C), m, z),
            msg="基向量不满足齐次约束: %s" % (z,))
    for x in exp:
        v = [x[j] - x0[j] for j in range(len(x))]
        self.assertTrue(in_lattice(res.basis, v),
                        msg="盒内解 %s 不在基生成格中: %s" % (x, res.basis))
    return res


class TestHandComputed(unittest.TestCase):
    """手算用例。"""

    def test_parity_conflict_single_var(self):
        # x ≡ 0 (mod 2) 且 x ≡ 1 (mod 2)：偶奇冲突，不可解。
        res = solve_mixed([], [], [[1], [1]], [0, 1], [2, 2])
        self.assertFalse(res.consistent)
        ob = res.obstruction
        self.assertEqual(ob.kind, "divisibility")
        # 独立核验障碍：w*A_aug 各分量被模数整除，w*b_aug 余非零。
        self.assertTrue(all(v % ob.modulus == 0 for v in ob.wA))
        self.assertEqual(ob.wb % ob.modulus, ob.remainder)
        self.assertNotEqual(ob.remainder, 0)
        # 手算：x - 2t1 = 0, x - 2t2 = 1 相减得 2(t2 - t1) = 1。
        self.assertEqual(ob.modulus, 2)

    def test_non_coprime_moduli_conflict(self):
        # x ≡ 1 (mod 4), x ≡ 2 (mod 6)：mod gcd(4,6)=2 得 1≡0，矛盾。
        res = solve_mixed([], [], [[1], [1]], [1, 2], [4, 6])
        self.assertFalse(res.consistent)
        self.assertEqual(res.obstruction.kind, "divisibility")
        self.assertEqual(res.obstruction.modulus, 2)

    def test_non_coprime_moduli_consistent(self):
        # x ≡ 1 (mod 4), x ≡ 1 (mod 6) 等价于 x ≡ 1 (mod 12)。
        res = solve_mixed([], [], [[1], [1]], [1, 1], [4, 6])
        self.assertTrue(res.consistent)
        self.assertEqual(res.particular[0] % 12, 1)
        self.assertEqual(len(res.basis), 1)
        self.assertEqual(abs(res.basis[0][0]), 12)
        for t in (-25, -1, 0, 1, 7):
            x = res.particular[0] + t * res.basis[0][0]
            self.assertEqual(x % 4, 1)
            self.assertEqual(x % 6, 1)

    def test_free_variable_with_congruence(self):
        # x + y = 4, x ≡ 0 (mod 2)。解：x 任意偶数，y = 4 - x。
        res = solve_mixed([[1, 1]], [4], [[1, 0]], [0], [2])
        self.assertTrue(res.consistent)
        x0 = res.particular
        self.assertEqual(x0[0] + x0[1], 4)
        self.assertEqual(x0[0] % 2, 0)
        self.assertEqual(len(res.basis), 1)
        z = res.basis[0]
        self.assertEqual(z[0] + z[1], 0)
        self.assertEqual(z[0] % 2, 0)
        # 手算具体解 (0,4),(2,2),(-2,6) 都应可达。
        for target in ([0, 4], [2, 2], [-2, 6]):
            v = [target[j] - x0[j] for j in range(2)]
            self.assertTrue(in_lattice(res.basis, v), msg="%s 不可达" % (target,))

    def test_no_equalities(self):
        # 无普通等式：2x ≡ 2 (mod 4) 等价于 x 为奇数。
        res = solve_mixed([], [], [[2]], [2], [4])
        self.assertTrue(res.consistent)
        self.assertEqual(res.particular[0] % 2, 1)
        self.assertEqual(len(res.basis), 1)
        self.assertEqual(abs(res.basis[0][0]), 2)

    def test_modulus_one_is_unconstrained(self):
        # m=1 不引入辅助变量，不施加任何约束。
        res = solve_mixed([[1]], [3], [[1]], [9], [1])
        self.assertTrue(res.consistent)
        self.assertEqual(res.n_aux, 0)
        self.assertEqual(res.active, ())
        self.assertEqual(res.particular, [3])
        self.assertEqual(res.basis, [])

    def test_all_congruences_trivial(self):
        # 只有 m=1 的同余且无等式：整个 Z^2 都是解。
        res = solve_mixed([], [], [[0, 0]], [0], [1])
        self.assertTrue(res.consistent)
        self.assertEqual(len(res.basis), 2)
        for target in ([3, -7], [0, 0], [-5, 11]):
            self.assertTrue(in_lattice(res.basis, target))

    def test_certificate_replay(self):
        # 增广变换链可独立重放：replay(A_aug, transforms) == (U, D, V)。
        res = solve_mixed([[2, 1]], [5], [[1, 1], [3, -1]], [1, 3], [4, 6])
        self.assertTrue(res.consistent)
        U, M, V = replay(res.A_aug, res.transforms)
        self.assertEqual(M, res.D)
        self.assertEqual(U, res.U)
        self.assertEqual(V, res.V)


class TestBoxEnumeration(unittest.TestCase):
    """小盒独立枚举对照（确定性随机）。"""

    def test_random_small_systems(self):
        rng = random.Random(20260925)
        trials = 0
        while trials < 120:
            n = rng.randint(1, 3)
            m0 = rng.randint(0, 2)
            p = rng.randint(1, 3)
            if m0 + p > 8 or n + p > 8:
                continue
            A = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(m0)]
            C = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(p)]
            b = [rng.randint(-4, 4) for _ in range(m0)]
            d = [rng.randint(-4, 4) for _ in range(p)]
            m = [rng.randint(1, 6) for _ in range(p)]
            check_box(self, A, b, C, d, m, B=5)
            trials += 1

    def test_fixed_cases_box(self):
        check_box(self, [], [], [[1], [1]], [0, 1], [2, 2])
        check_box(self, [], [], [[1], [1]], [1, 1], [4, 6])
        check_box(self, [], [], [[1], [1]], [1, 2], [4, 6])
        check_box(self, [], [], [[2]], [2], [4])
        check_box(self, [[1, 1]], [4], [[1, 0]], [0], [2])
        check_box(self, [[2, 1]], [5], [[1, 1], [3, -1]], [1, 2], [4, 6])
        check_box(self, [[1, -1, 2]], [3], [[2, 0, 1]], [1], [3])


class TestErrors(unittest.TestCase):
    def test_bool_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            solve_mixed([], [], [[True]], [0], [2])
        with self.assertRaises(IntegerRequiredError):
            solve_mixed([], [], [[1]], [0], [True])

    def test_float_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            solve_mixed([], [], [[1.0]], [0], [2])
        with self.assertRaises(IntegerRequiredError):
            solve_mixed([], [], [[1]], [0], [2.0])

    def test_nonpositive_modulus(self):
        with self.assertRaises(ValueError):
            solve_mixed([], [], [[1]], [0], [0])
        with self.assertRaises(ValueError):
            solve_mixed([], [], [[1]], [0], [-3])

    def test_both_empty_rejected(self):
        with self.assertRaises(MatrixShapeError):
            solve_mixed([], [], [], [], [])

    def test_dimension_mismatch(self):
        with self.assertRaises(DimensionMismatchError):
            solve_mixed([[1, 2]], [3], [[1]], [0], [2])
        with self.assertRaises(DimensionMismatchError):
            solve_mixed([], [], [[1]], [0], [2, 3])
        with self.assertRaises(DimensionMismatchError):
            solve_mixed([], [], [[1]], [0, 1], [2])

    def test_augmented_limits(self):
        # 9 条保留同余，增广行数 9 > 8。
        C = [[1]] * 9
        with self.assertRaises(MatrixLimitError):
            solve_mixed([], [], C, [0] * 9, [2] * 9)
        # n=5 且 4 条保留同余，增广列数 9 > 8。
        C = [[1, 0, 0, 0, 0]] * 4
        with self.assertRaises(MatrixLimitError):
            solve_mixed([], [], C, [0] * 4, [2] * 4)
        # 模数超上限。
        with self.assertRaises(MatrixLimitError):
            solve_mixed([], [], [[1]], [0], [10001])

    def test_input_not_mutated(self):
        A = [[1, 1]]
        b = [4]
        C = [[1, 0]]
        d = [0]
        m = [2]
        snapshot = ([r[:] for r in A], list(b), [r[:] for r in C],
                    list(d), list(m))
        solve_mixed(A, b, C, d, m)
        self.assertEqual((A, b, C, d, m), snapshot)


if __name__ == "__main__":
    unittest.main()
