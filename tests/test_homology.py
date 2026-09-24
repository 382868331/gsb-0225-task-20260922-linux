"""homology() 整同调测试。

期望值不来自被测实现：
  - 圆、填充三角形、Moore 空间（Z/2 挠）等例子全部手算硬编码；
  - Betti 数用 tests/_oracle.py 的 Fraction 高斯消元独立求有理秩；
  - 挠系数用枚举子式的行列式因子独立求 Smith 不变因子后取 >1 者；
  - 随机合法链复形由独立 Fraction 零空间基清零分母后的整数向量做
    随机整系数组合生成（构造上保证 d_p d_{p+1}=0，并用独立整数乘法
    当场断言）；
  - Euler 关系 Σ(-1)^p b_p = Σ(-1)^p dim C_p 作解析交叉核验；
  - 回传的 Smith 证书用独立矩阵乘法复核 U*d_p*V=D。
"""

import random
import unittest
from math import gcd

from smithcert import (
    BoundaryProductError,
    BudgetExhaustedError,
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    NonFiniteNumberError,
    homology,
)
from tests._oracle import (
    expected_invariants,
    fraction_nullspace,
    matmul,
    rational_rank,
)


# ----------------------------------------------------------------------
# 独立期望：不调用 smithcert.homology / smithcert.smith
# ----------------------------------------------------------------------
def chain_dims(boundaries):
    """(m_0, m_1, ..., m_D)。"""
    d1 = boundaries[0]
    return (len(d1),) + tuple(len(d[0]) for d in boundaries)


def expected_homology(boundaries):
    """独立算出 [(free_rank, torsion_tuple), ...]，次数 0..D。"""
    D = len(boundaries)
    dims = chain_dims(boundaries)
    qranks = [rational_rank(d) for d in boundaries]  # rank_Q(d_1..d_D)
    invs = [expected_invariants(d) for d in boundaries]

    out = []
    for p in range(D + 1):
        r_p = qranks[p - 1] if p >= 1 else 0
        r_next = qranks[p] if p < D else 0
        free_rank = dims[p] - r_p - r_next
        torsion = tuple(s for s in invs[p] if s > 1) if p < D else ()
        out.append((free_rank, torsion))
    return out


def verify_result(self, boundaries, res):
    """对一个 HomologyResult 做全套独立核验。"""
    D = len(boundaries)
    dims = chain_dims(boundaries)

    # 结构
    self.assertEqual(res.max_degree, D)
    self.assertEqual(len(res.groups), D + 1)
    self.assertEqual(len(res.certificates), D)
    self.assertEqual([g.degree for g in res.groups], list(range(D + 1)))

    # 各证书秩与独立有理秩一致
    for p, cert in enumerate(res.certificates, start=1):
        self.assertEqual(cert.rank, rational_rank(boundaries[p - 1]))
        # U*d_p*V=D 独立精确复核
        d = boundaries[p - 1]
        self.assertEqual(matmul(matmul(cert.U, d), cert.V), cert.D)
        self.assertEqual(tuple(cert.invariants), res.invariants[p - 1])
        self.assertEqual(cert.rank, res.ranks[p - 1])
    self.assertEqual(res.ranks[-1], 0)  # r_{D+1}=0 占位

    # 与独立期望逐项一致
    exp = expected_homology(boundaries)
    for p, (free, tor) in enumerate(exp):
        g = res.groups[p]
        self.assertEqual(g.free_rank, free, msg=f"H_{p} 自由秩: {boundaries}")
        self.assertEqual(g.torsion, tor, msg=f"H_{p} 挠系数: {boundaries}")
        # 挠系数保持整除链顺序
        for a, b in zip(g.torsion, g.torsion[1:]):
            self.assertEqual(b % a, 0)
        self.assertGreaterEqual(g.free_rank, 0)

    self.assertEqual(res.betti_numbers, tuple(f for f, _ in exp))
    self.assertEqual(res.torsion_coefficients, tuple(t for _, t in exp))

    # Euler 特征：Σ(-1)^p b_p = Σ(-1)^p dim C_p（有理同调维数）
    euler_chain = sum((-1) ** p * dims[p] for p in range(D + 1))
    euler_hom = sum((-1) ** q * g.free_rank
                    for q, g in enumerate(res.groups))
    self.assertEqual(euler_hom, euler_chain)

    # 包含关系 im d_{p+1} ⊆ ker d_p：r_p + r_{p+1} ≤ m_p
    for p in range(D + 1):
        r_p = res.ranks[p - 1] if p >= 1 else 0
        r_next = res.ranks[p] if p < D else 0
        self.assertLessEqual(r_p + r_next, dims[p])


def make_valid_pair(rng, m0, m1, m2, bound=3):
    """独立构造一对满足 d1*d2=0 的边界矩阵（不经过被测代码）。

    随机取 d1；用 Fraction 零空间基（独立消元）通分得到整数核向量，
    d2 的每列取这些核向量的随机整系数组合——构造上落在 ker d1 中。
    """
    d1 = [[rng.randint(-bound, bound) for _ in range(m1)] for _ in range(m0)]
    ker = fraction_nullspace(d1)
    int_ker = []
    for v in ker:
        mult = 1
        for x in v:
            mult = mult * x.denominator // gcd(mult, x.denominator)
        int_ker.append([int(x * mult) for x in v])

    d2 = [[0] * m2 for _ in range(m1)]
    for col in range(m2):
        if int_ker:
            coeffs = [rng.randint(-bound, bound) for _ in int_ker]
            for row in range(m1):
                d2[row][col] = sum(c * kv[row]
                                   for c, kv in zip(coeffs, int_ker))
    # 独立当场验证构造确实合法（零空间为空时 d2 为零矩阵，也合法）
    assert matmul(d1, d2) == [[0] * m2 for _ in range(m0)]
    return d1, d2


# ----------------------------------------------------------------------
# 手算例子（期望值硬编码）
# ----------------------------------------------------------------------
# 三角形空心圆周：3 顶点、3 有向边，无 2-单形。
CIRCLE_D1 = [
    [-1, -1, 0],
    [1, 0, -1],
    [0, 1, 1],
]
# 填充三角形：d2 的列为面边界 e01 - e02 + e12。
FILLED_D2 = [[1], [-1], [1]]

# Moore 空间风格：Z --[2]--> Z --[0]--> Z，H1 = Z/2。
Z2_D1 = [[0]]
Z2_D2 = [[2]]

# H1 = Z ⊕ Z/2：C2=Z -> C1=Z^3 -> C0=Z。
MIXED_D1 = [[0, 1, 0]]
MIXED_D2 = [[2], [0], [0]]


class TestHomologyHandComputed(unittest.TestCase):
    def test_circle_triangle(self):
        """空心三角形 ≃ S^1：H0=Z, H1=Z。"""
        res = homology([CIRCLE_D1])
        self.assertEqual(res.betti_numbers, (1, 1))
        self.assertEqual(res.torsion_coefficients, ((), ()))
        self.assertFalse(res.groups[0].is_trivial)
        self.assertEqual(res.groups[0].describe(), "Z")
        self.assertEqual(res.groups[1].describe(), "Z")
        verify_result(self, [CIRCLE_D1], res)

    def test_one_vertex_one_loop(self):
        """1×1 零边界 d1=[0]：一个顶点加一条自环边，也是 S^1。"""
        res = homology([[[0]]])
        self.assertEqual(res.betti_numbers, (1, 1))
        self.assertEqual(res.torsion_coefficients, ((), ()))
        verify_result(self, [[[0]]], res)

    def test_filled_triangle_contractible(self):
        """填充三角形（2-单形）：H0=Z 之外全平凡，可缩。"""
        res = homology([CIRCLE_D1, FILLED_D2])
        self.assertEqual(res.betti_numbers, (1, 0, 0))
        self.assertEqual(res.torsion_coefficients, ((), (), ()))
        self.assertTrue(res.groups[1].is_trivial)
        self.assertTrue(res.groups[2].is_trivial)
        self.assertEqual(res.groups[1].describe(), "0")
        verify_result(self, [CIRCLE_D1, FILLED_D2], res)

    def test_two_disjoint_circles(self):
        """两个顶点、两条自环边（d1 为 2×2 零矩阵）：两个不相交圆。"""
        z = [[0, 0], [0, 0]]
        res = homology([z])
        self.assertEqual(res.betti_numbers, (2, 2))
        self.assertEqual(res.torsion_coefficients, ((), ()))
        verify_result(self, [z], res)

    def test_z2_torsion(self):
        """代数挠例：d2=[2], d1=[0]，H0=Z，H1=Z/2，H2=0。"""
        res = homology([Z2_D1, Z2_D2])
        self.assertEqual(res.betti_numbers, (1, 0, 0))
        self.assertEqual(res.torsion_coefficients, ((), (2,), ()))
        self.assertEqual(res.group(1).describe(), "Z/2")
        verify_result(self, [Z2_D1, Z2_D2], res)

    def test_mixed_free_and_z2(self):
        """H1 = Z ⊕ Z/2，H0=H2=0。"""
        res = homology([MIXED_D1, MIXED_D2])
        self.assertEqual(res.betti_numbers, (0, 1, 0))
        self.assertEqual(res.torsion_coefficients, ((), (2,), ()))
        self.assertEqual(res.group(1).describe(), "Z ⊕ Z/2")
        self.assertFalse(res.group(1).is_trivial)
        verify_result(self, [MIXED_D1, MIXED_D2], res)

    def test_h0_torsion_coker(self):
        """d1=[5]：H0 = Z/5（余核挠），H1=0。"""
        res = homology([[[5]]])
        self.assertEqual(res.betti_numbers, (0, 0))
        self.assertEqual(res.torsion_coefficients, ((5,), ()))
        self.assertEqual(res.group(0).describe(), "Z/5")
        verify_result(self, [[[5]]], res)

    def test_zero_boundary_dim2(self):
        """d1,d2 全零：所有链群都是同调。"""
        d1 = [[0, 0], [0, 0]]
        d2 = [[0, 0, 0], [0, 0, 0]]
        res = homology([d1, d2])
        self.assertEqual(res.betti_numbers, (2, 2, 3))
        self.assertEqual(res.torsion_coefficients, ((), (), ()))
        verify_result(self, [d1, d2], res)

    def test_torsion_chain_z2_then_z4(self):
        """d2 对角 (2,4)（配零 d1）：H1 = Z/2 ⊕ Z/4，整除链 2|4。"""
        d1 = [[0, 0, 0]]          # 1×3
        d2 = [[2, 0], [0, 4], [0, 0]]  # 3×2，d1*d2=0
        res = homology([d1, d2])
        # dims (1,3,2)，r1=0，r2=2：H0=Z，H1=Z⊕Z/2⊕Z/4，H2=0
        self.assertEqual(res.betti_numbers, (1, 1, 0))
        self.assertEqual(res.torsion_coefficients, ((), (2, 4), ()))
        verify_result(self, [d1, d2], res)


class TestHomologyRandomFuzz(unittest.TestCase):
    """确定性随机合法链复形批量核验（固定种子，离线可复现）。"""

    def test_fuzz_dim1(self):
        rng = random.Random(20260925)
        for trial in range(50):
            m0 = rng.randint(1, 5)
            m1 = rng.randint(1, 5)
            bound = rng.choice([1, 2, 4])
            d1 = [[rng.randint(-bound, bound) for _ in range(m1)]
                  for _ in range(m0)]
            with self.subTest(trial=trial, shape=(m0, m1)):
                res = homology([d1])
                verify_result(self, [d1], res)

    def test_fuzz_dim2_generated_from_kernel(self):
        rng = random.Random(20260926)
        for trial in range(50):
            m0 = rng.randint(1, 4)
            m1 = rng.randint(1, 4)
            m2 = rng.randint(1, 3)
            d1, d2 = make_valid_pair(rng, m0, m1, m2)
            with self.subTest(trial=trial, shape=(m0, m1, m2)):
                res = homology([d1, d2])
                verify_result(self, [d1, d2], res)
                # 用户对象不被修改：再独立算一遍乘积仍为零
                self.assertEqual(
                    matmul(d1, d2), [[0] * m2 for _ in range(m0)]
                )

    def test_shape_limit_eight(self):
        """8×8 规模边界（接口上限）：零边界与一个手算非零例子。"""
        z = [[0] * 8 for _ in range(8)]
        res = homology([z])
        self.assertEqual(res.betti_numbers, (8, 8))
        verify_result(self, [z], res)

        # 8×8 d1 与 8×1 d2，d2 的列必须落在 ker d1 中：
        # d1 在右下角为 diag(1,2)（秩 2，不变因子 1,2），其余方向
        # 全在核中；d2 取被零化的第 0 方向、倍数为 2（不变因子 2）。
        d1 = [[0] * 8 for _ in range(8)]
        d1[6][6] = 1
        d1[7][7] = 2
        d2 = [[0] for _ in range(8)]
        d2[0][0] = 2
        res2 = homology([d1, d2])
        verify_result(self, [d1, d2], res2)
        # dims (8,8,1)，r1=2，r2=1：b0=6, b1=5, b2=0；
        # H0 的挠来自 d1 的因子 2，H1 的挠来自 d2 的因子 2。
        self.assertEqual(res2.ranks[:2], (2, 1))
        self.assertEqual(res2.betti_numbers, (6, 5, 0))
        self.assertEqual(res2.torsion_coefficients, ((2,), (2,), ()))


class TestHomologyInputErrors(unittest.TestCase):
    def test_not_a_list(self):
        with self.assertRaises(MatrixShapeError):
            homology(([[0]],))  # tuple 不是 list
        with self.assertRaises(MatrixShapeError):
            homology("x")

    def test_empty_and_dimension_limit(self):
        with self.assertRaises(MatrixShapeError):
            homology([])  # 至少要 d_1
        d = [[0]]
        with self.assertRaises(MatrixLimitError):
            homology([d, d, d])  # 维度 3 > 2

    def test_bad_member_matrix(self):
        with self.assertRaises(MatrixShapeError):
            homology([[1, 2], [3]])  # 行长度不一
        with self.assertRaises(MatrixShapeError):
            homology([[]])
        with self.assertRaises(MatrixShapeError):
            homology(["not-a-matrix"])
        # 错误信息能定位到第几个边界矩阵
        with self.assertRaises(MatrixLimitError) as ctx:
            homology([[[0]], [[10001]]])
        self.assertIn("d_2", str(ctx.exception))
        self.assertIn("[0][0]", str(ctx.exception))

    def test_bool_float_nan_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            homology([[[True]]])
        with self.assertRaises(IntegerRequiredError):
            homology([[[0]], [[2.0]]])
        with self.assertRaises(NonFiniteNumberError):
            homology([[[0]], [[float("nan")]]])

    def test_shape_not_linked(self):
        # d1 是 2×3，d2 必须 3×? 才能衔接；给 2×1 报维数不衔接
        d1 = [[0, 0, 0], [0, 0, 0]]
        d2 = [[0], [0]]
        with self.assertRaises(DimensionMismatchError):
            homology([d1, d2])

    def test_composite_nonzero(self):
        # d1*d2 在 [0][0] = 2 ≠ 0
        d1 = [[1, 0]]
        d2 = [[2], [0]]
        with self.assertRaises(BoundaryProductError) as ctx:
            homology([d1, d2])
        msg = str(ctx.exception)
        self.assertIn("d_1 * d_2", msg)
        self.assertIn("[0][0]", msg)
        self.assertIn("2", msg)

    def test_composite_nonzero_user_input_untouched(self):
        d1 = [[1, 0]]
        d2 = [[2], [0]]
        d1_saved = [row[:] for row in d1]
        d2_saved = [row[:] for row in d2]
        with self.assertRaises(BoundaryProductError):
            homology([d1, d2])
        self.assertEqual(d1, d1_saved)
        self.assertEqual(d2, d2_saved)

    def test_chain_condition_checked_before_budget(self):
        # 即使预算极小，链条件错误也要先报（不留“半计算”状态）
        d1 = [[1, 0]]
        d2 = [[2], [0]]
        with self.assertRaises(BoundaryProductError):
            homology([d1, d2], budget=1)

    def test_bad_budget(self):
        d1 = [[1]]
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    homology([d1], budget=bad)

    def test_budget_exhausted_propagates(self):
        # 需要多步消元的 d1，预算 1：BudgetExhaustedError 是“未完成”，
        # 不是同调结论。
        d1 = [[2, 1], [1, 2]]
        with self.assertRaises(BudgetExhaustedError):
            homology([d1], budget=1)
        res = homology([d1], budget=100_000)
        verify_result(self, [d1], res)

    def test_successful_call_does_not_mutate_input(self):
        d1 = [row[:] for row in CIRCLE_D1]
        d2 = [row[:] for row in FILLED_D2]
        d1_saved = [row[:] for row in d1]
        d2_saved = [row[:] for row in d2]
        homology([d1, d2])
        self.assertEqual(d1, d1_saved)
        self.assertEqual(d2, d2_saved)

    def test_element_limit_on_boundary(self):
        with self.assertRaises(MatrixLimitError):
            homology([[[10001]]])
        with self.assertRaises(MatrixLimitError):
            # 8 行上限
            homology([[[0] * 2 for _ in range(9)]])
        with self.assertRaises(MatrixLimitError):
            homology([[[0] * 9]])


if __name__ == "__main__":
    unittest.main()
