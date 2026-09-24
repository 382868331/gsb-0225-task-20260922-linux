"""homology() 测试：链复形公理验证、各阶自由秩与挠系数。

期望值来源（均不调用被测核心）：
  - 固定用例为手算结果（圆、填充三角形、Z/2 挠、RP² 胞腔链等）；
  - 随机用例由 tests/_oracle.py 的 expected_homology 独立计算
    （Fraction 高斯消元求有理秩 + 枚举子式行列式因子求挠系数）。
"""

import random
import unittest

from smithcert import (
    BudgetExhaustedError,
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    NonFiniteNumberError,
    NotAChainComplexError,
    homology,
)
from tests._oracle import (
    expected_homology,
    matmul,
)


def verify_homology(self, boundaries, res):
    """对一次 homology() 结果做独立核验（oracle 对照 + 证书复核）。"""
    dim = len(boundaries)
    self.assertEqual(res.dimension, dim)
    self.assertEqual(len(res.groups), dim + 1)
    self.assertEqual(len(res.boundary_ranks), dim)
    self.assertEqual(len(res.certificates), dim)

    # 各阶结果与独立 oracle 一致
    exp = expected_homology(boundaries)
    for k, grp in enumerate(res.groups):
        self.assertEqual(grp.degree, k)
        self.assertEqual(grp.free_rank, exp[k][0],
                         msg=f"H_{k} 自由秩: {grp.free_rank} != {exp[k][0]}")
        self.assertEqual(list(grp.torsion), exp[k][1],
                         msg=f"H_{k} 挠系数: {grp.torsion} != {exp[k][1]}")
        # 挠系数 > 1 且满足整除链
        for d in grp.torsion:
            self.assertGreater(d, 1)
        for a, b in zip(grp.torsion, grp.torsion[1:]):
            self.assertEqual(b % a, 0)

    # 每个边缘矩阵的 Smith 证书独立复核 U*∂*V == D
    for B, cert, rk in zip(boundaries, res.certificates,
                           res.boundary_ranks):
        self.assertEqual(matmul(matmul(cert.U, B), cert.V), cert.D)
        self.assertEqual(cert.rank, rk)


# ---- 手算固定用例：boundaries, 各阶 (自由秩, 挠系数) ----
# 圆 S¹：3 顶点 3 边，∂e0=v1-v0, ∂e1=v2-v1, ∂e2=v2-v0
CIRCLE_D1 = [
    [-1, 0, -1],
    [1, -1, 0],
    [0, 1, 1],
]
# 填充三角形：在圆上粘一个 2-单形，∂σ = e0 + e1 - e2
FILLED_TRIANGLE_D2 = [[1], [1], [-1]]

FIXED_CASES = [
    # 标题, boundaries, 期望各阶 (free_rank, torsion)
    ("圆 S1", [CIRCLE_D1],
     [(1, ()), (1, ())]),
    ("填充三角形（圆盘）", [CIRCLE_D1, FILLED_TRIANGLE_D2],
     [(1, ()), (0, ()), (0, ())]),
    ("Z/2 挠（∂1=[2]）", [[[2]]],
     [(0, (2,)), (0, ())]),
    ("RP2 胞腔链（H1=Z/2）", [[[0]], [[2]]],
     [(1, ()), (0, (2,)), (0, ())]),
    ("挠链 Z/2+Z/4", [[[0, 0]], [[2, 0], [0, 4]]],
     [(1, ()), (0, (2, 4)), (0, ())]),
    ("零边缘 1x3", [[[0, 0, 0]]],
     [(1, ()), (3, ())]),
    ("两个圆的楔（单顶点双边）", [[[0, 0]]],
     [(1, ()), (2, ())]),
    ("环面胞腔链（零边缘）", [[[0, 0]], [[0], [0]]],
     [(1, ()), (2, ()), (1, ())]),
    ("零伦 [[1]]", [[[1]]],
     [(0, ()), (0, ())]),
    ("满秩单点收缩", [[[1, 0], [0, 1]]],
     [(0, ()), (0, ())]),
    ("Z/3 挠", [[[3]]],
     [(0, (3,)), (0, ())]),
    ("混合自由与挠", [[[0, 0], [0, 2]]],
     [(1, (2,)), (1, ())]),
]


class TestHomologyFixed(unittest.TestCase):
    def test_hand_computed_cases(self):
        for title, boundaries, expected in FIXED_CASES:
            with self.subTest(case=title):
                res = homology(boundaries)
                got = [(g.free_rank, g.torsion) for g in res.groups]
                self.assertEqual(
                    got, [(f, tuple(t)) for f, t in expected],
                    msg=f"{title}: {got}",
                )
                verify_homology(self, boundaries, res)

    def test_input_not_mutated(self):
        boundaries = [[row[:] for row in CIRCLE_D1],
                      [row[:] for row in FILLED_TRIANGLE_D2]]
        snapshot = [[row[:] for row in B] for B in boundaries]
        homology(boundaries)
        self.assertEqual(boundaries, snapshot)

    def test_torsion_one_is_dropped(self):
        # ∂₂ 的不变因子为 1（Z/1 平凡），挠列表必须为空
        res = homology([CIRCLE_D1, FILLED_TRIANGLE_D2])
        self.assertEqual(res.groups[1].torsion, ())
        self.assertEqual(res.groups[1].free_rank, 0)


class TestHomologyFuzz(unittest.TestCase):
    """确定性随机链复形：∂₁=[P|0]、∂₂=[0;S] 分块构造保证 ∂₁∂₂=0。"""

    def test_fuzz_block_complexes(self):
        rng = random.Random(20260924)
        for trial in range(120):
            m = rng.randint(1, 4)
            p = rng.randint(0, 3)
            q = rng.randint(1, 3)
            r = rng.randint(1, 3)
            bound = rng.choice([1, 2, 3, 6])
            P = [[rng.randint(-bound, bound) for _ in range(p)]
                 for _ in range(m)]
            S = [[rng.randint(-bound, bound) for _ in range(r)]
                 for _ in range(q)]
            d1 = [P[i] + [0] * q for i in range(m)]
            d2 = [[0] * r for _ in range(p)] + S
            with self.subTest(trial=trial, shapes=((m, p + q), (p + q, r))):
                res = homology([d1, d2])
                verify_homology(self, [d1, d2], res)

    def test_fuzz_one_boundary(self):
        rng = random.Random(777)
        for trial in range(80):
            m = rng.randint(1, 5)
            n = rng.randint(1, 5)
            bound = rng.choice([1, 2, 4, 9])
            d1 = [[rng.randint(-bound, bound) for _ in range(n)]
                  for _ in range(m)]
            with self.subTest(trial=trial, shape=(m, n)):
                res = homology([d1])
                verify_homology(self, [d1], res)


class TestHomologyErrors(unittest.TestCase):
    def test_not_a_chain_complex(self):
        # ∂₁·∂₂ = [1] ≠ 0
        with self.assertRaises(NotAChainComplexError) as ctx:
            homology([[[1]], [[1]]])
        self.assertIn("(0,0)", str(ctx.exception))
        # 第一个非零元定位到具体位置
        with self.assertRaises(NotAChainComplexError) as ctx:
            homology([[[1, 0], [0, 1]], [[0], [3]]])
        self.assertIn("(1,0)", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    def test_shape_mismatch_between_boundaries(self):
        with self.assertRaises(DimensionMismatchError):
            homology([[[1, 2]], [[1, 2]]])

    def test_empty_and_non_list(self):
        with self.assertRaises(MatrixShapeError):
            homology([])
        with self.assertRaises(MatrixShapeError):
            homology("not a list")
        with self.assertRaises(MatrixShapeError):
            homology(True)

    def test_dimension_limit(self):
        with self.assertRaises(MatrixLimitError):
            homology([[[1]], [[0]], [[0]]])

    def test_matrix_limits_and_types(self):
        with self.assertRaises(MatrixLimitError):
            homology([[[10001]]])
        with self.assertRaises(MatrixLimitError):
            homology([[[1] * 9]])
        with self.assertRaises(IntegerRequiredError):
            homology([[[True]]])
        with self.assertRaises(IntegerRequiredError):
            homology([[[2.0]]])
        with self.assertRaises(NonFiniteNumberError):
            homology([[[float("nan")]]])
        with self.assertRaises(MatrixShapeError):
            homology([[]])
        with self.assertRaises(MatrixShapeError):
            homology([[[1, 2], [3]]])

    def test_budget_semantics(self):
        # 预算耗尽是“搜索未完成”，与同调结论无关
        with self.assertRaises(BudgetExhaustedError):
            homology([[[2, 1], [1, 2]]], budget=1)
        res = homology([[[2, 1], [1, 2]]], budget=100_000)
        verify_homology(self, [[[2, 1], [1, 2]]], res)
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    homology([[[1]]], budget=bad)

    def test_invalid_input_leaves_no_partial_state(self):
        # 第二个矩阵非法时整体抛错，不返回任何部分结果
        with self.assertRaises(NotAChainComplexError):
            homology([[[1]], [[1]]])
        with self.assertRaises(MatrixLimitError):
            homology([[[1]], [[10001]]])


if __name__ == "__main__":
    unittest.main()
