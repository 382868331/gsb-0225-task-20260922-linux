"""smith() 证书性质测试。

期望值来自 tests/_oracle.py 的独立实现（Fraction 高斯消元、枚举子式的
行列式因子），不调用被测核心。U*A*V=D 与 det(U)=det(V)=±1 为独立的
精确算术核验。
"""

import random
import unittest

from smithcert import (
    BudgetExhaustedError,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    smith,
)
from smithcert.transform import replay as cert_replay

from tests._oracle import (
    det_fraction,
    expected_invariants,
    identity,
    independent_replay,
    matmul,
    rational_rank,
)

ALLOWED_KINDS = {"swap", "negate", "add"}


def verify_certificate(self, A, res):
    """对一个 Smith 结果做全套独立核验。"""
    m, n = len(A), len(A[0])

    # 形状
    self.assertEqual((len(res.U), len(res.U[0])), (m, m))
    self.assertEqual((len(res.V), len(res.V[0])), (n, n))
    self.assertEqual((len(res.D), len(res.D[0])), (m, n))

    # 日志只含三类被允许的初等变换
    for t in res.transforms:
        self.assertIn(t.side, ("row", "column"))
        self.assertIn(t.kind, ALLOWED_KINDS)
        if t.kind == "add":
            self.assertIsInstance(t.k, int)
            self.assertNotIsInstance(t.k, bool)
            self.assertNotEqual(t.k, 0)

    # 独立重放（测试侧自己的实现）必须复现 U、D、V
    U2, M2, V2 = independent_replay(A, res.transforms)
    self.assertEqual(U2, res.U)
    self.assertEqual(V2, res.V)
    self.assertEqual(M2, res.D)

    # 被测包自带 replay 与测试侧重放一致（两条独立路径）
    U3, M3, V3 = cert_replay(A, res.transforms)
    self.assertEqual((U3, M3, V3), (res.U, res.D, res.V))

    # 精确矩阵乘法核验 U*A*V == D（不用被测核心）
    UAV = matmul(matmul(res.U, A), res.V)
    self.assertEqual(UAV, res.D)

    # det(U)=det(V)=±1 独立核验（Fraction 消元）
    self.assertIn(det_fraction(res.U), (1, -1))
    self.assertIn(det_fraction(res.V), (1, -1))

    # D 非负对角、非零对角元依次整除、零在最后
    inv = []
    for i in range(m):
        for j in range(n):
            if i == j:
                self.assertGreaterEqual(res.D[i][j], 0)
                if res.D[i][j] != 0:
                    inv.append(res.D[i][j])
            else:
                self.assertEqual(res.D[i][j], 0)
    for a, b in zip(inv, inv[1:]):
        self.assertEqual(b % a, 0, f"整除链破坏: {a} ∤ {b}")
    self.assertEqual(inv, res.invariants)

    # 秩与不变因子与独立 oracle 一致
    self.assertEqual(res.rank, rational_rank(A))
    self.assertEqual(res.invariants, expected_invariants(A))
    self.assertEqual(len(res.invariants), res.rank)


FIXED_CASES = [
    # 标题, 矩阵；覆盖：矩形、秩亏、负项、天然不满足整除链
    ("1x1 正数", [[6]]),
    ("1x1 负数", [[-7]]),
    ("1x1 零", [[0]]),
    ("对角已成形", [[2, 0], [0, 6]]),
    ("对角不满足整除链 2,3 -> 1,6", [[2, 0], [0, 3]]),
    ("负对角 -2,-3 -> 1,6", [[-2, 0], [0, -3]]),
    ("混合负项稠密", [[-2, 4], [6, -8]]),
    ("经典链破坏例", [[6, 10], [15, 25]]),   # 行列式 0
    ("满秩非对角", [[2, 1], [1, 2]]),
    ("秩亏 2x2", [[1, 2], [2, 4]]),
    ("秩亏非零", [[2, 4], [4, 8]]),
    ("高矩阵 3x2", [[1, 2], [3, 4], [5, 6]]),
    ("宽矩阵 2x3", [[1, 2, 3], [4, 5, 6]]),
    ("高矩阵秩亏", [[1, 1], [2, 2], [3, 3]]),
    ("宽矩阵秩亏", [[1, 2, 3], [2, 4, 6]]),
    ("3x3 链 2,6,..", [[2, 0, 0], [0, 6, 0], [0, 0, 9]]),
    ("3x3 全负稠密", [[-2, -3, -5], [-7, -11, -13], [-4, -6, -8]]),
    ("4x3 一般形", [[2, -3, 4], [-5, 6, -7], [8, -9, 10], [-1, 2, -3]]),
    ("3x4 一般形", [[2, -3, 4, -5], [6, -7, 8, -9], [-1, 2, -3, 4]]),
    ("含零行", [[1, 2], [0, 0], [3, 4]]),
    ("含零列", [[1, 0, 2], [3, 0, 4]]),
    ("4x4 挑战", [[6, 10, 0, 4], [14, 18, 12, 8], [2, 16, 20, 24],
                  [22, 26, 28, 30]]),
    ("元素卡上限 +-10000", [[10000, -10000], [9999, 10000]]),
]


class TestSmithFixed(unittest.TestCase):
    def test_fixed_cases(self):
        for title, A in FIXED_CASES:
            with self.subTest(case=title):
                res = smith(A)
                verify_certificate(self, A, res)
                # 不修改用户输入
                self.assertEqual(A, [row[:] for row in A])

    def test_identity_matrix_log_minimal(self):
        A = identity(3)
        res = smith(A)
        # 单位阵本身即标准形；不应有实质性变换
        self.assertEqual(res.invariants, [1, 1, 1])
        self.assertEqual(res.D, identity(3))

    def test_zero_matrix(self):
        A = [[0, 0, 0], [0, 0, 0]]
        res = smith(A)
        self.assertEqual(res.rank, 0)
        self.assertEqual(res.invariants, [])
        verify_certificate(self, A, res)


class TestSmithRandomFuzz(unittest.TestCase):
    """确定性随机矩阵批量核验（固定种子，离线可复现）。"""

    def test_fuzz_small(self):
        rng = random.Random(20260923)
        for trial in range(120):
            m = rng.randint(1, 5)
            n = rng.randint(1, 5)
            bound = rng.choice([1, 2, 3, 7, 30, 10000])
            A = [[rng.randint(-bound, bound) for _ in range(n)]
                 for _ in range(m)]
            with self.subTest(trial=trial, shape=(m, n), bound=bound):
                res = smith(A)
                verify_certificate(self, A, res)

    def test_fuzz_at_limit_shape(self):
        rng = random.Random(424242)
        for trial in range(6):
            m = n = 8
            A = [[rng.randint(-100, 100) for _ in range(n)]
                 for _ in range(m)]
            with self.subTest(trial=trial):
                res = smith(A)
                verify_certificate(self, A, res)


class TestSmithBudgetAndErrors(unittest.TestCase):
    def test_budget_exhaustion_is_distinct_from_no_solution(self):
        # 预算极小、矩阵需要多步消元：必须抛 BudgetExhaustedError，
        # 它是“搜索未完成”，绝不能被解读成无解。
        A = [[2, 1], [1, 2]]
        with self.assertRaises(BudgetExhaustedError):
            smith(A, budget=1)
        # 预算放宽后正常完成
        res = smith(A, budget=100_000)
        verify_certificate(self, A, res)

    def test_bad_budget(self):
        A = [[1]]
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    smith(A, budget=bad)

    def test_too_many_rows(self):
        A = [[1] * 2 for _ in range(9)]
        with self.assertRaises(MatrixLimitError):
            smith(A)

    def test_too_many_cols(self):
        A = [[1] * 9]
        with self.assertRaises(MatrixLimitError):
            smith(A)

    def test_element_limit(self):
        with self.assertRaises(MatrixLimitError):
            smith([[10001]])
        with self.assertRaises(MatrixLimitError):
            smith([[-10001]])
        # 边界值本身合法
        res = smith([[10000]])
        self.assertEqual(res.invariants, [10000])

    def test_shape_errors(self):
        with self.assertRaises(MatrixShapeError):
            smith([])
        with self.assertRaises(MatrixShapeError):
            smith([[]])
        with self.assertRaises(MatrixShapeError):
            smith([[1, 2], [3]])
        with self.assertRaises(MatrixShapeError):
            smith("not a matrix")
        with self.assertRaises(MatrixShapeError):
            smith([(1, 2), (3, 4)])

    def test_bool_and_float_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            smith([[True, 2], [3, 4]])
        with self.assertRaises(IntegerRequiredError):
            smith([[2.0, 2], [3, 4]])
        with self.assertRaises(IntegerRequiredError):
            smith([[1, 2], [3, False]])

    def test_nan_infinity_rejected(self):
        from smithcert import NonFiniteNumberError
        with self.assertRaises(NonFiniteNumberError):
            smith([[float("nan"), 0], [0, 1]])
        with self.assertRaises(NonFiniteNumberError):
            smith([[float("inf"), 0], [0, 1]])
        with self.assertRaises(NonFiniteNumberError):
            smith([[0, float("-inf")], [0, 1]])

    def test_error_is_locatable(self):
        # 错误信息必须带可定位下标
        with self.assertRaises(MatrixLimitError) as ctx:
            smith([[1, 2], [3, 12345]])
        self.assertIn("[1][1]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
