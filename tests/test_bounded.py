"""solve_bounded_integer() 测试：有限界、L1/字典序、预算与异常路径。"""

from itertools import product
import random
import unittest

from smithcert import (
    BOUNDED_STATUS_BUDGET_EXHAUSTED,
    BOUNDED_STATUS_FEASIBLE,
    BOUNDED_STATUS_INFEASIBLE,
    BOUNDED_STATUS_OPTIMAL,
    BOUNDED_STATUS_UNSUPPORTED,
    DimensionMismatchError,
    IntegerRequiredError,
    MatrixLimitError,
    NonFiniteNumberError,
    solve_bounded_integer,
)
from tests._oracle import matvec


def brute_force(A, b, lower, upper):
    """独立小盒全枚举参考：先 L1，再按普通整数元组字典序。"""
    candidates = []
    for x in product(*(range(lo, hi + 1)
                       for lo, hi in zip(lower, upper))):
        if matvec(A, list(x)) == list(b):
            candidates.append((sum(abs(v) for v in x), x))
    if not candidates:
        return None
    return min(candidates)


def verify_result(testcase, A, b, lower, upper, result):
    expected = brute_force(A, b, lower, upper)
    if expected is None:
        testcase.assertEqual(result.status, BOUNDED_STATUS_INFEASIBLE)
        testcase.assertFalse(result.feasible)
        testcase.assertIsNone(result.solution)
        return

    objective, solution = expected
    testcase.assertEqual(result.status, BOUNDED_STATUS_OPTIMAL)
    testcase.assertTrue(result.feasible)
    testcase.assertTrue(result.optimal)
    testcase.assertEqual(result.solution, list(solution))
    testcase.assertEqual(result.objective, objective)
    testcase.assertEqual(result.Ax, list(b))
    testcase.assertTrue(result.equation_check)
    testcase.assertTrue(result.bounds_check)
    testcase.assertTrue(all(lo <= v <= hi
                            for v, lo, hi in zip(result.solution,
                                                 lower, upper)))
    recomposed = [
        p + sum(c * z[i] for c, z in zip(result.coefficients,
                                         result.nullspace_basis))
        for i, p in enumerate(result.particular)
    ]
    testcase.assertEqual(recomposed, result.solution)


class TestBoundedFixed(unittest.TestCase):
    def test_l1_then_lexicographic(self):
        # x+y=2 的 L1 最小候选为 (0,2)、(1,1)、(2,0)，字典序最小是 (0,2)。
        result = solve_bounded_integer(
            [[1, 1]], [2], [-3, -3], [4, 4], budget=1000,
        )
        self.assertEqual(result.status, BOUNDED_STATUS_OPTIMAL)
        self.assertEqual(result.solution, [0, 2])
        self.assertEqual(result.objective, 2)
        self.assertEqual(result.Ax, [2])
        self.assertTrue(result.equation_check)
        self.assertTrue(result.bounds_check)

    def test_negative_variables_and_lexicographic_tie(self):
        # 候选 (-1,0) 与 (0,-1) 的 L1 相同，字典序选择 (-1,0)。
        result = solve_bounded_integer(
            [[1, -1]], [-1], [-2, -2], [2, 2], budget=1000,
        )
        self.assertEqual(result.solution, [-1, 0])
        self.assertEqual(result.objective, 1)

    def test_zero_rank_unique_solution(self):
        result = solve_bounded_integer(
            [[2, 0], [0, 3]], [4, 9], [0, 0], [10, 10],
        )
        self.assertEqual(result.status, BOUNDED_STATUS_OPTIMAL)
        self.assertEqual(result.solution, [2, 3])
        self.assertEqual(result.coefficients, [])

    def test_proven_infeasible_inside_bounds(self):
        result = solve_bounded_integer(
            [[2, 4]], [3], [-10, -10], [10, 10],
        )
        self.assertEqual(result.status, BOUNDED_STATUS_INFEASIBLE)
        self.assertIsNotNone(result.obstruction)

    def test_equation_consistent_but_box_empty(self):
        result = solve_bounded_integer(
            [[1, 1]], [2], [3, 3], [4, 4],
        )
        self.assertEqual(result.status, BOUNDED_STATUS_INFEASIBLE)
        self.assertFalse(result.feasible)

    def test_reversed_bounds_are_proven_infeasible(self):
        result = solve_bounded_integer(
            [[1]], [1], [2], [1],
        )
        self.assertEqual(result.status, BOUNDED_STATUS_INFEASIBLE)


class TestBoundedBruteForce(unittest.TestCase):
    def test_small_boxes_against_full_enumeration(self):
        cases = [
            ([[1, 1]], [2], [-3, -3], [4, 4]),
            ([[1, -1]], [1], [-2, -2], [3, 3]),
            ([[2, 2]], [2], [-3, -3], [3, 3]),
            ([[1, 1, 1]], [1], [-1, -1, -1], [2, 2, 2]),
            ([[1, 2], [0, 1]], [3, 1], [-2, -2], [3, 3]),
            ([[1, -1, 2], [0, 1, 1]], [1, 0],
             [-2, -2, -2], [2, 2, 2]),
        ]
        for A, b, lower, upper in cases:
            with self.subTest(A=A, b=b, lower=lower, upper=upper):
                result = solve_bounded_integer(
                    A, b, lower, upper, budget=10000,
                )
                verify_result(self, A, b, lower, upper, result)

    def test_random_small_boxes_against_full_enumeration(self):
        rng = random.Random(20260927)
        for trial in range(80):
            n = rng.randint(1, 6)
            m = rng.randint(1, 4)
            span = rng.randint(1, 3)
            center = rng.randint(-2, 2)
            lower = [center for _ in range(n)]
            upper = [center + span for _ in range(n)]
            A = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(m)]
            x = [rng.randint(lo, hi) for lo, hi in zip(lower, upper)]
            b = matvec(A, x)
            with self.subTest(trial=trial, shape=(m, n), A=A, b=b):
                result = solve_bounded_integer(
                    A, b, lower, upper, budget=100000,
                )
                verify_result(self, A, b, lower, upper, result)


class TestBoundedStatuses(unittest.TestCase):
    def test_unbounded_none_is_unsupported(self):
        result = solve_bounded_integer(
            [[1, 1]], [2], [None, 0], [4, 4],
        )
        self.assertEqual(result.status, BOUNDED_STATUS_UNSUPPORTED)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.solution)

    def test_unbounded_infinity_is_unsupported(self):
        result = solve_bounded_integer(
            [[1, 1]], [2], [0, 0], [float("inf"), 4],
        )
        self.assertEqual(result.status, BOUNDED_STATUS_UNSUPPORTED)

    def test_nan_bound_is_rejected(self):
        with self.assertRaises(NonFiniteNumberError):
            solve_bounded_integer(
                [[1]], [1], [0], [float("nan")],
            )

    def test_budget_exhausted_without_current_solution(self):
        result = solve_bounded_integer(
            [[1, 1]], [2], [-3, -3], [4, 4], budget=1,
        )
        self.assertEqual(result.status, BOUNDED_STATUS_BUDGET_EXHAUSTED)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.solution)

    def test_budget_exhausted_with_current_solution(self):
        result = solve_bounded_integer(
            [[1, 1]], [2], [-3, -3], [4, 4], budget=5,
        )
        self.assertEqual(result.status, BOUNDED_STATUS_FEASIBLE)
        self.assertTrue(result.feasible)
        self.assertFalse(result.optimal)
        self.assertEqual(matvec([[1, 1]], result.solution), [2])
        self.assertTrue(result.equation_check)
        self.assertTrue(result.bounds_check)


class TestBoundedInputErrors(unittest.TestCase):
    def test_variable_limit_is_six(self):
        with self.assertRaises(MatrixLimitError):
            solve_bounded_integer(
                [[1] * 7], [1], [0] * 7, [1] * 7,
            )

    def test_bound_length_mismatch(self):
        with self.assertRaises(DimensionMismatchError):
            solve_bounded_integer([[1, 1]], [1], [0], [1, 1])

    def test_bool_and_finite_float_bounds_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            solve_bounded_integer([[1]], [1], [True], [1])
        with self.assertRaises(IntegerRequiredError):
            solve_bounded_integer([[1]], [1], [0.0], [1])

    def test_bad_budget_rejected(self):
        with self.assertRaises(TypeError):
            solve_bounded_integer([[1]], [1], [0], [1], budget=True)
        with self.assertRaises(ValueError):
            solve_bounded_integer([[1]], [1], [0], [1], budget=0)


if __name__ == "__main__":
    unittest.main()
