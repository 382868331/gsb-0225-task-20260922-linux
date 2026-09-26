"""solve_bounded_integer() 测试。

独立参考直接在小有限盒上全枚举，不调用有界搜索；候选解再做
Ax=b 与逐变量边界两项精确检查。
"""

import itertools
import random
import unittest

from smithcert import (
    BUDGET_EXHAUSTED,
    DimensionMismatchError,
    FEASIBLE,
    INFEASIBLE,
    IntegerRequiredError,
    MatrixLimitError,
    MatrixShapeError,
    OPTIMAL,
    UNSUPPORTED,
    solve_bounded_integer,
    solve_integer,
)
from tests._oracle import matvec


def brute_force_optimum(A, b, bounds):
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    feasible = []
    for x in itertools.product(*ranges):
        x = list(x)
        if matvec(A, x) == list(b):
            feasible.append(x)
    if not feasible:
        return None
    return min(feasible, key=lambda x: (sum(abs(v) for v in x), tuple(x)))


def assert_result_certificate(self, res, A, b, bounds):
    self.assertIn(
        res.status,
        (OPTIMAL, FEASIBLE, INFEASIBLE, UNSUPPORTED, BUDGET_EXHAUSTED),
    )
    if res.solution is None:
        self.assertIsNone(res.residual)
        self.assertIsNone(res.bounds_check)
        self.assertIsNone(res.l1_norm)
        return
    self.assertEqual(matvec(A, res.solution), list(b))
    self.assertEqual(res.residual, list(b))
    self.assertEqual(
        res.bounds_check,
        [lo <= v <= hi for v, (lo, hi) in zip(res.solution, bounds)],
    )
    self.assertTrue(all(res.bounds_check))
    self.assertEqual(res.l1_norm, sum(abs(v) for v in res.solution))
    self.assertTrue(all(isinstance(v, int) and not isinstance(v, bool)
                        for v in res.solution))


class TestBoundedFixed(unittest.TestCase):
    def test_l1_then_lexicographic_optimum(self):
        A = [[1, 1]]
        b = [3]
        bounds = [(0, 3), (0, 3)]
        res = solve_bounded_integer(A, b, bounds)
        self.assertEqual(res.status, OPTIMAL)
        self.assertEqual(res.solution, [0, 3])
        self.assertEqual(res.l1_norm, 3)
        assert_result_certificate(self, res, A, b, bounds)

    def test_nullspace_moves_particular_inside_bounds(self):
        A = [[1, -1]]
        b = [2]
        bounds = [(0, 3), (0, 3)]
        res = solve_bounded_integer(A, b, bounds)
        self.assertEqual(res.status, OPTIMAL)
        self.assertEqual(res.solution, [2, 0])
        assert_result_certificate(self, res, A, b, bounds)
        self.assertEqual(matvec(A, res.nullspace_basis[0]), [0])

    def test_zero_objective_when_origin_feasible(self):
        A = [[1, -1, 2], [0, 0, 0]]
        b = [0, 0]
        bounds = [(0, 2), (0, 2), (0, 2)]
        res = solve_bounded_integer(A, b, bounds)
        self.assertEqual(res.status, OPTIMAL)
        self.assertEqual(res.solution, [0, 0, 0])
        assert_result_certificate(self, res, A, b, bounds)

    def test_bounds_can_make_consistent_system_infeasible(self):
        A = [[1]]
        b = [5]
        bounds = [(0, 4)]
        res = solve_bounded_integer(A, b, bounds)
        self.assertEqual(res.status, INFEASIBLE)
        self.assertIsNone(res.solution)
        self.assertIsNotNone(res.particular)

    def test_global_integer_inconsistency_is_still_infeasible(self):
        A = [[2]]
        b = [1]
        bounds = [(0, 4)]
        res = solve_bounded_integer(A, b, bounds)
        self.assertEqual(res.status, INFEASIBLE)
        self.assertIsNotNone(res.obstruction)
        self.assertEqual(res.obstruction.kind, "divisibility")

    def test_unbounded_input_is_unsupported_not_infeasible(self):
        A = [[1, 1]]
        b = [1]
        for bounds in (None, [None, None], [(0, 2), None],
                       [(None, 2), (0, 2)]):
            with self.subTest(bounds=bounds):
                res = solve_bounded_integer(A, b, bounds)
                self.assertEqual(res.status, UNSUPPORTED)
                self.assertIsNone(res.solution)

    def test_feasible_but_unproven_on_budget_stop(self):
        A = [[1, 1]]
        b = [1]
        bounds = [(-1000, 1000), (-1000, 1000)]
        res = solve_bounded_integer(A, b, bounds, budget=1)
        self.assertEqual(res.status, FEASIBLE)
        self.assertIsNotNone(res.solution)
        assert_result_certificate(self, res, A, b, bounds)

    def test_budget_exhausted_without_solution_is_distinct_from_infeasible(self):
        # 全域整数解存在；把盒放在远离 Smith 特解且不能一步判空的位置。
        # 极小预算若未能证完，只允许 budget_exhausted，不能报 infeasible。
        A = [[2, 2, 3], [1, 0, -1]]
        b = [1, 10000]
        bounds = [(-40, 40), (-40, 40), (-40, 40)]
        base = solve_integer(A, b)
        self.assertTrue(base.consistent)
        res = solve_bounded_integer(A, b, bounds, budget=2)
        self.assertIn(res.status, (BUDGET_EXHAUSTED, INFEASIBLE, OPTIMAL))
        if res.status == INFEASIBLE:
            self.assertIsNone(brute_force_optimum(A, b, bounds))
        if res.status == OPTIMAL:
            self.assertEqual(
                res.solution,
                brute_force_optimum(A, b, bounds),
            )


class TestBoundedBruteForceFuzz(unittest.TestCase):
    def test_random_small_boxes_match_full_enumeration(self):
        rng = random.Random(20260926)
        for trial in range(80):
            m = rng.randint(1, 3)
            n = rng.randint(1, 5)
            A = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(m)]
            x_true = [rng.randint(-2, 2) for _ in range(n)]
            b = matvec(A, x_true)
            lo = rng.randint(-3, 0)
            hi = rng.randint(0, 3)
            bounds = [(lo, hi) for _ in range(n)]
            with self.subTest(trial=trial, A=A, b=b, bounds=bounds):
                res = solve_bounded_integer(A, b, bounds)
                expected = brute_force_optimum(A, b, bounds)
                if expected is None:
                    self.assertEqual(res.status, INFEASIBLE)
                    self.assertIsNone(res.solution)
                else:
                    self.assertEqual(res.status, OPTIMAL)
                    self.assertEqual(res.solution, expected)
                    assert_result_certificate(self, res, A, b, bounds)

    def test_six_variable_limit_small_box(self):
        A = [[1, -1, 0, 1, -1, 0], [0, 1, -1, 0, 1, -1]]
        b = [1, 0]
        bounds = [(0, 2)] * 6
        res = solve_bounded_integer(A, b, bounds)
        self.assertEqual(res.status, OPTIMAL)
        self.assertEqual(res.solution, brute_force_optimum(A, b, bounds))
        assert_result_certificate(self, res, A, b, bounds)


class TestBoundedInputErrors(unittest.TestCase):
    def test_more_than_six_variables_rejected(self):
        with self.assertRaises(MatrixLimitError):
            solve_bounded_integer([[1] * 7], [1], [(0, 1)] * 7)

    def test_bounds_length_mismatch(self):
        with self.assertRaises(DimensionMismatchError):
            solve_bounded_integer([[1, 1]], [1], [(0, 1)])

    def test_bad_bounds_shape_and_reversed_interval(self):
        with self.assertRaises(MatrixShapeError):
            solve_bounded_integer([[1]], [1], [(0,)])
        with self.assertRaises(MatrixShapeError):
            solve_bounded_integer([[1]], [1], "bad")
        with self.assertRaises(MatrixShapeError):
            solve_bounded_integer([[1]], [1], [(2, 1)])

    def test_bool_float_bounds_rejected(self):
        with self.assertRaises(IntegerRequiredError):
            solve_bounded_integer([[1]], [1], [(True, 1)])
        with self.assertRaises(IntegerRequiredError):
            solve_bounded_integer([[1]], [1], [(0, 1.0)])

    def test_bad_budget(self):
        for bad in (0, -1, True, 2.5):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    solve_bounded_integer([[1]], [1], [(0, 1)], budget=bad)


if __name__ == "__main__":
    unittest.main()
