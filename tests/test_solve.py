"""整数求解 A x = b 与整数零空间基的测试。

期望值由测试侧构造（用选定整数 x 令 b = A x）或由独立 Fraction 消元给出；
不可解例直接核验障碍证书的算术（u·A 被 d 整除而 d ∤ u·b；或 u·A=0
而 u·b≠0），不调用被测核心产生期望值。
"""

from __future__ import annotations

import unittest

import _checks as checks

from smithcert import integer_solve


class IntegerSolveTest(unittest.TestCase):
    def test_square_full_rank(self) -> None:
        A = [[2, 4], [6, 8]]
        b = [6, 14]  # = A (1,1)
        res = checks.check_solvable_case(self, A, b, nullity=0)
        self.assertEqual(res.particular_solution, (1, 1))

    def test_wide_rank_deficient_with_nullspace(self) -> None:
        A = [[1, 2, 3], [4, 5, 6]]
        b = [5, 11]  # = A (1,-1,2)
        checks.check_solvable_case(self, A, b, nullity=1)

    def test_single_row_gcd_solvable(self) -> None:
        checks.check_solvable_case(self, [[6, 9]], [3], nullity=1)

    def test_single_row_gcd_unsolvable_residue(self) -> None:
        # 6x1+9x2=2：Q 上有解、Z 上无解，余数 2
        res = checks.check_unsolvable_case(self, [[6, 9]], [2], True)
        self.assertEqual(res.obstruction["residue"], 2)

    def test_1x1_residue(self) -> None:
        res = checks.check_unsolvable_case(self, [[2]], [1], True)
        self.assertEqual(res.obstruction["pivot"], 2)
        self.assertEqual(res.obstruction["residue"], 1)

    def test_dependent_rows_zero_obstruction(self) -> None:
        # 两行相同，b 不匹配：u·A=0 而 u·b≠0；Q 上亦无解
        checks.check_unsolvable_case(self, [[2, 2], [2, 2]], [2, 4], False)

    def test_overdetermined_consistent(self) -> None:
        checks.check_solvable_case(self, [[1], [1], [1]], [5, 5, 5], nullity=0)

    def test_overdetermined_inconsistent(self) -> None:
        checks.check_unsolvable_case(self, [[1], [1], [1]], [5, 5, 6], False)

    def test_wide_nullity_two(self) -> None:
        # [2 4 6] x = 2，秩 1，零化度 2
        checks.check_solvable_case(self, [[2, 4, 6]], [2], nullity=2)

    def test_negative_rhs_and_entries(self) -> None:
        A = [[-2, 4], [6, -8]]
        b = [-6, 14]  # = A (1,-1)
        res = checks.check_solvable_case(self, A, b, nullity=0)
        self.assertEqual(res.particular_solution, (1, -1))

    def test_zero_matrix_solvable(self) -> None:
        checks.check_solvable_case(self, [[0, 0], [0, 0]], [0, 0], nullity=2)

    def test_zero_matrix_unsolvable(self) -> None:
        checks.check_unsolvable_case(self, [[0, 0], [0, 0]], [0, -3], False)

    def test_integer_solution_not_rational_rounding(self) -> None:
        # 2x = 1 的唯一有理解是 1/2，任何“取整”都不可能正确；必须报无解
        res = integer_solve([[2]], [1])
        self.assertFalse(res.solvable)
        self.assertEqual(res.obstruction["kind"], "nondivisible_pivot")


if __name__ == "__main__":
    unittest.main()
