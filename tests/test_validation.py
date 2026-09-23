"""输入契约、错误语义与预算耗尽的测试。"""

from __future__ import annotations

import math
import unittest

from smithcert import (
    BudgetExhaustedError,
    ColAdd,
    InputValidationError,
    RowAdd,
    integer_solve,
    smith,
)


class InputValidationTest(unittest.TestCase):
    def expect_error(self, A, needle: str):
        with self.assertRaises(InputValidationError) as cm:
            smith(A)
        self.assertIn(needle, str(cm.exception))

    def test_reject_nan(self) -> None:
        self.expect_error([[1, float("nan")]], "NaN")

    def test_reject_infinity(self) -> None:
        self.expect_error([[float("inf")]], "Infinity")
        self.expect_error([[float("-inf")]], "Infinity")

    def test_reject_float(self) -> None:
        self.expect_error([[1, 2.0]], "float")

    def test_reject_bool(self) -> None:
        self.expect_error([[True, 2]], "bool")
        self.expect_error([[1, 2], [3, False]], "bool")

    def test_reject_str_and_none(self) -> None:
        self.expect_error([["1"]], "int")
        self.expect_error([[None]], "int")
        self.expect_error("not a matrix", "list/tuple")

    def test_reject_empty_and_ragged(self) -> None:
        self.expect_error([], "空")
        self.expect_error([[]], "空")
        self.expect_error([[1, 2], [3]], "不一致")

    def test_reject_dimension_limit(self) -> None:
        self.expect_error([[0] * 9 for _ in range(9)], "上限")
        self.expect_error([[0] * 9 for _ in range(2)], "上限")

    def test_reject_entry_limit(self) -> None:
        A = [[1] * 4 for _ in range(4)]
        A[2][1] = 10_001
        self.expect_error(A, "上限")
        B = [[1] * 4 for _ in range(4)]
        B[0][3] = -10_001
        self.expect_error(B, "上限")

    def test_error_is_locatable(self) -> None:
        with self.assertRaises(InputValidationError) as cm:
            smith([[0, 0], [0, float("nan")]])
        self.assertIn("A[1][1]", str(cm.exception))

    def test_boundary_values_accepted(self) -> None:
        # 边界内（含恰好 8x8、|a|=10000）必须接受
        A = [[10_000 if i == j else -10_000 for j in range(8)] for i in range(8)]
        res = smith(A)
        self.assertEqual(len(res.U), 8)
        self.assertEqual((len(res.D), len(res.D[0])), (8, 8))
        self.assertEqual(len(res.V), 8)

    def test_tuples_accepted_as_sequences(self) -> None:
        res = smith(((2, 4), (6, 8)))
        self.assertEqual(res.invariant_factors, (2, 4))

    def test_vector_validation(self) -> None:
        with self.assertRaises(InputValidationError):
            integer_solve([[1, 2], [3, 4]], [1])  # 长度不符
        with self.assertRaises(InputValidationError):
            integer_solve([[1, 2], [3, 4]], [1, True])  # bool
        with self.assertRaises(InputValidationError):
            integer_solve([[1, 2], [3, 4]], [1, float("nan")])
        # b 允许大整数（无 abs 上限）
        res = integer_solve([[1]], [10**40])
        self.assertTrue(res.solvable)
        self.assertEqual(res.particular_solution, (10**40,))


class BudgetTest(unittest.TestCase):
    def test_budget_exhausted_is_distinct_error(self) -> None:
        # 预算 1：任何非平凡矩阵在第一次变换后即耗尽
        with self.assertRaises(BudgetExhaustedError):
            smith([[6, 4], [0, 10]], budget=1)

    def test_budget_propagates_to_solve(self) -> None:
        with self.assertRaises(BudgetExhaustedError):
            integer_solve([[6, 4], [0, 10]], [2, 2], budget=1)

    def test_budget_exhausted_is_not_unsolvable(self) -> None:
        # 同一可解系统：预算耗尽抛异常；给足预算则正常证明可解
        A, b = [[6, 4], [0, 10]], [10, 10]
        with self.assertRaises(BudgetExhaustedError):
            integer_solve(A, b, budget=1)
        res = integer_solve(A, b)
        self.assertTrue(res.solvable)

    def test_invalid_budget(self) -> None:
        with self.assertRaises(TypeError):
            smith([[1]], budget=1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            smith([[1]], budget=0)

    def test_budget_error_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(BudgetExhaustedError, RuntimeError))


class OperationTest(unittest.TestCase):
    def test_valid_construction(self) -> None:
        self.assertEqual(RowAdd(1, 0, 3).multiple, 3)
        self.assertEqual(ColAdd(0, 2, -5).source, 2)

    def test_reject_bad_indices_and_multiples(self) -> None:
        for bad in (-1,):
            with self.assertRaises(ValueError):
                RowAdd(0, bad, 1)
        with self.assertRaises(TypeError):
            RowAdd(0, 1, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RowAdd(0, 0, 2)  # 同行
        with self.assertRaises(ValueError):
            RowAdd(0, 1, 0)  # 空操作
        with self.assertRaises(ValueError):
            ColAdd(2, 2, -1)

    def test_nan_is_not_an_int_either(self) -> None:
        self.assertTrue(math.isnan(float("nan")))


if __name__ == "__main__":
    unittest.main()
