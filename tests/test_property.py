"""确定性伪随机矩阵的批量性质测试（固定种子，可复现，离线）。"""

from __future__ import annotations

import random
import unittest

import _checks as checks
import _reference as ref

from smithcert import integer_solve, matmul, matvec, replay_certificate, smith


class RandomizedPropertyTest(unittest.TestCase):
    def test_smith_on_random_matrices(self) -> None:
        rng = random.Random(20260923)
        checked = 0
        for trial in range(120):
            m = rng.randint(1, 8)
            n = rng.randint(1, 8)
            # 以一定概率制造秩亏/零矩阵
            sparse = rng.random() < 0.25
            A = [
                [
                    0 if (sparse and rng.random() < 0.4) else rng.randint(-10_000, 10_000)
                    for _ in range(n)
                ]
                for _ in range(m)
            ]
            res = smith(A)
            U, D, V = res.U, res.D, res.V

            self.assertEqual(matmul(matmul(U, A), V), D, f"trial={trial}")
            U2, D2, V2 = replay_certificate(A, res.operations)
            self.assertEqual((U2, D2, V2), (U, D, V), f"trial={trial}")
            self.assertIn(ref.ref_det(U), (1, -1))
            self.assertIn(ref.ref_det(V), (1, -1))

            positives = list(res.invariant_factors)
            self.assertEqual(positives, ref.ref_invariant_factors(A), f"trial={trial}")
            self.assertEqual(res.rank, ref.ref_rank(A), f"trial={trial}")
            for a, b in zip(positives, positives[1:]):
                self.assertEqual(b % a, 0)
            # D 的非零对角恰为 positives 且零在最后
            for i in range(m):
                for j in range(n):
                    if i != j:
                        self.assertEqual(D[i][j], 0)
            for i, d in enumerate(positives):
                self.assertEqual(D[i][i], d)
                self.assertGreater(d, 0)
            checked += 1
        self.assertEqual(checked, 120)

    def test_solve_on_random_systems(self) -> None:
        rng = random.Random(424242)
        n_solvable = n_unsolvable = 0
        for trial in range(200):
            m = rng.randint(1, 6)
            n = rng.randint(1, 6)
            A = [[rng.randint(-20, 20) for _ in range(n)] for _ in range(m)]
            if rng.random() < 0.5:
                # 由整数 x 构造 b，必然整解
                x = [rng.randint(-50, 50) for _ in range(n)]
                b = matvec(A, x)
                res = checks.check_solvable_case(self, A, b)
                self.assertTrue(checks.rational_consistent(A, b))
                n_solvable += 1
            else:
                b = [rng.randint(-200, 200) for _ in range(m)]
                res = integer_solve(A, b)
                q_ok = checks.rational_consistent(A, b)
                if res.solvable:
                    self.assertTrue(q_ok)
                    self.assertEqual(matvec(A, list(res.particular_solution)), b)
                    n_solvable += 1
                else:
                    checks.check_unsolvable_case(self, A, b, q_ok)
                    n_unsolvable += 1
        # 两类都必须实际出现，避免测试被空分支“假通过”
        self.assertGreater(n_solvable, 0)
        self.assertGreater(n_unsolvable, 0)


if __name__ == "__main__":
    unittest.main()
