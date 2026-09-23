"""整数线性方程组 A x = b 的可解判定、特解与整数零空间基。

全部基于 Smith 证书（整数幺模变换），直接在整数环上推导，
**不**经过有理解再取整：

    U A V = D,  c = U b,  D y = c,  x = V y。

* D 的每个非零对角元 d_i 必须整除 c_i，否则无解，余数 ``c_i mod d_i``
  连同 U 的对应行（一条整数线性组合不可解证书）一并返回；
* D 的零行对应 c_i 必须为 0，否则同样无解；
* 特解取自由变量 y_{r..n-1} = 0；
* 整数零空间基取 V 的第 r, ..., n-1 列（D 在这些方向上恰为零）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._basic import matvec, validate_matrix, validate_vector
from ._smith import DEFAULT_BUDGET, smith_normal_form


@dataclass(frozen=True, slots=True)
class IntegerSolveResult:
    """``integer_solve`` 的结果。

    可解时：
      * ``solvable`` 为 True；
      * ``particular_solution`` 为一个整数特解 x0，A x0 = b；
      * ``null_space_basis`` 为整数零空间基（每个向量 z 满足 A z = 0），
        满列秩时为空列表；全部解为 x0 + 基的整数线性组合。
    不可解时：
      * ``solvable`` 为 False；
      * ``particular_solution`` 为 None；
      * ``null_space_basis`` 仍尽量给出（Smith 计算成功时）；
      * ``obstruction`` 描述首个障碍：变换后的行号、非零对角元、余数
        以及 U 的对应行（与 b 的整数线性组合给出障碍）。
    """

    solvable: bool
    particular_solution: tuple[int, ...] | None
    null_space_basis: tuple[tuple[int, ...], ...]
    rank: int
    nullity: int
    obstruction: dict[str, Any] | None
    transformed_rhs: tuple[int, ...] = field(default=())
    invariant_factors: tuple[int, ...] = field(default=())


def integer_solve(
    A: object,
    b: object,
    *,
    budget: int = DEFAULT_BUDGET,
) -> IntegerSolveResult:
    """判定并求解整数线性方程组 A x = b。

    参数经严格输入校验（int、拒绝 bool/NaN/Infinity，矩阵至多 8×8，
    元素绝对值 <= 10000；b 为等长整数向量，取值不设上限）。

    预算耗尽抛 :class:`~smithcert.BudgetExhaustedError`，与“证明无解”严格分开。
    """
    M = validate_matrix(A, "A")
    m, n = len(M), len(M[0])
    rhs = validate_vector(b, m, "b")

    cert = smith_normal_form(M, budget=budget)
    U, V, rank, factors = cert.U, cert.V, cert.rank, cert.invariant_factors
    c = matvec(U, rhs)  # 变换后的右端项，全程整数

    # 零空间基：V 的第 rank..n-1 列
    null_basis = tuple(
        tuple(V[i][k] for i in range(n)) for k in range(rank, n)
    )

    obstruction: dict[str, Any] | None = None

    # 1) 非零对角元的整除性检查（余数即不可解证据）
    for i, d in enumerate(factors):
        # d > 0；Python % 给出非负余数
        rem = c[i] % d
        if rem != 0:
            obstruction = {
                "kind": "nondivisible_pivot",
                "transformed_row": i,
                "pivot": d,
                "transformed_rhs_entry": c[i],
                "residue": rem,
                "certificate_row_U": tuple(U[i]),
                "message": (
                    f"U A V 的第 {i} 个对角元 {d} 不整除 (U b)[{i}] = {c[i]}，"
                    f"余数为 {rem}，故无整数解"
                ),
            }
            break

    # 2) 零行检查：秩之外的变换后方程要求 c_i = 0
    if obstruction is None:
        for i in range(rank, m):
            if c[i] != 0:
                obstruction = {
                    "kind": "zero_row_nonzero_rhs",
                    "transformed_row": i,
                    "pivot": 0,
                    "transformed_rhs_entry": c[i],
                    "residue": c[i],
                    "certificate_row_U": tuple(U[i]),
                    "message": (
                        f"系数矩阵秩为 {rank}，但 U 的第 {i} 行与 A 的各行"
                        f"整数线性组合为零而与 b 的组合为 {c[i]} != 0，故无解"
                    ),
                }
                break

    if obstruction is not None:
        return IntegerSolveResult(
            solvable=False,
            particular_solution=None,
            null_space_basis=null_basis,
            rank=rank,
            nullity=n - rank,
            obstruction=obstruction,
            transformed_rhs=tuple(c),
            invariant_factors=factors,
        )

    # 3) 构造特解：y_i = c_i / d_i（i < r），自由变量取 0，x0 = V y。
    y = [0] * n
    for i, d in enumerate(factors):
        y[i] = c[i] // d
    x0 = matvec(V, y)

    return IntegerSolveResult(
        solvable=True,
        particular_solution=tuple(x0),
        null_space_basis=null_basis,
        rank=rank,
        nullity=n - rank,
        obstruction=None,
        transformed_rhs=tuple(c),
        invariant_factors=factors,
    )
