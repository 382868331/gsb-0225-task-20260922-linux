"""固定输入演示：整数矩阵 Smith 标准形证书 + 整数可解判定。

运行：python demo.py
全部为仓库内固定数据的真实计算，无网络、无 sleep、无预录结果。
"""

from __future__ import annotations

from smithcert import (
    BudgetExhaustedError,
    InputValidationError,
    determinant,
    integer_solve,
    matmul,
    matvec,
    replay_certificate,
    smith,
)


def fmt_mat(M: list[list[int]]) -> str:
    width = max(len(str(v)) for row in M for v in row)
    return "\n".join(
        "      [" + ", ".join(f"{v:>{width}}" for v in row) + "]" for row in M
    )


def fmt_vec(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def main() -> None:
    print("=" * 72)
    print("演示 1：Smith 标准形证书（朴素对角线不满足整除链的例子）")
    print("=" * 72)
    # 注意：若只做朴素对角化，6 和 10 之间不存在整除关系（10 % 6 != 0）。
    A = [[6, 4], [0, 10]]
    cert = smith(A)
    print("A =")
    print(fmt_mat(A))
    print("U =")
    print(fmt_mat(cert.U))
    print("D =")
    print(fmt_mat(cert.D))
    print("V =")
    print(fmt_mat(cert.V))
    print(f"不变因子      : {list(cert.invariant_factors)}  (30 % 2 == 0，整除链成立)")
    print(f"变换日志条数  : {len(cert.operations)}（仅行/列交换、变号、加整数倍）")
    print(f"操作日志      : {[repr(op) for op in cert.operations]}")

    # 独立核验：精确矩阵乘法 + 独立 Bareiss 行列式 + 独立日志重放
    assert matmul(matmul(cert.U, A), cert.V) == cert.D, "U*A*V != D"
    assert determinant(cert.U) in (1, -1)
    assert determinant(cert.V) in (1, -1)
    U2, D2, V2 = replay_certificate(A, cert.operations)
    assert (U2, D2, V2) == (cert.U, cert.D, cert.V)
    print("核验          : U*A*V == D 精确成立；det(U)=±1、det(V)=±1；日志独立重放一致")
    print(f"               det(U) = {determinant(cert.U)}, det(V) = {determinant(cert.V)}")

    print()
    print("=" * 72)
    print("演示 2：整数可解系统的特解（满秩方阵）")
    print("=" * 72)
    b = [10, 10]
    sol = integer_solve(A, b)
    print(f"A x = {b}")
    print(f"可解          : {sol.solvable}")
    print(f"一个整数特解  : x0 = {fmt_vec(sol.particular_solution)}")
    assert matvec(A, list(sol.particular_solution)) == b
    print(f"核验 A*x0     : {matvec(A, list(sol.particular_solution))} == {b}")
    print(f"零空间基      : {list(sol.null_space_basis)}（满秩，空）")

    print()
    print("=" * 72)
    print("演示 3：不可解系统的余数证书（不是有理解取整）")
    print("=" * 72)
    b2 = [2, 2]
    bad = integer_solve(A, b2)
    print(f"A x = {b2}")
    print(f"可解          : {bad.solvable}")
    ob = bad.obstruction
    u = ob["certificate_row_U"]
    print(f"障碍类型      : {ob['kind']}")
    print(f"不可整除      : 对角元 {ob['pivot']} 不整除 (U b)[{ob['transformed_row']}]"
          f" = {ob['transformed_rhs_entry']}，余数 = {ob['residue']}")
    print(f"整数证书      : U 的该行 u = {fmt_vec(u)}")
    uA = [sum(u[i] * A[i][j] for i in range(2)) for j in range(2)]
    ub = sum(u[i] * b2[i] for i in range(2))
    print(f"               u·A = {uA}（每项都被 {ob['pivot']} 整除），"
          f"u·b = {ub}（余数 {ub % ob['pivot']} != 0），故无整数解")
    assert all(v % ob["pivot"] == 0 for v in uA)
    assert ub % ob["pivot"] == ob["residue"] != 0

    print()
    print("=" * 72)
    print("演示 4：秩亏矩形系统 —— 特解 + 整数零空间基")
    print("=" * 72)
    B = [[1, 2, 3], [4, 5, 6]]
    b3 = [5, 11]
    rs = integer_solve(B, b3)
    print("B =")
    print(fmt_mat(B))
    print(f"B x = {b3}")
    print(f"秩/零化度     : {rs.rank} / {rs.nullity}")
    print(f"一个整数特解  : x0 = {fmt_vec(rs.particular_solution)}")
    assert matvec(B, list(rs.particular_solution)) == b3
    print(f"核验 B*x0     : {matvec(B, list(rs.particular_solution))} == {b3}")
    for z in rs.null_space_basis:
        print(f"零空间基向量  : z = {fmt_vec(z)}，B*z = {matvec(B, list(z))}（精确为零向量）")
        assert matvec(B, list(z)) == [0, 0]

    print()
    print("=" * 72)
    print("演示 5：实际触发的输入拒绝边界（可定位、不静默纠正）")
    print("=" * 72)
    for label, bad_input, bad_b in [
        ("NaN 元素", [[2, float("nan")]], [1]),
        ("bool 元素", [[True, 2]], [1]),
        ("元素超上限", [[10_001]], [1]),
        ("b 长度不符", [[1, 2]], [1, 2, 3]),
    ]:
        try:
            integer_solve(bad_input, bad_b)
        except InputValidationError as exc:
            print(f"  - {label:8s} -> InputValidationError: {exc}")
        else:  # pragma: no cover - 演示用，不应发生
            raise AssertionError(f"{label} 本应被拒绝")

    print()
    print("-" * 72)
    print("预算耗尽与证明无解严格分开（同一可解系统 A x = [10,10]）：")
    try:
        integer_solve(A, [10, 10], budget=1)
    except BudgetExhaustedError as exc:
        print(f"  budget=1   -> BudgetExhaustedError: {exc}")
    sol_default = integer_solve(A, [10, 10])
    print(f"  默认预算   -> solvable={sol_default.solvable}，"
          f"x0={fmt_vec(sol_default.particular_solution)}")
    print()
    print("全部演示完成。")


if __name__ == "__main__":
    main()
