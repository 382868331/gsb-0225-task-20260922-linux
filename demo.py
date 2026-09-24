"""固定输入的离线演示（python demo.py）。

真实计算并打印：
  1) 一个含负项、矩形、秩亏矩阵的 Smith 证书（精确核验 U*A*V=D、
     det(U)=det(V)=±1、整除链、零在最后）；
  2) 该矩阵一个相容整数方程组的特解与整数零空间基（精确回代）；
  3) 一个“有理解但无整数解”的系统及其不可解余数证据；
  4) 一个实际触发的输入拒绝边界（元素超上限，带定位信息）；
  5) 低维链复形的整系数同调：圆、填充三角形、含 Z/2 挠的代数例、
     RP² 胞腔链（自由秩用独立的 Fraction 秩核验）；
  6) 一个实际触发的链复形公理拒绝（∂₁·∂₂ ≠ 0，带定位信息）。

全部输入为文件内确定性常量，无网络、无第三方依赖、无 sleep、
无预录结果，运行时间远小于 8 秒。
"""

from fractions import Fraction

from smithcert import (
    MatrixLimitError,
    NotAChainComplexError,
    homology,
    smith,
    solve_integer,
)


# ---- 演示用的独立精确核验小工具（与被测核心分开写）----
def matmul(A, B):
    k = len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k))
             for j in range(len(B[0]))] for i in range(len(A))]


def matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def det(A):
    n = len(A)
    X = [[Fraction(v) for v in row] for row in A]
    sign = 1
    for c in range(n):
        p = next((r for r in range(c, n) if X[r][c] != 0), None)
        if p is None:
            return 0
        if p != c:
            X[c], X[p] = X[p], X[c]
            sign = -sign
        for r in range(c + 1, n):
            f = X[r][c] / X[c][c]
            for j in range(c, n):
                X[r][j] -= f * X[c][j]
    d = Fraction(sign)
    for i in range(n):
        d *= X[i][i]
    return d.numerator  # 整数矩阵行列式必为整数


def show_matrix(name, M):
    print(f"{name} =")
    for row in M:
        print("   ", row)


def rank_fraction(A):
    """演示侧独立的有理数秩（Fraction 高斯消元，不调用被测核心）。"""
    X = [[Fraction(v) for v in row] for row in A]
    m, n = len(X), len(X[0])
    r = 0
    for col in range(n):
        p = next((i for i in range(r, m) if X[i][col] != 0), None)
        if p is None:
            continue
        X[r], X[p] = X[p], X[r]
        piv = X[r][col]
        for c in range(col, n):
            X[r][c] /= piv
        for i in range(m):
            if i != r and X[i][col] != 0:
                f = X[i][col]
                for c in range(col, n):
                    X[i][c] -= f * X[r][c]
        r += 1
    return r


def show_group(grp):
    parts = []
    if grp.free_rank:
        parts.append(f"Z^{grp.free_rank}" if grp.free_rank > 1 else "Z")
    parts += [f"Z/{d}" for d in grp.torsion]
    print(f"  H_{grp.degree} =", " ⊕ ".join(parts) if parts else "0")


def demo_homology(title, boundaries):
    print(title)
    for i, B in enumerate(boundaries):
        show_matrix(f"∂{i + 1}", B)
    res = homology(boundaries)
    for grp in res.groups:
        show_group(grp)
    # 独立核验：自由秩 = dim C_k − rank_Q(∂_k) − rank_Q(∂_{k+1})
    ranks = [rank_fraction(B) for B in boundaries]
    dim = len(boundaries)
    ok = True
    for k, grp in enumerate(res.groups):
        n_k = len(boundaries[0]) if k == 0 else len(boundaries[k - 1][0])
        expect = (n_k - (ranks[k - 1] if k else 0)
                  - (ranks[k] if k < dim else 0))
        ok = ok and (grp.free_rank == expect)
    print("  独立核验 自由秩 = dim C_k − rank(∂_k) − rank(∂_{k+1}) :", ok)
    print()


def main():
    print("=" * 70)
    print("演示 1：Smith 标准形证书（矩形、含负项、秩亏）")
    print("=" * 70)
    # 3x4；第 3 行 = 第 1 行 + 第 2 行，故秩亏为 2；含负项。
    A = [
        [-2, 0, 2, 4],
        [0, 3, 3, -6],
        [-2, 3, 5, -2],
    ]
    show_matrix("A (3x4)", A)
    cert = smith(A)
    show_matrix("D = U*A*V", cert.D)
    print("非零不变因子:", cert.invariants, " 秩 =", cert.rank)
    print("初等变换日志条数:", len(cert.transforms))

    # 独立精确核验
    UAV = matmul(matmul(cert.U, A), cert.V)
    print("独立核验 U*A*V == D :", UAV == cert.D)
    print("独立核验 det(U)     :", det(cert.U), "(要求 ±1)")
    print("独立核验 det(V)     :", det(cert.V), "(要求 ±1)")
    chain_ok = all(b % a == 0 for a, b in
                   zip(cert.invariants, cert.invariants[1:]))
    print("整除链 d_i | d_{i+1}:", chain_ok)
    print("零行在最后          :",
          all(cert.D[i][i] != 0 for i in range(cert.rank)))

    print()
    print("=" * 70)
    print("演示 1b：天然不满足整除链且含负项的对角阵 diag(2, -3)")
    print("=" * 70)
    A2 = [[2, 0], [0, -3]]
    c2 = smith(A2)
    show_matrix("D", c2.D)
    print("不变因子:", c2.invariants,
          "（行列式 abs=-6，s1*s2=6；1|6，整除链成立）")
    print("独立核验 U*A*V == D :",
          matmul(matmul(c2.U, A2), c2.V) == c2.D)

    print()
    print("=" * 70)
    print("演示 2：A x = b 的整数特解与整数零空间基")
    print("=" * 70)
    x_true = [1, -1, 2, 0]
    b = matvec(A, x_true)  # 由真实整数解放出 b，b = [2, 3, 5]
    print("b =", b, "（由 x_true =", x_true, "经 A 相乘确定性生成）")
    sol = solve_integer(A, b)
    print("整数可解:", sol.consistent)
    print("一个整数特解 x0 =", sol.particular)
    print("精确回代 A*x0   =", matvec(A, sol.particular), "== b :",
          matvec(A, sol.particular) == b)
    print("整数零空间基（%d 个向量，维数 n-rank = %d）:"
          % (len(sol.nullspace_basis), len(A[0]) - sol.rank))
    for z in sol.nullspace_basis:
        print("   z =", z, "  A*z =", matvec(A, z))
    print("通解 = x0 + k1*z1 + k2*z2（k1,k2 为任意整数，全程无取整）")

    print()
    print("=" * 70)
    print("演示 3：有理解但无整数解——不可解余数（不是搜索失败）")
    print("=" * 70)
    # det = -4 != 0，故在有理数上必有解；但 Smith 不变因子为 2,2，
    # 整数可解性要求 2 | (U b) 的每个对应分量。
    A3 = [[2, 4], [2, 2]]
    b3 = [1, 0]
    show_matrix("A3", A3)
    print("b3 =", b3)
    s3 = solve_integer(A3, b3)
    print("整数可解:", s3.consistent,
          "（行列式 det(A3) =", det(A3), "≠ 0，有理数解存在）")
    ob = s3.obstruction
    print("障碍类型:", ob.kind)
    print("证书障碍行 w = U[%d] =" % ob.index, ob.w)
    print("w*A3 =", ob.wA, " （每个分量都是模数", ob.modulus, "的倍数）")
    print("w*b3 =", ob.wb, " 除以", ob.modulus, "的余数 =",
          ob.remainder, "≠ 0  => 证明无整数解")
    print("独立复核: 所有 wA 分量被模整除 =",
          all(v % ob.modulus == 0 for v in ob.wA),
          "; wb 余数复算 =", ob.wb % ob.modulus)

    print()
    print("=" * 70)
    print("演示 4：实际触发的拒绝边界（非法输入立即报错，不静默纠正）")
    print("=" * 70)
    bad = [[2, 3], [4, 10001]]
    print("输入矩阵:", bad, "（元素 10001 超过 |元素| <= 10000 上限）")
    try:
        smith(bad)
    except MatrixLimitError as e:
        print("捕获 MatrixLimitError，可定位信息 ->", e)
    else:
        raise AssertionError("边界输入未被拒绝")

    print()
    print("=" * 70)
    print("演示 5：低维链复形的整系数同调（自由秩与挠系数）")
    print("=" * 70)
    # 圆 S¹：3 顶点 3 边，∂e0=v1-v0, ∂e1=v2-v1, ∂e2=v2-v0
    circle_d1 = [[-1, 0, -1], [1, -1, 0], [0, 1, 1]]
    demo_homology("5a. 圆 S¹（3 顶点 3 边，H₀=Z, H₁=Z）", [circle_d1])
    # 填充三角形：在圆上粘 2-单形 σ，∂σ = e0+e1-e2（列向量）
    demo_homology("5b. 填充三角形（圆盘，H₀=Z，其余为 0）",
                  [circle_d1, [[1], [1], [-1]]])
    # 含 Z/2 挠的代数例子：C₁=Z → C₀=Z，∂₁=[2]
    demo_homology("5c. 代数例 ∂₁=[2]（H₀=Z/2 挠）", [[[2]]])
    # RP² 胞腔链：∂₂=[2], ∂₁=[0]，H₁=Z/2
    demo_homology("5d. RP² 胞腔链（H₀=Z, H₁=Z/2）", [[[0]], [[2]]])

    print("=" * 70)
    print("演示 6：实际触发的链复形公理拒绝（∂₁·∂₂ ≠ 0）")
    print("=" * 70)
    bad_d1, bad_d2 = [[1, 0], [0, 1]], [[0], [3]]
    print("∂₁ =", bad_d1, " ∂₂ =", bad_d2,
          "（乘积在 (1,0) 处为 3 ≠ 0，不是链复形）")
    try:
        homology([bad_d1, bad_d2])
    except NotAChainComplexError as e:
        print("捕获 NotAChainComplexError，可定位信息 ->", e)
    else:
        raise AssertionError("非链复形输入未被拒绝")


if __name__ == "__main__":
    main()
