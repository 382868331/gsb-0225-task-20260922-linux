"""固定输入的离线演示（python demo.py）。

真实计算并打印：
  1) 一个含负项、矩形、秩亏矩阵的 Smith 证书（精确核验 U*A*V=D、
     det(U)=det(V)=±1、整除链、零在最后）；
  2) 该矩阵一个相容整数方程组的特解与整数零空间基（精确回代）；
  3) 一个“有理解但无整数解”的系统及其不可解余数证据；
  4) 一个实际触发的输入拒绝边界（元素超上限，带定位信息）；
  5) 链复形整同调：圆（H1=Z）、填充三角形（可缩）、含 Z/2 挠的
     代数例，自由秩/挠系数均用独立 Fraction 秩交叉核验；
  6) 一个 d_{p-1} d_p != 0 的非法链复形被拒绝（带定位，不留半状态）。

全部输入为文件内确定性常量，无网络、无第三方依赖、无 sleep、
无预录结果，运行时间远小于 90 秒。
"""

from fractions import Fraction

from smithcert import (
    BoundaryProductError,
    MatrixLimitError,
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


def rational_rank(A):
    """独立 Fraction 高斯消元求秩（仅用于 demo 的交叉核验）。"""
    m, n = len(A), len(A[0])
    X = [[Fraction(v) for v in row] for row in A]
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, m) if X[i][col] != 0), None)
        if pivot is None:
            continue
        X[r], X[pivot] = X[pivot], X[r]
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


def show_homology(title, boundaries):
    """打印一条链复形的各阶同调，并用独立有理秩核验自由秩。"""
    print("-" * 70)
    print(title)
    for idx, d in enumerate(boundaries, start=1):
        show_matrix(f"d_{idx} ({len(d)}x{len(d[0])})", d)
    res = homology(boundaries)
    D = res.max_degree
    dims = [len(boundaries[0])] + [len(d[0]) for d in boundaries]
    qranks = [rational_rank(d) for d in boundaries]
    for p, g in enumerate(res.groups):
        r_p = qranks[p - 1] if p >= 1 else 0
        r_next = qranks[p] if p < D else 0
        expect_free = dims[p] - r_p - r_next
        print(f"H_{p} = {g.describe():<12} 自由秩={g.free_rank} "
              f"挠系数={list(g.torsion)} "
              f"[独立秩核验 m_{p}-r_{p}-r_{p+1}={dims[p]}-{r_p}-{r_next}"
              f"={expect_free}，{'一致' if g.free_rank == expect_free else '不一致'}]")
    return res


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
    print("演示 5：自由链复形整同调——圆（维度 1）")
    print("=" * 70)
    # 三角形的三条有向边、三个顶点（空心圆周，无 2-单形）。
    circle_d1 = [
        [-1, -1, 0],
        [1, 0, -1],
        [0, 1, 1],
    ]
    r5 = show_homology("空心三角形（≃ S^1）：C1=Z^3 --d1--> C0=Z^3",
                       [circle_d1])
    assert r5.betti_numbers == (1, 1) and r5.torsion_coefficients == ((), ())
    # 手算：d1 秩 2，故 b0=3-2=1、b1=3-2=1，无挠。
    print("手算核验：rank(d1)=2 => b0=b1=1，d1 不变因子全 1 => 无挠。")

    print()
    print("=" * 70)
    print("演示 6：填充三角形（维度 2，可缩）")
    print("=" * 70)
    filled_d2 = [[1], [-1], [1]]  # 面边界 e01 - e02 + e12
    r6 = show_homology(
        "填充三角形：C2=Z --d2--> C1=Z^3 --d1--> C0=Z^3",
        [circle_d1, filled_d2],
    )
    assert r6.betti_numbers == (1, 0, 0)
    # 独立核验相邻边界复合确为零
    print("链条件独立复核 d1*d2 =",
          [row for row in matmul(circle_d1, filled_d2)], "（全零）")
    print("手算核验：d2 的列张成 ker d1 => H1=0；H2=ker d2=0；H0=Z。")

    print()
    print("=" * 70)
    print("演示 7：含 Z/2 挠的代数例（维度 2）")
    print("=" * 70)
    z2_d1, z2_d2 = [[0]], [[2]]
    r7 = show_homology("Moore 风格：C2=Z --[2]--> C1=Z --[0]--> C0=Z",
                       [z2_d1, z2_d2])
    assert r7.torsion_coefficients == ((), (2,), ())
    # 独立按定义核验：H1 = ker d1 / im d2 = Z / 2Z
    print("手算核验：ker d1=Z，im d2=2Z，故 H1 = Z/2Z = Z/2；H0=Z。")
    # 再展示一个自由与挠共存的例子
    show_homology("自由+挠：d1=[0,1,0] (1x3), d2 首列为 (2,0,0)^T",
                  [[[0, 1, 0]], [[2], [0], [0]]])

    print()
    print("=" * 70)
    print("演示 8：非法链复形被拒绝（d_1 * d_2 != 0，不留半状态）")
    print("=" * 70)
    bad_d1, bad_d2 = [[1, 0]], [[2], [0]]
    print("d_1 =", bad_d1, " d_2 =", bad_d2,
          " 乘积 =", matmul(bad_d1, bad_d2))
    saved1, saved2 = [row[:] for row in bad_d1], [row[:] for row in bad_d2]
    try:
        homology([bad_d1, bad_d2])
    except BoundaryProductError as e:
        print("捕获 BoundaryProductError，可定位信息 ->", e)
    else:
        raise AssertionError("非法链复形未被拒绝")
    print("调用后输入未被修改:",
          bad_d1 == saved1 and bad_d2 == saved2)
    # 同一批输入修正为合法复形后可正常计算（拒绝不污染后续调用）
    print("修正 d_2 为 [[0],[0]] 后重新计算 ->")
    fixed = show_homology("修正后的复形", [bad_d1, [[0], [0]]])
    assert fixed.betti_numbers == (0, 1, 1)


if __name__ == "__main__":
    main()
