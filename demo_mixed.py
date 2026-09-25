"""solve_mixed 现场计算演示（固定输入，真实计算，无 sleep/预录）。

演示内容：
  1. 等式 + 同余混合系统：特解、规范化后的解差格基，并独立枚举核验；
  2. 偶奇冲突（不可解）：打印可独立核验的整除障碍；
  3. 增广证书链：replay 独立重放核验 U*Ã*V = D；
  4. 非互素模数与 m_i = 1 无约束处理。
运行：./.venv/bin/python demo_mixed.py（实测远低于 90 秒）。
"""

import itertools
import time

from smithcert import replay, solve_mixed


def check(x, A, b, C, d, m):
    """独立核验一个点是否满足全部约束（纯算术）。"""
    if A:
        for row, bi in zip(A, b):
            if sum(a * v for a, v in zip(row, x)) != bi:
                return False
    if C:
        for row, di, mi in zip(C, d, m):
            if (sum(c * v for c, v in zip(row, x)) - di) % mi != 0:
                return False
    return True


def main():
    t0 = time.perf_counter()

    print("=== 1. 混合系统：x + y + z = 10, x ≡ 1 (mod 3), y ≡ 2 (mod 4) ===")
    A, b = [[1, 1, 1]], [10]
    C, d, m = [[1, 0, 0], [0, 1, 0]], [1, 2], [3, 4]
    res = solve_mixed(A, b, C, d, m)
    assert res.consistent
    print("特解 particular =", res.particular)
    print("解差格基 nullspace_basis =", res.nullspace_basis)
    print("增广系统: %d 行 x %d 列（含 %d 个辅助变量）"
          % (len(res.augmented_A), len(res.augmented_A[0]), res.n_aux))
    assert check(res.particular, A, b, C, d, m)
    for z in res.nullspace_basis:
        # 基向量满足齐次约束：A z = 0 且 C_i·z ≡ 0 (mod m_i)
        assert check(z, [[0] * 3], [0], C, [0] * len(d), m)
    # 独立枚举核验：盒内解集 == 特解 + 基的整数组合
    box = range(-2, 13)
    sols = {x for x in itertools.product(box, repeat=3)
            if check(x, A, b, C, d, m)}
    gen = set()
    for zs in itertools.product(range(-9, 10), repeat=len(res.nullspace_basis)):
        x = tuple(res.particular[j]
                  + sum(t * g[j] for t, g in zip(zs, res.nullspace_basis))
                  for j in range(3))
        if all(v in box for v in x):
            gen.add(x)
    assert sols == gen and sols, (len(sols), len(gen))
    print("盒 [-2,12]^3 内枚举解 %d 个，与基生成集完全一致" % len(sols))

    print()
    print("=== 2. 偶奇冲突：x ≡ 0 (mod 2) 且 x ≡ 1 (mod 2) ===")
    res2 = solve_mixed(None, None, [[1], [1]], [0, 1], [2, 2])
    ob = res2.obstruction
    print("consistent =", res2.consistent, "| obstruction.kind =", ob.kind)
    print("w =", ob.w, " w·Ã =", ob.wA, " w·b̃ =", ob.wb,
          " modulus =", ob.modulus, " remainder =", ob.remainder)
    # 障碍独立核验：w·Ã 各分量被 modulus 整除，w·b̃ 不然
    assert all(v % ob.modulus == 0 for v in ob.wA)
    assert ob.wb % ob.modulus == ob.remainder != 0
    print("障碍核验通过：w·Ã ≡ 0 (mod %d) 而 w·b̃ ≢ 0，故无整数解"
          % ob.modulus)

    print()
    print("=== 3. 增广证书链独立重放 ===")
    sr = res.solve_result
    U, M, V = replay(res.augmented_A, sr.transforms)
    assert U == sr.U and M == sr.D and V == sr.V
    print("replay(Ã, transforms) 重放 %d 步变换，U*Ã*V = D 核验通过"
          % len(sr.transforms))

    print()
    print("=== 4. 非互素模数与 m_i = 1 ===")
    res4 = solve_mixed(None, None, [[1], [1], [1]], [1, 3, 999], [4, 6, 1])
    print("x ≡ 1 (mod 4), x ≡ 3 (mod 6), x ≡ 999 (mod 1[无约束])")
    print("特解 =", res4.particular, " 基 =", res4.nullspace_basis,
          " 丢弃的模1同余下标 =", res4.dropped)
    assert res4.particular[0] % 12 == 9
    assert res4.nullspace_basis == [[12]]
    assert res4.dropped == (2,)

    dt = time.perf_counter() - t0
    print()
    print("全部现场计算与核验通过，用时 %.3f 秒（预算 90 秒）" % dt)


if __name__ == "__main__":
    main()
