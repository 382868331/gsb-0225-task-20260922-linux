"""有限整数上下界内的 Smith 求解现场演示。

运行：python demo_iteration.py
无网络、无第三方依赖、无 sleep、无预录答案；打印独立 Ax=b 与边界核验。
"""

from smithcert import (
    solve_bounded_integer,
    solve_integer,
)


def matvec(A, x):
    return [sum(a * v for a, v in zip(row, x)) for row in A]


def show_result(title, result, A):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("status        :", result.status)
    print("bounds        :", list(zip(result.lower, result.upper)))
    print("particular x0 :", result.particular)
    print("null basis    :", result.nullspace_basis)
    if result.feasible:
        print("coefficients  :", result.coefficients)
        print("solution      :", result.solution)
        print("L1 objective  :", result.objective)
        print("A*x           :", result.Ax)
        print("b             :", result.b)
        print("Ax == b       :", matvec(A, result.solution) == result.b)
        print("bounds ok     :", all(
            lo <= v <= hi
            for v, lo, hi in zip(result.solution, result.lower, result.upper)
        ))
    print()


def main():
    # 3 个变量、2 个约束、1 维整数零空间。目标先最小化 |x1|+|x2|+|x3|，
    # 并列时按 (x1,x2,x3) 字典序。
    A = [
        [1, 1, 1],
        [1, -1, 2],
    ]
    b = [3, 2]
    lower = [-4, -4, -4]
    upper = [5, 5, 5]
    result = solve_bounded_integer(A, b, lower, upper, budget=10000)
    show_result("演示 1：最优有界整数解", result, A)

    # 同一方程但盒为空：整数方程本身可解，界内不可行必须明确是 infeasible。
    empty = solve_bounded_integer(
        A, b, [4, 4, 4], [5, 5, 5], budget=10000,
    )
    show_result("演示 2：方程可解但有限盒内无解", empty, A)

    # 有一个方向无界：本题不冒充可行或不可行。
    unbounded = solve_bounded_integer(
        A, b, [None, -4, -4], [5, 5, 5], budget=10000,
    )
    print("=" * 72)
    print("演示 3：无界输入状态")
    print("=" * 72)
    print("status:", unbounded.status, "(不是 infeasible，也不截断特解)")
    print()

    # 极小预算展示“已有可行解但未证最优”。
    partial = solve_bounded_integer(A, b, lower, upper, budget=4)
    show_result("演示 4：预算耗尽且已有当前可行解", partial, A)

    # 原无界 Diophantine 接口保持可用。
    general = solve_integer(A, b)
    print("=" * 72)
    print("演示 5：原 solve_integer 接口仍返回整数通解结构")
    print("=" * 72)
    print("consistent :", general.consistent)
    print("x0         :", general.particular)
    print("null basis :", general.nullspace_basis)


if __name__ == "__main__":
    main()
