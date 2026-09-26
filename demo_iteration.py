"""有限整数界内 Smith 求解的离线演示。

现场计算并展示：
  1. 找到 L1 最小、再字典序最小的最优解；
  2. 全局整数解存在，但有限界内确定无解；
  3. 无界输入返回 unsupported；
  4. 极小预算下区分“已有可行但未证最优”和“无当前可行解且预算耗尽”。

无网络、无第三方依赖、无 sleep、无预录答案。
"""

from smithcert import (
    BUDGET_EXHAUSTED,
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNSUPPORTED,
    solve_bounded_integer,
)


def matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def show(title, res, A, b, bounds):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("A =", A)
    print("b =", b)
    print("bounds =", bounds)
    print("status:", res.status)
    print("nodes_evaluated:", res.nodes_evaluated,
          "budget_used:", res.budget_used, "/", res.budget_total)
    print("reason:", res.reason)
    if res.solution is not None:
        x = res.solution
        print("solution:", x, " L1 =", res.l1_norm)
        print("certificate A*x =", matvec(A, x), "; expected b =", b,
              "; exact:", matvec(A, x) == b)
        print("bounds check:",
              [lo <= value <= hi for value, (lo, hi) in zip(x, bounds)])
        print("residual field:", res.residual,
              "; bounds field:", res.bounds_check)
    else:
        print("solution: None")
    print()


def main():
    A1 = [[1, 1]]
    b1 = [3]
    bounds1 = [(0, 3), (0, 3)]
    show("1) 最优：最小 L1，同 L1 下字典序最小",
         solve_bounded_integer(A1, b1, bounds1), A1, b1, bounds1)

    A2 = [[2, 2, 3], [1, 0, -1]]
    b2 = [1, 10]
    bounds2 = [(-40, 40), (-40, 40), (-40, 40)]
    show("2) 非平凡零空间：由特解加整数零空间改进到最优",
         solve_bounded_integer(A2, b2, bounds2), A2, b2, bounds2)

    A3 = [[1]]
    b3 = [5]
    bounds3 = [(0, 4)]
    show("3) 确定无解：全局解 x=5 存在，但不在 [0,4] 内",
         solve_bounded_integer(A3, b3, bounds3), A3, b3, bounds3)

    A4 = [[1, 1]]
    b4 = [1]
    bounds4 = [(None, 3), (0, 3)]
    res4 = solve_bounded_integer(A4, b4, bounds4)
    print("=" * 72)
    print("4) 无界输入：返回 unsupported，不冒充不可行")
    print("=" * 72)
    print("status:", res4.status, "; reason:", res4.reason)
    print()

    A5 = A2
    b5 = [1, 10]
    bounds5 = bounds2
    res5 = solve_bounded_integer(A5, b5, bounds5, budget=1)
    print("=" * 72)
    print("5a) 预算=1：已找到 Smith 特解但未证明最优")
    print("=" * 72)
    print("status:", res5.status)
    assert res5.status == FEASIBLE
    print("current solution:", res5.solution)
    print("A*x == b:", matvec(A5, res5.solution) == b5)
    print()

    b6 = [1, 20]
    res6 = solve_bounded_integer(A5, b6, bounds5, budget=1)
    full6 = solve_bounded_integer(A5, b6, bounds5)
    print("=" * 72)
    print("5b) 预算=1：预算耗尽且尚无当前可行解（不是 infeasible）")
    print("=" * 72)
    print("tiny-budget status:", res6.status)
    assert res6.status == BUDGET_EXHAUSTED
    print("tiny-budget solution:", res6.solution)
    print("默认预算完整搜索 status:", full6.status)
    assert full6.status == OPTIMAL
    print("完整搜索最优解:", full6.solution, "L1 =", full6.l1_norm)
    print("A*x == b:", matvec(A5, full6.solution) == b6)
    print("bounds check:", all(full6.bounds_check))


if __name__ == "__main__":
    main()

