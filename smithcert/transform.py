"""初等变换日志（证书）与独立重放。

只允许三类整数幺模初等操作：
  - "swap"   交换两行/两列；
  - "negate" 某行/某列乘 -1；
  - "add"    源行/列的 k 倍加到目标行/列（k 为任意整数）。

每个 Transform 描述对“工作矩阵 M”的一次操作：side="row" 的操作
使 M <- Q M，并对 U 施加同一个行操作（U <- Q U）；side="column" 的
操作使 M <- M P，并对 V 施加同一个列操作（V <- V P）。
日志即证书：replay 不使用 smith() 的任何内部状态，从单位矩阵与 A
出发逐条重放，独立得到 U、V、D，可用于外部核验 U*A*V == D。
"""

from dataclasses import dataclass

from .errors import IntegerRequiredError


@dataclass(frozen=True)
class Transform:
    """一次初等变换记录。

    side:      "row" 或 "column"
    kind:      "swap" | "negate" | "add"
    i, j:      swap 为两个下标；negate 只用 i；add 为 j 行/列的 k 倍
               加到 i 行/列（i 是被修改目标，j 是源）
    k:         add 的整数倍；其余 kind 为 None
    """

    side: str
    kind: str
    i: int
    j: int = -1
    k: "int | None" = None


def _apply(M, t):
    """就地对二维 list M 施加“行”变换 t。"""
    if t.kind == "swap":
        M[t.i], M[t.j] = M[t.j], M[t.i]
    elif t.kind == "negate":
        row = M[t.i]
        for c in range(len(row)):
            row[c] = -row[c]
    elif t.kind == "add":
        target = M[t.i]
        source = M[t.j]
        k = t.k
        for c in range(len(target)):
            target[c] += k * source[c]
    else:  # 构造日志的内部代码保证不会发生
        raise ValueError(f"未知变换类型 {t.kind!r}")


def _apply_column(M, t, ncols):
    """就地对二维 list M 施加“列”变换 t（ncols 为 M 的列数）。"""
    if t.kind == "swap":
        for r in range(len(M)):
            M[r][t.i], M[r][t.j] = M[r][t.j], M[r][t.i]
    elif t.kind == "negate":
        for r in range(len(M)):
            M[r][t.i] = -M[r][t.i]
    elif t.kind == "add":
        k = t.k
        for r in range(len(M)):
            M[r][t.i] += k * M[r][t.j]
    else:
        raise ValueError(f"未知变换类型 {t.kind!r}")


def replay(A, transforms, *, shape=None):
    """独立重放日志。

    输入原始矩阵 A 与变换列表，返回 (U, M, V)，其中 M 是把全部变换
    依次作用于 A 的副本所得矩阵，U、V 从单位阵同步累积。该函数只用
    初等操作定义，不依赖 smith() 内部，因此是真正独立的核验路径。

    shape=(m,n) 可由调用方传入以避免重复测量；不做业务校验，A 假定为
    合法整数矩阵（外部核验代码自行先用 validate_matrix 校验）。
    """
    if shape is None:
        m, n = len(A), len(A[0])
    else:
        m, n = shape

    M = [row[:] for row in A]
    U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for t in transforms:
        if t.side == "row":
            # M <- Q M：对 U 施加同一个行操作，U <- Q U。
            _apply(M, t)
            _apply(U, t)
        elif t.side == "column":
            # M <- M P：对 V 施加同一个列操作，V <- V P。
            # 列操作按日志顺序连乘，故直接作用到 V 的列上，不取逆。
            _apply_column(M, t, n)
            _apply_column(V, t, n)
        else:
            raise ValueError(f"未知变换侧 {t.side!r}")
    return U, M, V


def checked_k(k):
    """内部用：倍乘系数必须是精确 int（拒绝 bool）。"""
    if isinstance(k, bool) or not isinstance(k, int):
        raise IntegerRequiredError(
            f"初等变换倍数必须是精确 int，收到 {type(k).__name__}: {k!r}"
        )
    return k
