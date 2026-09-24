"""有限自由链复形的整同调（利用既有 Smith 标准形证书）。

输入一条维度 D ≤ 2 的自由阿贝尔链复形的边界矩阵序列
``[d_1, d_2, ..., d_D]``（题面限定 D ≤ 2）：

    C_D --d_D--> C_{D-1} --d_{D-1}--> ... --d_1--> C_0

其中 ``d_p`` 为 m_{p-1} × m_p 整数矩阵（C_p 维数为 m_p）。先校验：

  1. 每个矩阵形状/整数性/元素上限（沿用 smith() 的同一契约）；
  2. 相邻矩阵形状衔接：d_p 的列数等于 d_{p+1} 的行数；
  3. 链复形条件 **d_{p-1} * d_p = 0**（任一非零复合都带定位报错）。

再对每个 d_p 调用现成的 smith()，由 Smith 标准形读取同调：

  设 r_p = rank(d_p)（r_0 = r_{D+1} = 0），d_p 的非零不变因子为
  s_{p,1} | s_{p,2} | ... | s_{p,r_p}，则

      H_p(C) = ker d_p / im d_{p+1}
             ≅ Z^(m_p - r_p - r_{p+1})
               ⊕ ⨁_{i: s_{p+1,i} > 1} Z / s_{p+1,i}

即第 p 阶自由秩（Betti 数）为 m_p - r_p - r_{p+1}，挠系数为
d_{p+1} 的 Smith 不变因子中大于 1 者（保持整除链顺序）；H_D 无挠
系数来源（d_{D+1} 视为零映射）。

全程精确整数运算；不另写消元算法，直接复用 smith() 的证书结果，
同一份证书随结果返回以便外部独立复核。
"""

from dataclasses import dataclass, field

from .errors import (
    BoundaryProductError,
    DimensionMismatchError,
    MatrixLimitError,
    MatrixShapeError,
)
from .smith import smith, _DEFAULT_BUDGET
from .validate import validate_matrix

_MAX_CHAIN_DIM = 2
_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000


@dataclass(frozen=True)
class HomologyGroup:
    """单个次数 p 的整同调群 H_p。

    degree:       次数 p；
    free_rank:    自由部分秩（Betti 数 b_p）；
    torsion:      挠系数元组（均 > 1，依次满足整除链），群为
                  Z^free_rank ⊕ Z/t_1 ⊕ ... ⊕ Z/t_k；
    空自由秩且无挠系数即平凡群 0。
    """

    degree: int
    free_rank: int
    torsion: tuple = ()

    @property
    def is_trivial(self):
        """H_p 是否为平凡群 0。"""
        return self.free_rank == 0 and not self.torsion

    def describe(self):
        """形如 'Z^2 ⊕ Z/2'、'0' 的可读记号。"""
        parts = []
        if self.free_rank == 1:
            parts.append("Z")
        elif self.free_rank > 1:
            parts.append(f"Z^{self.free_rank}")
        parts.extend(f"Z/{t}" for t in self.torsion)
        return " ⊕ ".join(parts) if parts else "0"


@dataclass(frozen=True)
class HomologyResult:
    """homology() 的结果。

    max_degree: 链复形最高次数 D（同调群次数为 0..D）。
    groups:     HomologyGroup 元组，按次数 0..D 排列。
    ranks:      各边界矩阵的秩 (r_1,...,r_D)，另补 r_0=r_{D+1}=0，
                内部以长度 D+1 元组存放 (r_1,...,r_D, r_{D+1}=0)。
    invariants: 各边界矩阵非零 Smith 不变因子元组（顺序与 ranks 同）。
    certificates: 每个边界矩阵的 SmithResult（U/D/V/日志全在其中）。
    betti_numbers / torsion_coefficients: 各组自由秩与挠系数的便捷元组。
    """

    max_degree: int
    groups: tuple
    ranks: tuple
    invariants: tuple
    certificates: tuple = field(default_factory=tuple)

    @property
    def betti_numbers(self):
        return tuple(g.free_rank for g in self.groups)

    @property
    def torsion_coefficients(self):
        return tuple(g.torsion for g in self.groups)

    def group(self, p):
        """取第 p 阶同调群 H_p（越界抛 IndexError）。"""
        return self.groups[p]


def _matmul_small(A, B):
    """精确整数矩阵乘法（仅用于复合校验，形状已由调用方保证衔接）。"""
    k = len(B)
    q = len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(q)]
            for i in range(len(A))]


def _validate_boundaries(boundaries):
    """校验边界矩阵序列的结构、形状衔接与各矩阵合法性。

    返回各矩阵的 (rows, cols) 表；不复制、不修改用户对象。
    """
    if not isinstance(boundaries, list):
        raise MatrixShapeError(
            f"boundaries: 必须是 list[list[list[int]]]（边界矩阵序列），"
            f"收到 {type(boundaries).__name__}"
        )
    if len(boundaries) < 1:
        raise MatrixShapeError(
            "boundaries: 至少提供 1 个边界矩阵 d_1（本接口维度 1..2）"
        )
    if len(boundaries) > _MAX_CHAIN_DIM:
        raise MatrixLimitError(
            f"boundaries: 链复形维度 {len(boundaries)} 超过上限 "
            f"{_MAX_CHAIN_DIM}（本接口只接受维度 ≤ {_MAX_CHAIN_DIM}）"
        )

    shapes = []
    for idx, d in enumerate(boundaries):
        p = idx + 1  # boundaries[idx] 即 d_p
        m, n = _validate_one(d, name=f"d_{p}")
        shapes.append((m, n))

    # 形状衔接：d_p 是 m_{p-1}×m_p，d_{p+1} 是 m_p×m_{p+1}，
    # 即 d_p 的列数必须等于 d_{p+1} 的行数。
    for idx in range(len(shapes) - 1):
        cols_p = shapes[idx][1]
        rows_next = shapes[idx + 1][0]
        if cols_p != rows_next:
            raise DimensionMismatchError(
                f"d_{idx + 1} 的列数 {cols_p} 与 d_{idx + 2} 的行数 "
                f"{rows_next} 不衔接（d_p: m_{{p-1}}×m_p，"
                f"需 m_p 一致才能复合）"
            )
    return shapes


def _validate_one(d, *, name):
    """单个边界矩阵走既有矩阵校验（非 list 行也给出可定位信息）。"""
    if not isinstance(d, list):
        raise MatrixShapeError(
            f"{name}: 边界矩阵必须是 list[list[int]]，收到 {type(d).__name__}"
        )
    return validate_matrix(
        d, name=name,
        max_rows=_MAX_ROWS, max_cols=_MAX_COLS, max_abs=_MAX_ABS,
    )


def _check_chain_condition(boundaries):
    """逐对校验 d_{p-1} * d_p = 0；发现首个非零元即带定位抛错。"""
    for idx in range(1, len(boundaries)):
        prev = boundaries[idx - 1]  # d_p（idx 从 1 起，复合为 d_p d_{p+1}）
        nxt = boundaries[idx]
        P = _matmul_small(prev, nxt)
        for i, row in enumerate(P):
            for j, v in enumerate(row):
                if v != 0:
                    raise BoundaryProductError(
                        f"d_{idx} * d_{idx + 1} != 0：乘积在 [{i}][{j}] "
                        f"处为 {v}（链复形要求相邻边界复合为零；"
                        f"d_{idx} 为 {len(prev)}×{len(prev[0])}，"
                        f"d_{idx + 1} 为 {len(nxt)}×{len(nxt[0])}）"
                    )


def homology(boundaries, *, budget=_DEFAULT_BUDGET):
    """计算有限自由链复形的整同调群（维度 ≤ 2）。

    参数:
      boundaries: ``[d_1, ..., d_D]``，d_p 为 m_{p-1}×m_p 的
                  list[list[int]]；1 ≤ D ≤ 2；矩阵行/列 ≤ 8、
                  元素绝对值 ≤ 10000；必须满足 d_{p-1} d_p = 0。
      budget:     传给每个 smith() 调用的初等变换步数上限（正 int，
                  拒绝 bool）；耗尽抛 BudgetExhaustedError，只表示
                  搜索未完成，不给任何同调结论。

    返回 HomologyResult：次数 0..D 每阶的自由秩（Betti 数）与挠系数
    （>1 的 Smith 不变因子，整除链顺序），并回传每个边界矩阵的完整
    Smith 证书。一次调用要么完整返回要么抛异常，不存在部分结果；
    不修改用户传入的矩阵。

    异常:
      MatrixShapeError：序列或矩阵形状非法；
      DimensionMismatchError：相邻边界矩阵维数不衔接；
      MatrixLimitError：超维度 2 / 超矩阵规模 / 超元素上限；
      BoundaryProductError：某对相邻矩阵复合非零（信息含定位）；
      IntegerRequiredError / NonFiniteNumberError：元素非精确整数；
      BudgetExhaustedError：Smith 消元预算耗尽。
    """
    shapes = _validate_boundaries(boundaries)
    _check_chain_condition(boundaries)

    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget 必须是 int，收到 {type(budget).__name__}")
    if budget <= 0:
        raise ValueError(f"budget 必须为正整数，收到 {budget}")

    D = len(boundaries)

    # 各阶链群维数：m_0 = d_1 的行数；m_p = d_p 的列数（1 ≤ p ≤ D）。
    dims = [shapes[0][0]] + [shapes[idx][1] for idx in range(D)]

    certificates = tuple(smith(d, budget=budget) for d in boundaries)
    ranks = tuple(cert.rank for cert in certificates) + (0,)  # r_1..r_D, r_{D+1}=0
    invariants = tuple(tuple(cert.invariants) for cert in certificates)

    groups = []
    for p in range(D + 1):
        r_p = ranks[p - 1] if p >= 1 else 0          # rank(d_p)，d_0 不存在
        r_next = ranks[p]                              # rank(d_{p+1})，越界补 0
        free_rank = dims[p] - r_p - r_next
        if free_rank < 0:
            # 形状与复合校验通过后数学上不可能发生；显式报错而非静默。
            raise AssertionError(
                f"内部错误：H_{p} 的自由秩算出 {free_rank} < 0"
            )
        # H_p 的挠来自 d_{p+1} 的不变因子（>1 者）；H_D 无此项。
        torsion = tuple(s for s in invariants[p] if s > 1) if p < D else ()
        groups.append(HomologyGroup(
            degree=p, free_rank=free_rank, torsion=torsion,
        ))

    return HomologyResult(
        max_degree=D,
        groups=tuple(groups),
        ranks=ranks,
        invariants=invariants,
        certificates=certificates,
    )
