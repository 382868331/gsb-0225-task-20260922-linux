"""低维链复形的整系数同调：各阶自由秩（Betti 数）与挠系数。

约定（列约定）：boundaries[k] 是边缘矩阵 ∂_{k+1}: C_{k+1} → C_k，
形状 (dim C_k) × (dim C_{k+1})，第 j 列是第 j 个 (k+1) 维生成元的
边界在 C_k 基下的坐标。链复形维数 ≤ 2，即传入 1 个（∂₁）或
2 个（∂₁, ∂₂）边缘矩阵。

计算前先做两件精确验证：
  1. 相邻矩阵形状衔接：∂_{k+1} 的行数 == ∂_k 的列数；
  2. 链复形公理：∂_k · ∂_{k+1} = 0（精确整数零矩阵），
     违例抛 NotAChainComplexError（含第一个非零元定位）。

随后对每个边缘矩阵复用本库的 Smith 标准形证书。经典结论：
设 ∂_k 的秩为 r_k、∂_{k+1} 的 Smith 不变因子为 d_1|...|d_{r}，则

  H_k ≅ Z^{b_k} ⊕ Z/d_1 ⊕ ... ⊕ Z/d_s，

其中 b_k = dim C_k − r_k − r_{k+1}（∂_0、∂_{dim+1} 视为零映射），
挠系数只保留 d_i > 1 的项（Z/1 是平凡群）。挠系数按整除链顺序给出。
全程精确整数运算，无浮点、无有理取整。
"""

from dataclasses import dataclass, field

from .errors import (
    DimensionMismatchError,
    MatrixLimitError,
    MatrixShapeError,
    NotAChainComplexError,
)
from .smith import smith, _DEFAULT_BUDGET
from .validate import validate_matrix

_MAX_ROWS = 8
_MAX_COLS = 8
_MAX_ABS = 10_000
_MAX_DIMENSION = 2


@dataclass(frozen=True)
class HomologyGroup:
    """某一阶的整系数同调群。

    degree:    阶数 k（0..dimension）。
    free_rank: 自由部分的秩（第 k 个 Betti 数）。
    torsion:   挠系数元组 (d_1, ..., d_s)，每个 d_i > 1 且
               d_i | d_{i+1}；空元组表示无挠。
               即 H_k ≅ Z^free_rank ⊕ Z/d_1 ⊕ ... ⊕ Z/d_s。
    """

    degree: int
    free_rank: int
    torsion: tuple = ()


@dataclass(frozen=True)
class HomologyResult:
    """homology() 的结果。

    dimension:      链复形维数（边缘矩阵个数，1 或 2）。
    groups:         各阶 HomologyGroup，长度 dimension+1。
    boundary_ranks: 各边缘矩阵的秩（长度 dimension）。
    certificates:   各边缘矩阵的 SmithResult 证书（长度 dimension），
                    供外部复核 U*∂*V = D。
    """

    dimension: int
    groups: tuple
    boundary_ranks: tuple
    certificates: tuple = field(default_factory=tuple)


def _first_nonzero_product_entry(A, B):
    """精确计算 A*B，返回第一个（按行优先序）非零元的 (i, j, value)。

    全零返回 None。A 为 m×p，B 为 p×n（调用方已保证形状衔接）。
    """
    p = len(B)
    n = len(B[0])
    for i, row in enumerate(A):
        for j in range(n):
            v = 0
            for t in range(p):
                a = row[t]
                if a:
                    v += a * B[t][j]
            if v != 0:
                return i, j, v
    return None


def homology(boundaries, *, budget=_DEFAULT_BUDGET):
    """计算维数 ≤ 2 链复形的各阶整系数同调群。

    参数:
      boundaries: list，含 1 个或 2 个边缘矩阵 [∂₁] 或 [∂₁, ∂₂]。
                  每个矩阵为 list[list[int]]，行数、列数均 ≤ 8，
                  元素绝对值 ≤ 10000；∂₂ 的行数必须等于 ∂₁ 的列数，
                  且必须满足 ∂₁·∂₂ = 0。
      budget:     每个边缘矩阵各自的 Smith 消元步数上限（正 int，
                  拒绝 bool）；耗尽抛 BudgetExhaustedError——只表示
                  搜索未完成，不代表任何同调结论。

    返回 HomologyResult：
      groups[k] 给出 H_k ≅ Z^free_rank ⊕ Z/d_1 ⊕ ... ⊕ Z/d_s
      （torsion=(d_1,...,d_s)，d_i>1 且依次整除）。

    异常:
      MatrixShapeError:        boundaries 不是非空 list；
      MatrixLimitError:        边缘矩阵超过 2 个（维数上限 2），
                               或矩阵超过 8×8 / |元素| ≤ 10000；
      DimensionMismatchError:  相邻边缘矩阵形状不衔接；
      IntegerRequiredError / NonFiniteNumberError: 元素类型非法；
      NotAChainComplexError:   ∂_k·∂_{k+1} ≠ 0（含定位信息）。

    输入不被修改（内部只用副本），失败时不留下任何部分结果。
    """
    if isinstance(boundaries, bool) or not isinstance(boundaries, list):
        raise MatrixShapeError(
            f"boundaries: 必须是边缘矩阵的 list，收到 {type(boundaries).__name__}"
        )
    if len(boundaries) == 0:
        raise MatrixShapeError("boundaries: 至少需要 1 个边缘矩阵 ∂₁，收到空 list")
    if len(boundaries) > _MAX_DIMENSION:
        raise MatrixLimitError(
            f"boundaries: 链复形维数 {len(boundaries)} 超过上限 {_MAX_DIMENSION}"
        )

    shapes = []
    for idx, B in enumerate(boundaries):
        shapes.append(validate_matrix(
            B, name=f"boundaries[{idx}]",
            max_rows=_MAX_ROWS, max_cols=_MAX_COLS, max_abs=_MAX_ABS,
        ))

    # 相邻形状衔接：∂_{k+1} 的行数 == ∂_k 的列数
    for k in range(len(boundaries) - 1):
        _, cols_k = shapes[k]
        rows_next, _ = shapes[k + 1]
        if rows_next != cols_k:
            raise DimensionMismatchError(
                f"boundaries[{k + 1}] 的行数 {rows_next} 与 "
                f"boundaries[{k}] 的列数 {cols_k} 不衔接"
            )

    # 链复形公理：∂_k · ∂_{k+1} = 0（精确整数核验，先验证再计算）
    for k in range(len(boundaries) - 1):
        bad = _first_nonzero_product_entry(boundaries[k], boundaries[k + 1])
        if bad is not None:
            i, j, v = bad
            raise NotAChainComplexError(
                f"boundaries[{k}]*boundaries[{k + 1}] 在位置 ({i},{j}) "
                f"的值为 {v} ≠ 0：相邻边缘算子乘积必须为零矩阵"
            )

    certs = tuple(smith(B, budget=budget) for B in boundaries)
    ranks = tuple(c.rank for c in certs)
    dim = len(boundaries)

    groups = []
    for k in range(dim + 1):
        # dim C_k：k=0 时为 ∂₁ 的行数，否则为 ∂_k 的列数
        n_k = shapes[0][0] if k == 0 else shapes[k - 1][1]
        r_prev = ranks[k - 1] if k >= 1 else 0
        r_next = ranks[k] if k < dim else 0
        free = n_k - r_prev - r_next
        if free < 0:
            raise AssertionError("内部错误：自由秩为负（秩关系被破坏）")
        torsion = ()
        if k < dim:
            torsion = tuple(d for d in certs[k].invariants if d > 1)
        groups.append(HomologyGroup(degree=k, free_rank=free, torsion=torsion))

    return HomologyResult(
        dimension=dim,
        groups=tuple(groups),
        boundary_ranks=ranks,
        certificates=certs,
    )
