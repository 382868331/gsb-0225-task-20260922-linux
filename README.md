# smithcert — 整数矩阵 Smith 标准形证书、整数方程组求解与低维链复形同调

离线、纯标准库的可复用题库：对整数矩阵 A 计算带**初等变换证书**的
Smith 标准形（SNF），并据此判定整数线性方程组 A x = b 的可解性、
给出一个整数特解与整数零空间的一组 Z-基；在 SNF 之上进一步对
**维数 ≤ 2 的链复形**（先验证相邻边缘矩阵乘积为零）输出各阶同调的
**自由秩（Betti 数）与挠系数**。全程精确整数运算，
**没有浮点、没有“先求有理解再取整”**。

## 运行环境

- Python **3.14.7**（仅使用标准库：`dataclasses`、`fractions`、`math`、
  `itertools`、`random`，其中 `fractions`/`random` 只出现在测试与 demo 的
  独立核验/数据生成中，核心算法不依赖）。
- Linux 原生运行，装好 Python 后**无需安装任何其他依赖**，无需网络、
  账号、密钥、数据库、中间件、云 API 或 Docker。

## 准确命令

```bash
# 必要测试（36 个测试用例，含数百次确定性随机子用例）
python -m unittest discover -s tests -v

# 固定输入演示（实测约 0.04 秒，远小于 90 秒；无 sleep、无预录结果）
python demo.py
```

在仓库根目录（本文件所在目录）执行。

## 公开接口

### `smith(A, *, budget=100000) -> SmithResult`

- `A`：`list[list[int]]`，行数、列数均 **≤ 8**，每个元素
  **绝对值 ≤ 10000**。
- 返回不可变 `SmithResult`：
  - `U`（m×m）、`V`（n×n）：整数幺模阵；
  - `D`（m×n）：非负对角阵，非零对角元为正、依次满足
    `d_i | d_{i+1}`，零行/零列在最后；
  - 恒等式 **`U * A * V = D`**；
  - `invariants`：非零不变因子 `[d_1,...,d_r]`；`rank`：秩；
  - `transforms`：全变换日志，元组 of `Transform(side, kind, i, j, k)`。
- `Transform` 只允许三类整数幺模初等操作（作用于工作矩阵）：
  - `("row"|"column", "swap", i, j)`：交换两行/两列；
  - `("row"|"column", "negate", i)`：某行/列乘 −1；
  - `("row"|"column", "add", i, j, k)`：第 j 行/列的 k 倍加到
    第 i 行/列（k 为任意整数）。
- `replay(A, transforms)`：从单位阵出发**独立重放**日志得到
  `(U, D, V)`，供外部复核；库内部的 U/V 也只由重放产生。

### `solve_integer(A, b, *, budget=100000) -> IntegerSolveResult`

- `b`：长度 m 的 `list[int]`，元素绝对值 ≤ 10000。
- 返回：
  - `consistent: bool`；
  - 可解时：`particular`（一个整数特解 x₀）、`nullspace_basis`
    （整数零空间的一组 Z-基，每个向量长度 n，共 n−rank 个），
    通解 `x = x₀ + Σ zⱼ·basisⱼ`（zⱼ ∈ ℤ）；
  - 无解时：`obstruction` 为可独立按算术核验的整数证据：
    - `kind="divisibility"`：`w·A` 各分量均被 `modulus` 整除，
      而 `w·b` 的非零 `remainder` 不能被整除（“有理解但无整数解”
      由这类证据刻画）；
    - `kind="zero_row"`：`w·A = 0` 但 `w·b ≠ 0`；
    - `w` 就是证书矩阵 U 的某一行，`wA`/`wb` 已预算好便于复核。
  - 同时回传所用 Smith 证书（`U,D,V,invariants,rank,c,transforms`）。
- 求解路径：由 `U A V = D`，令 `c = U b`，在对角形上逐行检查
  `d_i | c_i` 及零行 `c_i = 0`；特解由 `y_i = c_i/d_i`、`x = V y`
  直接整数构造；零空间基取 V 的第 rank…n−1 列（V 幺模，整数双射，
  故这是整数核本身的 Z-基，不存在有理基放大/取整）。

### `homology(boundaries, *, budget=100000) -> HomologyResult`

计算**维数 ≤ 2 的链复形**的各阶整系数同调群（本次迭代新增）。

- `boundaries`：边缘矩阵的 list，含 1 个（`[∂₁]`）或 2 个
  （`[∂₁, ∂₂]`）矩阵。约定 `boundaries[k]` 是
  `∂_{k+1}: C_{k+1} → C_k`，形状 `(dim C_k) × (dim C_{k+1})`，
  第 j 列是第 j 个 (k+1) 维生成元边界的坐标。每个矩阵沿用
  `smith()` 的输入上限（≤ 8×8、|元素| ≤ 10000）。
- 计算前**先精确验证链复形公理**：
  - 相邻形状衔接：`∂₂` 的行数 == `∂₁` 的列数，否则
    `DimensionMismatchError`；
  - 相邻乘积为零：`∂₁·∂₂` 必须是精确零矩阵，否则抛
    `NotAChainComplexError`，信息含乘积中第一个非零元的位置与数值，
    如 `boundaries[0]*boundaries[1] 在位置 (1,0) 的值为 3 ≠ 0`。
- 返回不可变 `HomologyResult`：
  - `dimension`：链复形维数（边缘矩阵个数，1 或 2）；
  - `groups`：各阶 `HomologyGroup(degree, free_rank, torsion)`，
    长度 `dimension+1`。`H_k ≅ Z^free_rank ⊕ Z/d_1 ⊕ ... ⊕ Z/d_s`，
    `torsion=(d_1,...,d_s)` 只保留 `d_i > 1` 的挠系数
    （Z/1 是平凡群），按整除链顺序；
  - `boundary_ranks`：各边缘矩阵的秩；
  - `certificates`：各边缘矩阵的完整 `SmithResult` 证书，
    供外部复核 `U*∂*V = D`。
- 数学依据：设 `r_k = rank(∂_k)`（`∂_0`、`∂_{dim+1}` 视为零映射），
  则自由秩 `b_k = dim C_k − r_k − r_{k+1}`；挠系数是 `∂_{k+1}` 的
  Smith 不变因子中大于 1 的项。全部复用本库的 Smith 证书，
  无浮点、无有理取整。
- `budget`：传给每个边缘矩阵各自的 `smith()` 调用，语义不变。
- 错误语义：`boundaries` 非 list 或为空抛 `MatrixShapeError`；
  超过 2 个边缘矩阵抛 `MatrixLimitError`（维数上限 2，常量
  `MAX_CHAIN_DIMENSION`）；矩阵本身的形状/类型/上限错误与
  `smith()` 完全一致。验证失败即抛错，不返回任何部分结果，
  不修改用户输入。

## 输入与错误语义（先固定的契约）

- 元素必须是**精确 `int`**：`bool`（即使是 int 子类）一律拒绝；
  `float` 一律拒绝（包括 `2.0`，不做隐式取整）；`NaN`、`±Infinity`
  抛 `NonFiniteNumberError`，其他非 int 抛 `IntegerRequiredError`。
- 形状错误（非 list、空矩阵、行长度不一）抛 `MatrixShapeError`；
  `b` 长度不符抛 `DimensionMismatchError`；超上限抛
  `MatrixLimitError`。错误信息含可定位下标，如
  `A[1][1]: 元素 10001 的绝对值超过上限 10000`。
- 链复形输入的额外错误：相邻边缘矩阵乘积非零抛
  `NotAChainComplexError`（信息含乘积中第一个非零元的位置与数值）；
  边缘矩阵超过 2 个抛 `MatrixLimitError`（维数上限 2）。
- 非法输入立即抛错，**不静默纠正**；函数不修改用户传入的对象
  （内部使用副本，有测试守护）。
- `budget` 必须为正 int（同样拒绝 bool）。预算耗尽抛
  `BudgetExhaustedError`——它只表示**搜索未完成**，与“证明无解”
  严格分开：无解时 `solve_integer` 正常返回 `consistent=False` 与
  余数/零行证据，绝不使用该异常。公开上限内实测最坏用例约 3.3 千步，
  默认预算 10 万步。
- 接口无状态：一次调用要么完整返回，要么抛异常，不存在部分结果。

## 误差口径

本库**不产生任何浮点结果**：Smith 消元、可解判定、特解、零空间基、
同调自由秩与挠系数全部是 Python 任意精度整数精确运算，所有恒等式
（UAV=D、A x₀=b、Az=0、∂₁∂₂=0、整除余数、det=±1）均为精确相等/
精确整除，没有容差，也不存在“用宽容差掩盖边界错误”。测试与 demo
中唯一的分数运算是独立核验器用的 `fractions.Fraction`（行列式、
有理秩），它本身也是精确的。

## 算法与设计取舍

- Smith：整数 pivot-chasing 消元。每轮用带余除法把不能被当前枢轴
  整除的行/列元素的余数换到枢轴位，枢轴绝对值严格下降，必终止；
  行、列清零后还扫描整个剩余子块，若有元素不被枢轴整除则把该列
  加回枢轴列重新消元——这一步保证 `d_k` 整除剩余子块，归纳得到
  标准整除链，无需事后“修补”对角元。
- U/V 不在消元中增量维护，而是结束后从单位阵按日志**重放**得到，
  使“证书可独立复核”成为构造路径本身；测试另有一份独立实现的
  重放器交叉比对。
- 取值上限是接口边界而非演示规模：demo 用小型固定矩阵，测试含
  8×8、|元素|=10000 的边界用例与数百次随机用例，demo 不穷举上限。
- 独立测试期望不来自被测核心：不变因子用枚举全部子式的行列式因子
  （Δ_k 的 gcd，s_k=Δ_k/Δ_{k−1}）独立求出；秩用 Fraction 高斯消元；
  整数可解性用 A 与增广矩阵行列式因子相等的独立判据；零空间基的
  完备性用“极大子式 gcd=1”的饱和格判据核验；同调的自由秩用
  `b_k = dim C_k − rank_Q(∂_k) − rank_Q(∂_{k+1})`（Fraction 秩）
  独立求出，挠系数用 ∂_{k+1} 的行列式因子独立求出。`UAV=D` 往返
  一致只作补充性核验。
- 同调（本次迭代新增）：先精确验证 `∂₁·∂₂ = 0` 再计算；每个边缘
  矩阵的 Smith 证书原样回传，外部可独立复核 `U*∂*V = D`。测试用
  手算例子（圆、填充三角形、Z/2 挠、RP² 胞腔链、挠链 Z/2⊕Z/4）
  与分块构造的确定性随机链复形（`∂₁=[P|0], ∂₂=[0;S]` 保证
  乘积为零）双重核对。

## 已知限制

- 仅实现本题契约的有限子集：m,n ≤ 8、|元素| ≤ 10000 的整数矩阵；
  不宣称兼容任何行业标准格式或更大规模输入（超限显式报错）。
- 同调仅限**维数 ≤ 2 的链复形**（1 或 2 个边缘矩阵，对应
  H₀、H₁、H₂）；只输出各阶自由秩与挠系数，**不提供**代表链/
  生成元在任意链坐标下的表达，也不计算链映射的诱导同态。
- 不是通用代数系统：不处理多项式环、一般 PID 上的矩阵、稀疏格式、
  流式/分批输入；不提供网站、CLI 产品或其他扩展。
- 中间整数可能变大（Smith 理论上的正常现象），8×8 上限下无压力；
  Python 大整数保证精确，但本库不承诺上限外的性能。

## 文件结构

```
smithcert/            题库（可 import）
  __init__.py         公开 API 导出与常量
  errors.py           错误类型
  validate.py         输入校验（形状/整数性/有限性/上限）
  transform.py        Transform 日志与独立重放
  smith.py            Smith 标准形核心
  solve.py            Ax=b 整数可解判定、特解、零空间基、障碍证据
  homology.py         链复形公理验证与各阶同调（自由秩、挠系数）
tests/
  _oracle.py          独立参考（Fraction 消元、枚举子式行列式因子、
                      独立同调期望值）
  test_smith.py       Smith 证书测试
  test_solve.py       整数求解测试
  test_homology.py    链复形同调测试（手算例 + 随机复形 + 错误语义）
demo.py               固定输入演示
README.md
```
