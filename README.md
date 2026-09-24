# smithcert — 整数矩阵 Smith 标准形证书、整数方程组求解与自由链复形整同调

离线、纯标准库的可复用题库：对整数矩阵 A 计算带**初等变换证书**的
Smith 标准形（SNF），并据此判定整数线性方程组 A x = b 的可解性、
给出一个整数特解与整数零空间的一组 Z-基；此外接受**维度 ≤ 2** 的
有限自由链复形边界矩阵序列，先验证相邻边界复合为零，再利用既有
Smith 证书输出各阶**自由秩（Betti 数）与挠系数**。全程精确整数运算，
**没有浮点、没有“先求有理解再取整”**。

## 运行环境

- Python **3.14.7**（仅使用标准库：`dataclasses`、`fractions`、`math`、
  `itertools`、`random`，其中 `fractions`/`random` 只出现在测试与 demo 的
  独立核验/数据生成中，核心算法不依赖）。
- Linux 原生运行，装好 Python 后**无需安装任何其他依赖**，无需网络、
  账号、密钥、数据库、中间件、云 API 或 Docker。

## 准确命令

```bash
# 必要测试（48 个测试用例，含数百次确定性随机子用例）
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

- `boundaries`：边界矩阵序列 `[d_1, ..., d_D]`，`d_p` 为
  m_{p−1}×m_p 的 `list[list[int]]`，表示
  `C_D --d_D--> C_{D-1} --> ... --d_1--> C_0`。
  **`D` 只允许 1 或 2**；每个矩阵沿用 `smith` 的上限（行、列 ≤ 8，
  元素绝对值 ≤ 10000）。
- 先做输入校验（形状、精确 int、上限），再检查相邻矩阵**形状衔接**
  （`d_p` 列数 = `d_{p+1}` 行数），最后用精确整数乘法逐对验证
  **`d_p d_{p+1} = 0`**；任一非零复合抛 `BoundaryProductError`，
  信息含复合阶数与第一个非零元的 `[行][列]` 和值。全部校验通过后
  才开始计算，**不留半更新状态**，且不修改用户传入的矩阵。
- 返回不可变 `HomologyResult`：
  - `max_degree`：D；`groups`：次数 0..D 的 `HomologyGroup`；
  - `HomologyGroup.degree / free_rank / torsion`：自由秩（Betti 数）
    与挠系数元组（均 >1，保持 Smith 整除链顺序）；
    `is_trivial` 判平凡群，`describe()` 给 `"Z^2 ⊕ Z/2"`、`"0"` 记号；
  - `betti_numbers`、`torsion_coefficients`：按次数排列的便捷元组；
  - `ranks`：各边界矩阵的秩（末尾补一个 `r_{D+1}=0` 占位）；
  - `invariants`：各边界矩阵的非零 Smith 不变因子；
  - `certificates`：每个 `d_p` 的完整 `SmithResult`（U/D/V/日志），
    可直接用既有 `replay` 等手段独立复核。
- 计算公式（由 Smith 标准形直接读出，r_p = rank d_p，约定
  r_0 = r_{D+1} = 0）：
  - **自由秩（Betti 数）**：`b_p = m_p − r_p − r_{p+1}`；
  - **挠系数**：`H_p` 的挠 = `d_{p+1}` 的 Smith 不变因子中 **>1** 者
    （`H_D` 无挠系数来源）。即
    `H_p ≅ Z^{b_p} ⊕ ⨁ Z/s`。
- 维度 1 例子：圆的边/点边界 `d1`（3×3，秩 2，不变因子全 1）→
  `H_0 = Z, H_1 = Z`。
- 维度 2 例子：填充三角形追加面边界 `d2`（3×1，像张成 ker d1）→
  `H_0 = Z, H_1 = H_2 = 0`（可缩）；
  代数例 `d1=[0], d2=[2]` → `H_1 = Z/2`。

## 输入与错误语义（先固定的契约）

- 元素必须是**精确 `int`**：`bool`（即使是 int 子类）一律拒绝；
  `float` 一律拒绝（包括 `2.0`，不做隐式取整）；`NaN`、`±Infinity`
  抛 `NonFiniteNumberError`，其他非 int 抛 `IntegerRequiredError`。
- 形状错误（非 list、空矩阵、行长度不一）抛 `MatrixShapeError`；
  `b` 长度不符或相邻边界矩阵不衔接抛 `DimensionMismatchError`；
  超上限（矩阵 8×8 / 元素 10000 / 链复形维度 2）抛
  `MatrixLimitError`。错误信息含可定位下标，如
  `A[1][1]: 元素 10001 的绝对值超过上限 10000`，边界矩阵按
  `d_p[行][列]` 定位（如 `d_2[0][0]`）。
- 链复形相邻边界复合非零抛 `BoundaryProductError`
  （`ChainComplexError` 子类），信息形如
  `d_1 * d_2 != 0：乘积在 [0][0] 处为 2`；校验全部通过后才计算，
  一次调用要么完整返回要么抛异常，没有部分结果。
- 非法输入立即抛错，**不静默纠正**；函数不修改用户传入的对象
  （内部使用副本，有测试守护）。
- `budget` 必须为正 int（同样拒绝 bool）。预算耗尽抛
  `BudgetExhaustedError`——它只表示**搜索未完成**，与“证明无解”
  严格分开：无解时 `solve_integer` 正常返回 `consistent=False` 与
  余数/零行证据，绝不使用该异常；`homology` 预算耗尽时同样只表示
  搜索未完成，不代表任何同调结论。公开上限内实测最坏用例约 3.3
  千步，默认预算 10 万步。
- 接口无状态：一次调用要么完整返回，要么抛异常，不存在部分结果。

## 误差口径

本库**不产生任何浮点结果**：Smith 消元、可解判定、特解与零空间基、
同调自由秩与挠系数全部是 Python 任意精度整数精确运算，所有恒等式
（UAV=D、A x₀=b、Az=0、整除余数、det=±1、d_p d_{p+1}=0）均为精确
相等/精确整除，没有容差，也不存在“用宽容差掩盖边界错误”。测试与
demo 中唯一的分数运算是独立核验器用的 `fractions.Fraction`
（行列式、有理秩），它本身也是精确的。

## 算法与设计取舍

- Smith：整数 pivot-chasing 消元。每轮用带余除法把不能被当前枢轴
  整除的行/列元素的余数换到枢轴位，枢轴绝对值严格下降，必终止；
  行、列清零后还扫描整个剩余子块，若有元素不被枢轴整除则把该列
  加回枢轴列重新消元——这一步保证 `d_k` 整除剩余子块，归纳得到
  标准整除链，无需事后“修补”对角元。
- U/V 不在消元中增量维护，而是结束后从单位阵按日志**重放**得到，
  使“证书可独立复核”成为构造路径本身；测试另有一份独立实现的
  重放器交叉比对。
- 同调不重写消元：`homology` 对每个边界矩阵调用现成 `smith()`，
  由 `U d_p V = D` 直接读出秩与不变因子。`d_p` 的像在 Smith 基下
  是前 r_p 个坐标方向且逐坐标为 s_{p,i} 倍；商掉该像后，ker d_p
  （m_p−r_p 维自由群）再模 im d_{p+1} 得到
  `Z^{m_p−r_p−r_{p+1}}` 加上 d_{p+1} 的非 1 不变因子对应的
  `Z/s` 挠群——这是结构定理的直接推论，不是枚举或启发式。
  `homology` 只**读取** Smith 结果，链复形校验在调用 Smith 之前
  全部完成，因此非法输入不可能产生“半个结果”。
- 取值上限是接口边界而非演示规模：demo 用小型固定矩阵，测试含
  8×8、|元素|=10000 的边界用例与数百次随机用例，demo 不穷举上限。
- 独立测试期望不来自被测核心：不变因子用枚举全部子式的行列式因子
  （Δ_k 的 gcd，s_k=Δ_k/Δ_{k−1}）独立求出；秩用 Fraction 高斯消元；
  整数可解性用 A 与增广矩阵行列式因子相等的独立判据；零空间基的
  完备性用“极大子式 gcd=1”的饱和格判据核验。同调的自由秩由独立
  Fraction 秩按 b_p=m_p−r_p−r_{p+1} 重算、挠系数由独立行列式因子
  给出，并以 Euler 特征 `Σ(−1)^p b_p = Σ(−1)^p dim C_p` 交叉核验；
  随机合法链复形由独立求出的整数核向量组合**构造**（先验保证
  d_p d_{p+1}=0，再当场用整数乘法断言）。`UAV=D` 往返一致只作
  补充性核验。

## 已知限制

- 仅实现本题契约的有限子集：m,n ≤ 8、|元素| ≤ 10000 的整数矩阵；
  链复形维度 D ≤ 2（即 `homology` 只接受 1–2 个边界矩阵）；
  不宣称兼容任何行业标准格式或更大规模输入（超限显式报错）。
- 不提供同调层面的代表链/任意同调类坐标/链映射诱导同态（本轮不
  在范围内）；`homology` 只输出各阶群结构（自由秩与挠系数）与
  各边界矩阵的 Smith 证书。
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
  homology.py         自由链复形整同调（自由秩、挠系数），复用 smith()
tests/
  _oracle.py          独立参考（Fraction 消元、枚举子式行列式因子）
  test_smith.py       Smith 证书测试
  test_solve.py       整数求解测试
  test_homology.py    链复形同调测试（手算例子 + 独立核验 + 错误用例）
demo.py               固定输入演示
README.md
```
