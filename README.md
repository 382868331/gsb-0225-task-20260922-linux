# smithcert — 整数矩阵 Smith 标准形证书与整数方程组求解

离线、纯标准库的可复用题库：对整数矩阵 A 计算带**初等变换证书**的
Smith 标准形（SNF），并据此判定整数线性方程组 A x = b 的可解性、
给出一个整数特解与整数零空间的一组 Z-基；进一步支持
**等式 + 模同余混合系统** `solve_mixed`（A x = b 且每行
C_i x ≡ d_i (mod m_i)）。全程精确整数运算，
**没有浮点、没有“先求有理解再取整”**。

## 运行环境

- Python **3.14.7**（仅使用标准库：`dataclasses`、`fractions`、`math`、
  `itertools`、`random`，其中 `fractions`/`random` 只出现在测试与 demo 的
  独立核验/数据生成中，核心算法不依赖）。
- Linux 原生运行，装好 Python 后**无需安装任何其他依赖**，无需网络、
  账号、密钥、数据库、中间件、云 API 或 Docker。

## 准确命令

```bash
# 必要测试（41 个测试用例，含数百次确定性随机子用例与小盒枚举对照）
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

### `solve_mixed(A, b, C, d, m, *, budget=100000) -> MixedSolveResult`

求解混合系统：整数向量 x（长度 n）同时满足
`A x = b` 与每行同余 `C_i·x ≡ d_i (mod m_i)`。

- `A`、`C` 允许为 `[]`（分别表示“无普通等式”“无同余条件”），
  但不能同时为空（无法确定 n）；两者均非空时列数必须一致。
- `m` 为长度等于 C 行数的正整数列表（拒绝 bool/float/0/负数，
  上限 10000）；**`m_i = 1` 视为无约束**，不引入辅助变量。
- 方法：每条 `m_i > 1` 的同余引入一个独立整数辅助变量 t_i，写成
  `C_i·x − m_i·t_i = d_i`，与原等式拼成增广系统后直接调用现有
  整数求解内核 `solve_integer`，**不另搭替代实现**。
- **增广系统行数（等式行 + 保留同余数）与列数（原变量数 + 保留
  同余数）均不得超过 8**，超限抛 `MatrixLimitError`。
- 返回 `MixedSolveResult`：
  - 可解时：`particular` 为**原变量空间**的整数特解（长度 n），
    `basis` 为原变量解差格的一组 Z-基；全部解为
    `x = particular + Σ z_j·basis_j`（z_j ∈ ℤ）。
  - 投影不“简单删掉辅助坐标”：增广零空间基的前 n 个坐标只是
    解差格的**生成集**（投影满射 ⇒ 不遗漏，但可能冗余），库内对其
    再做一次 Smith 整数规范化（以生成向量为行的矩阵 M，U₂·M 的
    非零行即行格的 Z-基），保证返回的基线性无关且生成整个投影格。
  - 不可解时：`obstruction` 为增广系统上可独立按算术核验的整数
    障碍（`w·A_aug` 各分量被 `modulus` 整除而 `w·b_aug` 余非零，
    或零行障碍），偶奇冲突、非互素模数冲突都归约为这类整除证据。
  - 证书字段保留**增广变换链**：`A_aug, b_aug, U, D, V, c,
    transforms`（可 `replay(A_aug, transforms)` 独立重放），以及
    投影规范化的 `norm_transforms`；`n_vars/n_aux/active` 记录
    原变量数、实际引入的辅助变量数与被保留的同余下标。

## 输入与错误语义（先固定的契约）

- 元素必须是**精确 `int`**：`bool`（即使是 int 子类）一律拒绝；
  `float` 一律拒绝（包括 `2.0`，不做隐式取整）；`NaN`、`±Infinity`
  抛 `NonFiniteNumberError`，其他非 int 抛 `IntegerRequiredError`。
- 形状错误（非 list、空矩阵、行长度不一）抛 `MatrixShapeError`；
  `b` 长度不符抛 `DimensionMismatchError`；超上限抛
  `MatrixLimitError`。错误信息含可定位下标，如
  `A[1][1]: 元素 10001 的绝对值超过上限 10000`。
- 非法输入立即抛错，**不静默纠正**；函数不修改用户传入的对象
  （内部使用副本，有测试守护）。
- `budget` 必须为正 int（同样拒绝 bool）。预算耗尽抛
  `BudgetExhaustedError`——它只表示**搜索未完成**，与“证明无解”
  严格分开：无解时 `solve_integer` 正常返回 `consistent=False` 与
  余数/零行证据，绝不使用该异常。公开上限内实测最坏用例约 3.3 千步，
  默认预算 10 万步。
- 接口无状态：一次调用要么完整返回，要么抛异常，不存在部分结果。

## 误差口径

本库**不产生任何浮点结果**：Smith 消元、可解判定、特解与零空间基
全部是 Python 任意精度整数精确运算，所有恒等式（UAV=D、A x₀=b、
Az=0、整除余数、det=±1）均为精确相等/精确整除，没有容差，
也不存在“用宽容差掩盖边界错误”。测试与 demo 中唯一的分数运算是
独立核验器用的 `fractions.Fraction`（行列式、有理秩），它本身也是
精确的。

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
  完备性用“极大子式 gcd=1”的饱和格判据核验。`UAV=D` 往返一致只作
  补充性核验。

## 已知限制

- 仅实现本题契约的有限子集：m,n ≤ 8、|元素| ≤ 10000 的整数矩阵；
  不宣称兼容任何行业标准格式或更大规模输入（超限显式报错）。
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
  mixed.py            等式+模同余混合求解（增广→内核→投影规范化）
tests/
  _oracle.py          独立参考（Fraction 消元、枚举子式行列式因子）
  test_smith.py       Smith 证书测试
  test_solve.py       整数求解测试
  test_mixed.py       混合系统测试（手算用例 + 小盒独立枚举对照）
demo.py               固定输入演示
README.md
```
