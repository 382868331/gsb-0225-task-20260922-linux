# smithcert — 整数矩阵 Smith 标准形证书与整数方程组求解

离线、纯标准库的可复用题库：对整数矩阵 A 计算带**初等变换证书**的
Smith 标准形（SNF），并据此判定整数线性方程组 A x = b 的可解性、
给出一个整数特解与整数零空间的一组 Z-基。全程精确整数运算，
**没有浮点、没有“先求有理解再取整”**。

## 运行环境

- Python **3.14.7**（仅使用标准库：`dataclasses`、`fractions`、`math`、
  `itertools`、`random`。新增有界投影用 `fractions.Fraction` 做精确
  有理边界计算；全程不使用浮点）。
- Linux 原生运行，装好 Python 后**无需安装任何其他依赖**，无需网络、
  账号、密钥、数据库、中间件、云 API 或 Docker。

## venv 与准确命令

```bash
# 在仓库根目录创建并启用 Linux 原生 Python 3.14.7 虚拟环境
python -m venv .venv
source .venv/bin/activate

# 无第三方依赖；如需要，可执行空依赖安装命令确认环境
python -m pip install -r requirements.txt

# 全部测试（原 Smith/整数解测试与新增有界优化测试）
python -m unittest discover -s tests -v

# 本轮新增固定输入演示（现场计算，无 sleep、无预录结果，控制在 90 秒内）
python demo_iteration.py

# 旧演示仍可运行
python demo.py
```

命令在仓库根目录（本文件所在目录）执行。

## 本轮新增：有限盒内 L1 最优整数解

### `solve_bounded_integer(A, b, lower, upper, *, budget=100000) -> BoundedSolveResult`

- `A` 为 m×n 整数矩阵，本题有界接口限定 **n ≤ 6**；`b` 长度为 m；
  `lower`/`upper` 是长度 n 的逐变量整数界。
- 求 `A x=b` 且 `lower_i≤x_i≤upper_i` 的整数解中，`Σ|x_i|` 最小的一组；
  L1 并列时再按整数向量字典序最小选择。
- 返回状态：
  - `optimal`：已证明 L1 与字典序均最优；
  - `feasible`：找到可行解但预算耗尽，尚未证明最优；
  - `infeasible`：已证明该有限界内无整数可行解；
  - `budget_exhausted`：预算耗尽且尚无当前可行解，不能冒充不可行；
  - `unsupported`：存在 `None` 或 `±float("inf")` 的无界方向。
- 可行结果含 `solution`、`objective`、`Ax`、`equation_check`、
  `bounds_check`、`particular`、`nullspace_basis`、`coefficients`，其中
  `solution = particular + Σ coefficient_j * nullspace_basis[j]`。
- 有限 `float`、`bool`、长度错误仍立即抛错；无界标记不抛错而返回
  `unsupported`，因为这是本题明确不支持的问题类型，不是非法算术结果。

### 为什么不是“截特解”或“随便限系数小盒”

有界求解先调用原 `solve_integer()` 得到精确整数特解 `x0` 和整数核 Z-基
`B`，全部候选写成 `x=x0+Bz, z∈Z^k`。随后：

1. 把原始上下界改写成关于 `z` 的 2n 个线性不等式，用
   `fractions.Fraction` 枚举有界多面体顶点，精确求每个 `z_j` 的实投影
   最小/最大值并取整数 ceil/floor，形成不漏解的初始整数系数盒。
2. 初始投影盒覆盖全部可行整数系数；分支后继续把每个子盒当作有限搜索域。
   不使用会误删解的“独立其他变量极值”强区间传播。
3. 对每个原变量的候选区间求 `|x_i|` 的独立下界，求和得到 L1 有效下界；
   下界超过当前最优值时剪枝。
4. L1 下界等于当前值时，再用逐变量区间最小向量做字典序剪枝。
5. 对最宽系数二分；搜索结束意味着每个整数系数要么完成单点检查，要么
   被有效下界剪掉。初始盒有限且覆盖全部可行解，所以这是完整穷尽搜索，
   可给出最优或确定无解结论。

原 `solve_integer()` 的无界 Diophantine 求解、特解、零空间基与障碍证书
继续保持可用。新增 `tests/test_bounded.py` 用 `itertools.product` 独立
全枚举小盒，核对 L1、字典序、`Ax=b`、边界与通解系数，并覆盖无解、无界、
两种预算耗尽状态和异常参数。

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

## 输入与错误语义（先固定的契约）

- `A`/`b` 元素必须是**精确 `int`**：`bool`（即使是 int 子类）一律拒绝；
  有限 `float` 一律拒绝（包括 `2.0`，不做隐式取整）；`NaN`、`±Infinity`
  抛 `NonFiniteNumberError`，其他非 int 抛 `IntegerRequiredError`。
  `solve_bounded_integer()` 的上下界中，`None`/`±Infinity` 表示无界并
  返回 `unsupported`；有限 `float`、`bool`、`NaN` 仍立即抛错。
- 形状错误（非 list、空矩阵、行长度不一）抛 `MatrixShapeError`；
  `b` 长度不符抛 `DimensionMismatchError`；超上限抛
  `MatrixLimitError`。错误信息含可定位下标，如
  `A[1][1]: 元素 10001 的绝对值超过上限 10000`。
- 非法输入立即抛错，**不静默纠正**；函数不修改用户传入的对象
  （内部使用副本，有测试守护）。
- `budget` 必须为正 int（同样拒绝 bool）。预算耗尽抛
  `BudgetExhaustedError`——它只表示**搜索未完成**，与“证明无解”
  严格分开：无解时 `solve_integer` 正常返回 `consistent=False` 与
  余数/零行证据，绝不使用该异常。`solve_bounded_integer` 则把新增
  分支搜索预算耗尽编码为 `feasible` 或 `budget_exhausted` 状态，不把
  它解释成 infeasible。公开上限内 Smith 最坏用例约 3.3 千步，默认
  预算 10 万步。
- 接口无状态：一次调用要么完整返回，要么抛异常，不存在部分结果。

## 误差口径

本库**不产生任何浮点结果**：Smith 消元、可解判定、特解与零空间基
使用 Python 任意精度整数；有界投影使用 `fractions.Fraction` 做精确
有理边界。所有恒等式（UAV=D、A x₀=b、Az=0、整除余数、det=±1、
Ax=b、上下界检查）均为精确相等/精确整除，没有容差，也不存在
“用宽容差掩盖边界错误”。

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

- Smith/无界整数解接口支持 m,n ≤ 8、|元素| ≤ 10000；新增有限盒优化
  接口限定变量数 n ≤ 6。超限显式报错，不宣称更大规模输入的性能。
- 有界优化只接受逐变量有限整数上下界；任一方向无界时返回
  `unsupported`，不做无界 L1 优化。
- 搜索预算按访问分支节点计；预算耗尽时严格表示“未证明”，只有完整
  穷尽才返回 `optimal` 或 `infeasible`。
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
  solve.py            整数解、零空间基、障碍证据、有限盒 L1 分支求解
tests/
  _oracle.py          独立参考（Fraction 消元、枚举子式行列式因子）
  test_smith.py       Smith 证书测试
  test_solve.py       整数求解测试
  test_bounded.py     小盒全枚举、预算、无界与异常路径测试
demo.py               原固定输入演示
demo_iteration.py     本轮有限盒 L1 求解演示
README.md
```
