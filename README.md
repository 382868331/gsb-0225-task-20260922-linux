# smithcert — 整数矩阵 Smith 标准形证书与整数方程组求解

离线、纯标准库的可复用题库：对整数矩阵 A 计算带**初等变换证书**的
Smith 标准形（SNF），并据此判定整数线性方程组 A x = b 的可解性、
给出一个整数特解与整数零空间的一组 Z-基；并在最多 6 个变量的有限
逐变量整数上下界内，求可行解中 L1 范数最小、再字典序最小的一组。
全程精确整数运算，
**没有浮点、没有“先求有理解再取整”**。

## 运行环境与 venv

- Python **3.14.7**（仅使用标准库：`dataclasses`、`fractions`、`math`、
  `itertools`、`random`，其中 `fractions`/`random` 只出现在测试与 demo 的
  独立核验/数据生成中，核心算法不依赖）。
- Linux 原生运行，装好 Python 后**无需安装任何其他依赖**，无需网络、
 账号、密钥、数据库、中间件、云 API 或 Docker。

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python --version   # 应显示 Python 3.14.7
# requirements.txt 为空；标准库即可，仍可执行：
python -m pip install --upgrade pip
```

## 准确命令

```bash
# 必要测试（含 Smith、无界整数求解、有界最优求解；确定性随机子用例）
python -m unittest discover -s tests -v

# 原固定输入演示（无 sleep、无预录结果）
python demo.py

# 本题新增有界求解迭代/分支搜索演示（控制在 90 秒内；无 sleep）
python demo_iteration.py
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

### `solve_bounded_integer(A, b, bounds, *, budget=100000) -> BoundedSolveResult`

- 有界接口的 `A` 仍为整数矩阵，**变量数 n ≤ 6**、行数 m ≤ 8；
  `bounds` 为长度 n 的 `list[(lower, upper)]`，所有界必须是精确有限
  `int` 且 `lower <= upper`。
- 先调用未改动的 Smith/无界 Diophantine 路径得到 `particular` 与
  `nullspace_basis`；再在**原始变量 x 的有限盒**中搜索，不把特解简单
  截断到边界，也不限定零空间系数的任意“小盒”。
- 每个分支保存每变量有限整数域 `[l_i,u_i]`。对每条方程反复做区间投影：
  固定其他变量当前域后，推出变量 i 的可证明整数上下界；域变空即剪除。
- 区间不再收缩时，选当前最宽变量做中点二叉划分：`[l,m]` 与 `[m+1,u]`。
  两个子域不交且完整覆盖父域；因为所有界有限，递归深度与宽度有限。
- 每个节点用精确 L1 下界剪枝；L1 下界等于当前最好值时，再用域的逐分量
  字典序下界剪除不可能改进的节点。完整结束才返回 `optimal` 或
  `infeasible`；预算耗尽严格保留未完成语义。
- 返回状态常量：
  - `"optimal"`：`solution` 证明为 L1 最小、再字典序最小；
  - `"feasible"`：已找到当前可行解，但预算耗尽、未证明最优；
  - `"infeasible"`：已证明给定有限界内无整数解；
  - `"unsupported"`：`bounds=None`、某项为 `None`、或某个界缺失，
    表示无界输入；本题明确不支持，不能解释为不可行；
  - `"budget_exhausted"`：预算耗尽且当前没有可行解，不表示无解。
- 可行结果的证书字段包含：`solution`、`l1_norm`、`residual=A x`、
  `bounds_check`（逐变量布尔值）以及原 `particular`/`nullspace_basis`；
  无界方程组本身不可解时仍回传 Smith `obstruction`。
- `budget` 度量有界区间收缩与分支节点；Smith 构造使用自身公开默认预算，
  避免把用于证明最优性的搜索预算误耗在证书构造上。

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
- 有界接口中，界为 `None` 或缺失不是非法异常，而是正常返回
  `status="unsupported"`；界结构错误、长度不符、bool/float、下界大于
  上界仍立即抛错。有界搜索的预算耗尽**不抛异常**，而是返回
  `feasible` 或 `budget_exhausted`，从而显式携带“已有/尚无可行解”。
- 无界 Diophantine 接口无状态：一次调用要么完整返回，要么抛异常。
  有界搜索接口在预算耗尽时会返回带当前搜索状态的结果，以区分
  “可行未证最优”和“无当前可行解”。

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
- 有界最优搜索是精确 branch-and-bound：区间投影只使用线性不等式的可靠
  外包界，宁可保留更大的域，也不会删除可能可行点；二叉划分完备覆盖。
  测试用小有限盒的独立全枚举核对目标值与字典序，并逐项检查 `Ax=b` 与
  边界；另测无界 `unsupported`、界内不可行、可行未证最优和预算耗尽但无
  当前可行解。
- 该方法不依赖 MILP/切割平面/浮点 LP；对至多 6 个变量、预算内的小整数
  盒可靠。大盒可能在预算内无法证最优，这会如实报告为未完成。
- 独立测试期望不来自被测核心：不变因子用枚举全部子式的行列式因子
  （Δ_k 的 gcd，s_k=Δ_k/Δ_{k−1}）独立求出；秩用 Fraction 高斯消元；
  整数可解性用 A 与增广矩阵行列式因子相等的独立判据；零空间基的
  完备性用“极大子式 gcd=1”的饱和格判据核验。`UAV=D` 往返一致只作
  补充性核验。

## 已知限制

- 仅实现本题契约的有限子集：无界求解 m,n ≤ 8、|元素| ≤ 10000；
  有界最优接口 n ≤ 6；不宣称兼容行业标准格式或更大规模输入（超限显式
  报错）。
- 有界优化只接受每变量有限整数上下界；无界目标/无界变量返回
  `unsupported`，不尝试通用无界整数优化。
- 区间收缩与分支搜索受 `budget` 限制；预算耗尽时结果可能是可行但未证
  最优，或尚无当前可行解，不能据此声称不可行。
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
  bounded.py          有限逐变量界内 L1/字典序最优分支搜索
tests/
  _oracle.py          独立参考（Fraction 消元、枚举子式行列式因子）
  test_smith.py       Smith 证书测试
  test_solve.py       整数求解测试
  test_bounded.py     有界最优、全枚举参考与状态/异常测试
demo.py               固定输入演示
demo_iteration.py     有界求解分支搜索演示
README.md
```
