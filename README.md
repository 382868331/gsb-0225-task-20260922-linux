# smithcert —— 整数矩阵 Smith 标准形证书

离线、纯 Python 标准库的可复用题库：计算整数矩阵的 Smith 标准形并输出
**完整初等变换证书**（U、D、V 及全部操作日志），并基于证书求解整数线性
方程组 `A x = b`（可解判定、一个整数特解、整数零空间基）。

- Python：**3.14.7**（代码只使用长期稳定语法，3.10+ 亦可运行）
- 依赖：**无任何第三方依赖**，仅标准库
- 运行环境：Linux 原生；离线，不需要网络、账号、密钥、数据库或 Docker

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部必要测试
python demo.py                            # 固定输入演示（约 0.05 秒）
```

两个命令都在仓库根目录执行。

```python
from smithcert import smith, integer_solve, replay_certificate

cert = smith([[6, 4], [0, 10]])
cert.U, cert.D, cert.V                    # U*A*V == D（精确整数相等）
# U=[[-2,1],[5,-2]], D=[[2,0],[0,30]], V=[[0,1],[1,6]]
cert.operations                           # 完整变换日志（可重放）
cert.invariant_factors                    # (2, 30)：30 % 2 == 0

# 独立重放：仅按日志实例化初等矩阵并做精确乘法，不经过消元算法
U2, D2, V2 = replay_certificate(A, cert.operations)

sol = integer_solve([[6, 4], [0, 10]], [10, 10])
sol.solvable                # True
sol.particular_solution     # (1, 1)
sol.null_space_basis        # ()  满秩时为空

bad = integer_solve([[6, 4], [0, 10]], [2, 2])
bad.solvable                # False
bad.obstruction["residue"]  # 6：对角元 30 ∤ (Ub)[1]=6 的非零余数
```

## 接口契约

### `smith(A, *, budget=100000) -> SmithResult`

- 输入 `A`：整数矩阵（`list`/`tuple` 嵌套），`1 <= 行数, 列数 <= 8`，
  每个元素为 `int` 且 `|元素| <= 10000`。
- 输入矩阵**不会被修改**（内部使用副本）。
- 返回字段：

  | 字段 | 含义 |
  |---|---|
  | `U` | m×m 整数幺模矩阵，`det(U) = ±1` |
  | `D` | m×n Smith 标准形：非负对角，零排在最后；非零项 `d_i` 满足 `d_i | d_{i+1}` |
  | `V` | n×n 整数幺模矩阵，`det(V) = ±1` |
  | `operations` | 按执行顺序排列的完整初等操作日志（dataclass，见下） |
  | `rank` | 秩 |
  | `invariant_factors` | 正不变因子元组 `(d_1, …, d_r)` |
  | `transformations_used` | 实际使用的初等变换次数 |

- 恒等式 **`U * A * V == D`** 按精确整数相等成立。
- `U`、`V` 仅由三类初等操作构建：行/列交换（`RowSwap`/`ColSwap`，
  行列式 −1）、整行/列变号（`RowSign`/`ColSign`，行列式 −1）、
  把一行/列的**整数倍**加到另一行/列（`RowAdd`/`ColAdd`，行列式 +1）。
  这些类型可从 `smithcert` 导入，字段为 0 基下标。

### `integer_solve(A, b, *, budget=100000) -> IntegerSolveResult`

- `A` 契约同上；`b` 为长度等于 `A` 行数的整数向量（元素不设绝对值上限，
  Python 大整数精确运算）。
- 返回字段：`solvable`、`particular_solution`（`tuple[int,...]` 或 `None`）、
  `null_space_basis`（整数向量元组，满列秩时为 `()`）、`rank`、`nullity`、
  `obstruction`（不可解时的障碍字典）、`transformed_rhs`（即 `U b`）、
  `invariant_factors`。
- 可解时：`A * x0 = b` 精确成立；每个基向量 `z` 满足 `A z = 0`；全部整数解为
  `x0 + Σ k_i z_i（k_i ∈ Z）`。基取 `V` 的第 `r, …, n−1` 列，**在整数环上直接
  推导，不经过有理解再取整**。
- 不可解时 `obstruction` 给出两类证书之一：
  - `nondivisible_pivot`：某非零对角元 `d` 不整除 `(U b)[i]`，给出非负余数
    `residue` 与 `U` 的对应行 `u`，满足 `u·A` 各分量被 `d` 整除而
    `d ∤ u·b`；
  - `zero_row_nonzero_rhs`：`u·A = 0` 而 `u·b ≠ 0`（秩不足导致的不相容）。
  注意：Q 上有解但 Z 上无解（余数非零）与 Q 上即无解（零行障碍）在证书中
  明确区分。

### `replay_certificate(A, operations) -> (U, D, V)`

独立重放器：把日志中的每条操作实例化为初等矩阵、做精确矩阵乘法重建
`U`、`V` 并计算 `D = U A V`。不调用消元算法，供证书独立核验。

## 错误语义

- `InputValidationError(ValueError)`：输入不满足契约。错误信息带可定位下标
  （如 `A[1][2] 不能是 NaN`），**不静默纠正**。包括：
  - 拒绝 `NaN`、`+Infinity`、`-Infinity`（仅在出现 float 时检查并明确报错）；
  - 拒绝一切 `float`（本库只做精确整数运算）；
  - **拒绝 `bool`**（`True`/`False` 虽是 `int` 子类也不接受）；
  - 拒绝非整数类型、空矩阵、空行、参差（行长度不一致）矩阵；
  - 超过 8 行/列、元素绝对值超过 10000；
  - `b` 长度与行数不符。
- `BudgetExhaustedError(RuntimeError)`：初等变换次数超过 `budget`
  （默认 100000）。这表示**算法预算耗尽，不是“无解”的证明**；增大 `budget`
  可重试。可解性问题的“无解”只通过 `integer_solve` 的 `solvable=False`
  及障碍证书表达，二者严格分开。本契约的输入规模（8×8、|元素|≤10000）下
  默认预算远大于实际需要（随机 8×8 实测 ≤ 数百次变换）。
- 所有异常都继承 `SmithCertError`。

## 数值精度口径

- 本库**全程使用 Python 任意精度整数，不存在任何浮点运算**，因此没有
  舍入误差、没有误差容差（tolerance）：`U*A*V == D`、`A*x0 == b`、
  `A*z == 0`、余数判定全部按整数精确相等/整除核验。
- 测试中的独立参考（`fractions.Fraction` 高斯消元、Bareiss 行列式、
  全部 k 阶子式 gcd）同样是精确运算；不存在“用宽容差掩盖边界错误”的情形。

## 算法与设计取舍

1. **整欧几里得消元**：每个主元位置先用行/列交换把非零元移到对角；再用
   “截断除法（向 0 取整）求余 + 交换”对主元行/列做欧几里得约化，余数交换
   每次严格减小主元绝对值，故必然终止；随后整除清零，主元变号为正。
2. **整除链修复**：朴素对角化不保证 `d_i | d_{i+1}`（如 `diag(6,10,15)`
   的正确 SNF 是 `diag(1,30,30)`）。主元清零十字后，若剩余子块存在不被主元
   整除的元素，用一次列加法把它带入主元列再重新约化，直到主元整除整个
   剩余子块，从而保证不变因子整除链。
3. **证书导向**：U、V 随每个初等操作就地同步维护，完整操作日志可独立重放；
   求解直接复用 Smith 证书，`x0 = V·(c_i/d_i, 自由变量=0)`，零空间基为
   V 的后列——不产生分数，也不存在“有理基再清分母”的取整路径。
4. **终止性预算化**：变换计数与解耦的异常类型使“没算完”和“证明无解”
   不可混淆。
5. 测试期望值来自**独立参考**（Bareiss 行列式；全部 k 阶子式 gcd 的
   行列式因子 oracle Δ_k，`d_k = Δ_k/Δ_{k-1}`；`Fraction` 消元秩/零空间），
   独立参考不调用被测核心；`U*A*V=D` 等往返一致只作补充核验。另含固定种子
   的确定性随机矩阵批量性质测试。

## 已知限制（本题契约范围）

- 仅实现本题要求的子集：整数矩阵 Smith 标准形与整数线性方程组；
  不宣称兼容任何行业标准协议，也不提供一般 PID/主理想整环或多项式矩阵功能。
- 矩阵规模与元素范围被接口限定为 8×8、|元素| ≤ 10000（`b` 无元素上限）。
  核心算法本身对更大输入在数学上仍可用，但超出本题声明与验证范围。
- 零空间基是一组整数基（V 的后列，Q 上线性无关且恰好张成整数解的仿射格），
  不额外做 LLL 短化或 Hermite 规范形风格的最小化；基向量可能不是最短的。
- 不包含并发/批处理接口；所有操作均为无状态函数调用，不存在部分变更问题。
- 无 CLI 产品：终端命令仅用于运行测试与固定输入演示。

## 仓库结构

```
smithcert/            题库（纯标准库）
  __init__.py          公开接口 smith / integer_solve / replay_certificate
  _basic.py            输入校验、精确整数矩阵乘法、Bareiss 行列式
  _smith.py            Smith 消元引擎（含预算与日志）
  _solve.py            整数可解判定、特解、零空间基
  _ops.py              六类初等操作数据类型
  _replay.py           独立日志重放器
  _errors.py           异常类型
tests/                 必要测试（python -m unittest discover -s tests -v）
  _reference.py        独立参考：子式 gcd oracle、Bareiss、Fraction 消元
  _checks.py           共享的独立核验断言
  test_smith.py / test_solve.py / test_validation.py / test_property.py
demo.py                固定输入真实计算演示（python demo.py）
```
