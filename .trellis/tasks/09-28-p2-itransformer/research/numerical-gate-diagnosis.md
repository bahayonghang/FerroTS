# P2 第 7 步数值门诊断

日期：2026-09-29。状态：原 AC3 失败记录保留。2026-09-29 用户明确回复“批准 AC3 修订，继续实施（推荐）”。第 5 节修订已同步 PRD/design/implement；修订后的第 7 步已验证：CPU 1 passed、CUDA 构建 2 passed（含 CPU/CUDA），日志分别为 runs/p1-development/20260929-021908-3702758.log 与 20260929-021841-8428835.log。

## 1. 结论与边界

九组合成夹具、每组 38 个参数的真实 MSE 梯度和连接性探针在 CPU/CUDA 均通过。单步 Adam 的参数更新对照在每个设备各失败 16 项，全部为两层 attention 的 key_projection.bias；长度为 1 的案例没有失败。两种矩和全部 step 计数均通过。

相同官方 MSE 梯度直接输入 Rust Adam 后，CPU/CUDA 全部参数、两种矩和 step 均通过原阈值。固定官方模型自身在 CPU 与 CUDA 间也产生 16 项 key bias 更新超限。该对照关闭 TF32，使用同一权重、输入、目标、FP32、单步 Adam 和固定超参数。官方源码、原始夹具和现行阈值未改动。

现有证据把失败定位到不同后端真实 MSE 反向中接近零的 key bias 梯度残差，经 Adam 分母中的 eps 放大。尚未逐个归因每个底层归约内核的舍入差异。不能据此宣告完整 P2 实现正确，也不能把同梯度优化器对照替换为现行端到端门后宣告通过。

## 2. 实测结果

每个设备共 9 × 38 = 342 个参数实例。前向/梯度容差为 atol=1e-5、rtol=1e-4；Adam 参数和矩为 atol=1e-6、rtol=0。

| 对照 | CPU | CUDA |
| --- | ---: | ---: |
| 全部逐层/最终前向案例 | 9/9 通过 | 9/9 通过 |
| MSE 参数梯度 | 342/342 通过 | 342/342 通过 |
| 连接性探针梯度 | 342/342 通过 | 342/342 通过 |
| MSE 梯度最大绝对误差 | 2.384185791015625e-7 | 1.4901161193847656e-7 |
| 真实 MSE 单步 Adam 参数 | 326/342 通过 | 326/342 通过 |
| 参数最大绝对误差 | 1.6952864825725555e-5 | 2.126116305589676e-5 |
| 一阶矩 / 二阶矩 / step | 各 342/342 通过 | 各 342/342 通过 |
| 同官方梯度 Adam 参数 / 两种矩 / step | 各 342/342 通过 | 各 342/342 通过 |
| 同官方梯度参数最大绝对误差 | 1.4901161193847656e-8 | 1.4901161193847656e-8 |

官方 CPU 重新执行与原夹具的参数检查全部通过。官方 CUDA 的 IEEE FP32 对照仅有 16 项 updated 检查失败，最大误差为 2.126116305589676e-5，约为现行阈值的 21.26 倍。其前向、梯度、矩检查均通过。

首轮官方 CUDA 诊断使用 cuDNN 默认 allow_tf32=true，出现 167 项参数检查失败。该记录保留。第二轮显式禁用 matmul/cuDNN TF32 后得到上述 16 项结果。诊断设置只影响该诊断进程，未更改正式 runner 或训练配置。

## 3. 数值机制

对同一个 query，给所有 key 加上相同 bias，会给 attention score 的同一行增加同一个常数。softmax 对行常数平移不变。因此在精确算术下，真实 MSE 对共享 key bias 的梯度为零。连接性探针另外使用 key projection 的平方均值，能验证该参数仍连接计算图。

FP32 反向可能得到不同的非零舍入残差。标准 K4 案例第二层 key bias 的第 4 个元素（零起始索引），官方 CPU 梯度为 -1.3969838619232178e-9，官方 CUDA 为 9.89530235528946e-10；两者都满足现行梯度容差。

第一次 Adam 更新在数学上为 lr × g / (abs(g) + eps)。固定 lr=1e-4、eps=1e-8，零附近的更新对梯度的敏感度约为 10000。上述残差差异产生约 2.1261e-5 的参数差异。该结果解释了梯度门通过而更新门失败，且与官方模型自身的设备对照一致。

## 4. 可复核证据

- Rust 全参数检查：crates/ferrots-train/tests/p2_numerics.rs。失败断言保留，未增加 ignore 或排除 key bias。
- 官方设备对照：tools/validation/p2_numerical_diagnosis.py。读取固定官方类及既有合成夹具，不重写夹具。
- CPU 完整报告：runs/p2-development/numerics-cpu-1790665902688502900.json。
- CUDA 完整报告：runs/p2-development/numerics-cuda-1790665958538339900.json。
- 官方默认 CUDA 设置报告：runs/p2-development/official-cpu-cuda-control-20260929.json。
- 官方 IEEE FP32 报告：runs/p2-development/official-cpu-cuda-ieee-control-20260929.json。
- CPU 失败日志：runs/p1-development/20260929-021140-0069184.log。
- CUDA 失败日志：runs/p1-development/20260929-021234-7972934.log。
- 官方 IEEE FP32 命令日志：runs/p1-development/20260929-021230-0911246.log。

复核官方对照时使用一个新的报告路径；脚本使用排他创建，不覆盖原证据：

    . ./tools/validation/p1_env.ps1 -Mode cuda
    Invoke-P1Reference tools/validation/p2_numerical_diagnosis.py --ieee-fp32 --output runs/p2-development/official-control-new.json

## 5. 已批准的验收修订方案

用户于 2026-09-29 明确批准将 AC3 拆成模型反向门和优化器门。该修订改变原端到端参数更新验收语义；原失败记录保留，不能表述为原 AC3 通过。

1. 保留所有参数的真实 MSE 梯度门、非零连接性探针及原 atol=1e-5、rtol=1e-4，不排除 key bias。
2. 使用同一份官方真实 MSE 梯度输入两端 Adam，比较全部参数、两种矩和 step，保留 atol=1e-6、rtol=0。全部 38 个参数都参加。
3. 继续保存各后端独立反向后更新的全参数误差清单，明确记录 key bias 的端到端差异，不把该诊断结果写成“原 AC3 通过”。
4. 保留正式训练的真实 MSE、key bias、Adam eps/lr、种子、dropout、完整 106 次矩阵和 32 项 D4 门。不得归零梯度、移除参数、使用探针训练或修改正式指标。

批准后先按修订后的 AC3 重新执行 CPU/CUDA 第 7 步。通过后依次进入第 8–20 步。正式训练仍受恢复门和完整回归门约束。
