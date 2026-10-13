# Informer 设计

## 来源与当前边界

目标为[TSLib固定模型](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/Informer.py)，不混用其他仓库结构。固定源encoder及decoder self/cross均用ProbAttention；SelfAttention_Family.py声明但未调用attention-weight dropout。参考权重、采样索引、buffer和真实损失梯度只从公共/合成输入生成。

当前 crates/ferrots-nn/src/model.rs:11 的输入已隔离监督，:40为静态派发入口，:31的eval forward_id固定0，不能直接代表上游采样状态。crates/ferrots-train/src/session.rs:95 的snapshot无buffer；checkpoint/training.rs:124拒绝非空buffer，:401保存空buffer。这些是需贯穿的接口。

## 模型与输入

扩充 ModelSpec、ForecastModel、registry，沿用现有Train/Eval派发。配置来自固定脚本：层数、维度、heads、factor、distil、activation、dropout、label_len。模型内将history尾部label_len步拼接zeros[B,H,C]，时间特征同样拼接，从decoder末尾取H步。ModelInput不新增监督值。初版gap=0，掩码监督及输出映射沿用当前训练边界。

## 可导算子与稀疏性

- PrefixSum：CUDA逐行scan，后向反向scan，不使用Candle的平方级cumsum展开；先保证正确可导。
- SampledQkScore/TopQuerySelect：仅采样key子集，选top-query后计算u×Lk分数。索引不求导，gather/scatter的值可导。
- ReplaceRows：覆盖初始上下文，被覆盖旧值梯度为零，更新值接收对应梯度；选中query必须唯一。
- Causal mask：从选中query原位置直接构造[B,heads,u,Lk]，不照抄上游先创建L×S再索引的ProbMask。
- 蒸馏：CircularConv1d padding=2，BatchNorm1d、ELU、MaxPool1d(kernel3,stride2,padding1)；保存argmax，重叠窗口梯度累加，tie与夹具一致。
- B/head=1保留轴；L=1保证至少一个sample/query。边界处理单列，普通形状严格对齐上游。

PrefixSum显存口径：先创建FP32输入与输出cotangent并同步作为基线；测量forward开始至backward完成的活跃分配峰值增量，计入输出、返回输入梯度和workspace，不计基线已有输入/cotangent。使用可靠allocator计数或完整生命周期的专用分配记录；测不到可信峰值则AC2保持未通过，不能用50ms nvidia-smi采样推断。复杂度门同时断言score/mask实际shape，不只拟合显存曲线。

## 随机性、buffer与恢复

训练每次forward从持久化训练身份派生采样，和dropout分离site。验证/预测使用独立可重建身份（阶段、epoch/run、batch、layer/site），验证不推进训练流；预测记录种子/采样身份。逐层夹具直接共享索引，正式多种子用D4判定，不宣称逐bit重现PyTorch RNG。

模型边界增加命名buffer读取/恢复，含running_mean、running_var、num_batches_tracked；不注册为Var、不进Adam。snapshot、manifest、save/load、resume、freeze_for_eval使用同一buffer合同；先核对名称全集/shape/dtype/版本再整体应用。基线/iTransformer仍为空buffer，不加迁移层。

独立freeze复制权重与buffer，验证不改训练对象。若采样状态改变恢复身份版本，在版本边界明确拒绝不兼容恢复。仍在optimizer-step/空梯度边界原子发布generation。

## 验收映射

| AC | 机制 | 实施步骤 |
| --- | --- | --- |
| AC1 | 共享权重/索引夹具，全参数真实损失梯度 | 2–3 |
| AC2/AC3 | scan分配峰值、u×Lk score/mask断言 | 2、6 |
| AC4/AC5 | buffer端到端合同、隔离随机流、独立freeze | 4–5 |
| AC6 | 模型内decoder与标签隔离 | 3、5 |
| AC7 | 官方48+Rust48及原D4 | 7 |
| AC8 | 水泥9次、按名称聚合、独立复算 | 8 |
| AC9/AC10 | 身份/收据/性能、能力表、最终质量 | 6–9 |

ETT采用ett-lsf-v1及固定TSLib脚本profile。水泥采用cement-twin-v1全部合法窗口，H15/30/60、种子2021/2022/2023。一卡一个进程，记录实际设备。P1跨主机单种子表注明历史参考，不跨GPU宣称性能提升。

运行时、工具、配置或上游变化即使用新suite身份，保留失败attempt。正式矩阵前先评审CUDA短运行预算。未过全部门保持preview。
