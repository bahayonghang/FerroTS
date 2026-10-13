# 实验比较与扩展设计

## 最小结构

复用现有run目录的resolved config、数据/scaler/窗口指纹、检查点身份、指标和receipt。先实现compare及静态报告，不先建立数据库或通用调度框架。当前P2套件格式是已核验入口；P1本机原目录缺失，只能将已提交表明确作为历史数字参考，不能伪造receipt或公平性能比较。

比较在一次边界检查协议版本、数据/切分/目标变量/单位/评价窗口；标签相同但fingerprint不同也拒绝。不同模型源码/设备本来可能不同，展示身份与公平性限制，不以源SHA相同作为跨模型比较条件。无raw evidence的引用不冒充完整可重放run。

## 增量A：compare、报告与扩展矩阵

compare实现CLI到统一结果表，报告复用该表；不并行维护另一份指标。浮点计算用完整精度，展示可格式化，std为ddof=0。水泥按变量名聚合；同一target时间的不同origin不被无意平均。

扩展矩阵采用docs/datasets/custom-lsf.md，不重新定义切分；每模型Weather/Exchange/ECL/Traffic/ILI各4H×3seed=60。最初DLinear/iTransformer/Informer三模型180；Informer未通过时对应格保持未运行，P1/P2的比较报告仍可先交付。WaveForM加入则总240；其Weather12格只有身份/profile/评价合同均匹配才复用，否则重新生成本矩阵身份。

非ETT不设父D4门，但必须全窗口覆盖和完整来源。无上游对应脚本的配置标“派生”，公开参考数值标数据/协议差异，不把不同切分数字当公平胜负。每批先用CUDA短运行预算，Traffic/ECL等大通道任务测峰值后再放行。

## 增量B：回测

复用Dataset/scaler/TrainingSession，每折新建状态，不能携带上折Adam、early-stop或scaler。ETT按时间顺序；水泥只在完整稳定运行段边界分折。每条预测绑定fold/origin/target时间/step，统计保留origin维度；训练只使用该折允许的历史。回测与固定split正式复现是不同协议身份，不混表。

## 增量C：搜索与调度

网格/随机搜索只消费验证集指标。选中配置冻结后由独立final-evaluate阶段打开测试集一次，记录可观测输入读取与命令receipt。预算包含试验数/最长epoch/设备，随机搜索种子固定。

先用独立CUDA进程和设备分配表，一卡一实验；失败保留attempt并仅恢复该格，已完成格重新校验后可复用。分阶段timer在CUDA边界同步；全卡采样与allocator内存分开标识。SQLite只在已出现查询性能需求时另行设计，不新增生产依赖。

## 验收映射

| AC | 机制 | 实施步骤 |
| --- | --- | --- |
| AC1/AC2 | 单一结果表、比较边界、身份与原始产物绑定 | 1–2 |
| AC3 | 精确180格manifest，五协议profile与分批预算 | 3–4 |
| AC4 | 每折独立训练/scaler，origin索引 | 5 |
| AC5 | 验证选择与独立测试阶段，读取receipt | 6 |
| AC6 | 每卡独立attempt、同步timer及内存来源 | 3、6 |
| AC7 | 候选模型各自任务与验收 | 7 |
| AC8 | CUDA质量、报告/隐私/能力表最终复核 | 8 |

候选优先PatchTST监督预测，随后根据水泥结果与实现成本选TSMixer或TimeMixer/TCN；TSMixer与TimeMixer不混名。选型和新模型完整设计在独立任务完成，P4不预先扩展所有模型配置。
