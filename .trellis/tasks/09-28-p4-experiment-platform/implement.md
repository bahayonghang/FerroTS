# P4 实施计划

planning；按增量审批和执行。首个compare报告不等待全部模型。

1. [ ] 冻结run输入合同与协议比较边界，整理已有P1/P2证据及不可重放的历史参考范围（AC1）。
2. [ ] 实现compare与Markdown/HTML静态报告，核对逐run/汇总数值、缺失格及私有数据隔离，先交付首个报告（AC1、AC2）。
3. [ ] 按custom-lsf-v1生成首批三模型180格manifest；明确Informer前置和ILI长度，CUDA短运行测大通道显存/耗时，审批分批预算（AC3、AC6）。
4. [ ] 分批完成五公共数据集，按格保留成功/失败/缺失；发布来源明确的均值/std表，不用部分矩阵宣称全部完成（AC3）。
5. [ ] 实现ETTh1/水泥rolling-origin，每折重新fit训练状态与scaler，验证折边界、origin维度及防泄漏（AC4）。
6. [ ] 实现验证集搜索、单独final-test阶段与最小单/双GPU独立实验管理；验证测试读取次数和局部恢复，汇总性能（AC5、AC6）。
7. [ ] 为选中的PatchTST或其他候选另建任务并审批；复用协议，完成公共/水泥验收后加入结果矩阵（AC7）。
8. [ ] 适用CUDA质量/工具测试、报告重建、私有guard与独立证据评审，更新docs/results和能力表后提交收尾（AC8）。

基础验证命令采用P3 implement.md的固定Linux CUDA环境与fmt/Clippy/workspace/tool tests；针对compare的合成产物测试无需训练，数据防泄漏用行为证据而非重述实现。涉及模型行为则CUDA优先，CPU诊断不替代验收。

产物在runs/p4-experiments/<suite-id>/；公开统计在docs/results/p4-*.md，私有图与逐行预测留在忽略目录。已完成运行只有精确身份匹配且receipt复核通过才复用，不覆盖失败attempt。不同阶段分别提交，搜索/可选索引不阻塞增量A。
