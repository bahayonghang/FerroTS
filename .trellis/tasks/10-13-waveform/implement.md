# WaveForM 实施计划

状态planning，不启动实现或训练。执行前审批本计划；研究首关未通过时不启动后续依赖步骤。

1. [ ] 核对论文/仓库SHA/许可、缺失依赖清单，决定参考运行和实现依据，写入研究结论（AC1）。
2. [ ] 冻结三协议、种子、split/scaler/列序/尾批及训练日程；参考适配变更单列，比较窗口数量（AC2）。
3. [ ] CUDA Haar DWT/IDWT正逆重建及真实损失反向，对比symmetric边界和奇数/H15/30/60/B1/尾批（AC3、AC4）。
4. [ ] 实现共享图、top-k、膨胀卷积、双向MixHop与系数预测头；逐层前向和全参数梯度，接入配置/静态派发（AC3、AC4）。
5. [ ] CUDA短训练与训练/保存/恢复/评价/预测闭环；验证对象隔离，记录图/小波成本与129次预算；批准后冻结suite（AC5、AC8）。
6. [ ] ETT官方48+Rust48、Weather官方12+Rust12，计算32+8项同协议对齐检查，保留全部失败证据（AC6）。
7. [ ] 水泥9次、108格报告、H60三变量独立复算（AC7）。
8. [ ] 跑适用最终CUDA质量门并独立审阅，更新NOTICE、模型卡、能力表与结果报告，核对129次与preview边界后提交收尾（AC8）。

验证环境及基础命令沿用P3 implement.md所列fmt/CUDA Clippy/workspace tests、参考工具tests、私有数据guard与diff检查。只运行相关CUDA行为测试，不调度常规CPU-only训练。新runner/fixture命令在步骤2定义后实现，当前不存在。

产物在runs/waveform-acceptance/<suite-id>/，报告在docs/results/waveform-{ett,weather,cement,performance}.md。失败不覆盖旧attempt；源码/工具变化建立新身份。若许可或CUDA算子门不能完成，保留研究状态并推进独立任务，不发布已复现声明。
