# P0 评审点 B：内联质量评审

日期：2026-09-28。分支：feat/p0-foundation。范围：P0 的五个 crate、参考夹具、环境文档、任务记录与新增训练规范。按用户要求内联评审，未派生子代理。

## 结论

13 步实施与验证完成。trellis-check 的代码质量、测试覆盖、规范同步、依赖方向和跨层状态流检查通过。P0 可以提交。评审点 A 已获用户确认；本轮无需重复申请实施或提交批准。

## 验收项映射

| PRD 验收项 | 机制与验证 | 结果 |
| --- | --- | --- |
| AC1 CPU 构建不依赖 CUDA 环境 | 默认 features 为空；移除 CUDA/CUDNN 环境变量和 Toolkit PATH 后 cargo build --workspace | 通过；step-13-cpu-final.log |
| AC2 CUDA 构建及环境记录 | native Windows，统一静态 CRT；Candle =0.11.0；docs/environment.md 记录 SHA、环境、失败历史 | 通过；step-13-cuda-final.log |
| AC3 doctor 与显式设备失败 | core 解析 → nn resolve_device → Candle context → cudarc 查询；真实六参数网络与 AdamW；cuda:9 返回 1 | 通过；step-10-doctor-static-crt.log、step-10-doctor-invalid-device.log，CLI 2 项测试 |
| AC4 CPU F64 有限差分 | train_ops 的 5 项测试；softmax、masked softmax、LN x/gamma/beta、总体方差、attention Q/K/V | 通过；常量列、L=1、全掩码均覆盖 |
| AC5 CPU/CUDA F32 一致性 | cuda_consistency 的 3 项测试；atol=1e-5、rtol=1e-4；dropout mask 完全一致 | 通过 |
| AC6 无反向路径检测 | fused_path_cuts_grad 2 项测试；no_fused_in_train 1 项；临时引入禁止调用后守卫失败，再恢复源码 | 通过；step-07-source-guard.log |
| AC7 Torch 优化器夹具 | 两种优化器、两个参数组、10 步、缺失梯度；参数与两种矩状态 atol=1e-6 | 通过；fixture metadata 保存超参数 |
| AC8 恢复训练 | 10 步连续与 5+保存/加载+5；CPU 参数/矩逐位一致，CUDA atol=1e-6；计数器一致 | 通过 |
| AC9 全量门槛 | fmt、CPU/CUDA Clippy -D warnings、CPU/CUDA workspace tests | 通过；CPU 24 项，CUDA 30 项，失败与忽略均为 0 |

## 最终检查命令

CPU 验证使用固定 rustc/rustdoc 1.98.0，并在进程内移除 CUDA 变量与 Toolkit PATH。GPU 验证使用 docs/environment.md 的静态 CRT 设置，单独的 target/windows-static 目录。以下命令全部退出 0：

- cargo fmt --all --check
- cargo build --workspace
- cargo clippy --workspace --all-targets -- -D warnings
- cargo test --workspace
- cargo build --workspace --features cuda
- cargo clippy --workspace --all-targets --features cuda -- -D warnings
- cargo test --workspace --features cuda

证据：step-13-cpu-final.log、step-13-cuda-final.log。检查是在最后一次 Rust 测试修改后执行的。后续只更新文档、验收状态和提交元信息。

## 评审修正与实现说明

1. 事务性加载测试原先使用 detach 作为参考快照，仍与 Var 共享存储。已改为 detach().copy()，确保前一个参数被错误修改时测试能够检测。修正后重新执行两套 Clippy 与全部测试。
2. RNG 放在 nn 并由 train 重导出，避免依赖环。显式小端 32 字节 key 取代未指定的 hash。设计文档已同步。
3. InferenceOnly 公开构造函数无法限制调用方模块。入口额外拒绝任何 track_op 张量，覆盖输入与 affine 参数。设计文档与回归测试已同步。
4. CLI 为同一 cudarc 版本增加 runtime feature，以读取实际 CUDA 版本；GPU context 仍由 Candle 创建。
5. 新增 backend/training-foundation.md，覆盖接口、错误矩阵、状态文件格式、随机流、检查命令与错误用法。未扩展无关占位规范。

## 跨层检查

- Core 不依赖 Candle；cli → train → nn → core、data → core 无循环。
- 设备解析错误、CUDA 初始化错误逐层返回；无 CPU 回退分支。
- nn 使用基础可微运算；推理融合路径隔离；源码守卫纳入 cargo test。
- 模型名称、形状、dtype 与优化器参数组均校验后加载。读取路径为 safetensors/JSON → 完整校验 → 替换状态；保存到调用方 staging 目录。完整检查点发布事务由后续任务负责。
- NamedParams 克隆共享变量；测试和 doctor 的更新前快照使用独立张量副本或主机向量。
- Python 仅用于开发期生成合成夹具；Rust 构建不依赖 Python。data crate 保留 P0 空骨架。

## 环境偏差与剩余说明

- LNK4098 已通过统一静态 CRT 消除，无 /NODEFAULTLIB、allow 或 ignore 警告设置。
- MSVC 中文“正在创建库”信息仍触发 rustc linker_messages。该信息未被抑制。最终 CPU/CUDA Clippy -D warnings 均通过；链接与测试均成功。
- nvcc 报告 13.3.33，运行期 CUDA API 报告 13.4。两者按实际输出分别记录，版本差异原因未查明。
- 直接 Cargo 曾触发依赖目录下的 rustup 自动安装 Rust 1.78.0。该安装未事先征得同意，已向用户报告并记录。之后禁用自动安装、固定 rustc/rustdoc 1.98.0；未经同意未删除该工具链。未安装 CUDA 12.9 或 WSL Toolkit。
- task.py validate 通过；已有总计划书超过 32,768 字节注入上限的提示仍存在。本轮内联读取任务计划与相关证据，该提示没有阻止检查。
- P1 当前只有 PRD，状态为 planning，design.md 和 implement.md 尚不存在。交接时必须完成详细规划与审批，再启动 P1 实施。

## 提交边界

仅提交 P0 代码、合成夹具、环境/规范与任务验证记录；不提交 data/、runs/、target/ 或 Python 虚拟环境。提交前对暂存区再次运行 tools/check_private_data.sh 和 git diff --cached --check。仅本地提交，不推送。
