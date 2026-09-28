# datasets-protocol 执行计划

## 检查清单

1. [x] `.gitignore` 增加 `runs/`、`tools/reference/.cache/`。
   - 验证：`git check-ignore runs/x tools/reference/.cache/x`
2. [x] `tools/check_private_data.sh`。
   - 验证：`bash tools/check_private_data.sh`（退出码 0）；在临时索引上验证失败路径：`cp .git/index "$TMP/idx" && GIT_INDEX_FILE="$TMP/idx" git add -f data/ETT-small/ETTh1.csv && GIT_INDEX_FILE="$TMP/idx" bash tools/check_private_data.sh`（退出码 1），真实索引不变
3. [x] `docs/datasets/manifest.toml`（9 个公共文件，数值取自父任务 `research/public-datasets.md`）与 `tools/datasets/verify_manifest.py`。
   - 验证：`python tools/datasets/verify_manifest.py docs/datasets/manifest.toml`（9 个文件全部 OK）
4. [x] `tools/reference/`：`fetch_upstream.py` 固定 TSLib 与 iTransformer SHA 并克隆到 `.cache/`；在文档中记录 SHA。
   - 验证：`uv run --project tools/reference python tools/reference/fetch_upstream.py`
5. [x] 从固定 SHA 的官方脚本读取 ETT 与 5 个非 ETT 数据集的 L、H、`--freq`、`timeenc` 等参数，写入 `docs/datasets/ett.md` 与 `docs/datasets/custom-lsf.md`，每个参数附来源文件与行号。
6. [x] 收集公开参考数值（iTransformer、Informer、DLinear），附论文表号或仓库文件与 SHA。
7. [x] `run_tslib.py` 在本机运行官方 DLinear / ETTh1 / H=96 一次。
   - 验证：输出 MSE/MAE，结果写入 `runs/reference/`（不提交）
8. [x] `tools/datasets/profile_cement.py`：迁入父任务脚本，增加 1 min 聚合与运行段统计（段数、段长分布、各切分有效窗口数）。
   - 验证：`PYTHONIOENCODING=utf-8 uv run --with polars --with numpy python tools/datasets/profile_cement.py`
9. [x] `docs/datasets/cement-yangquan.md`：数据卡 + `cement-twin-v1` 全部参数；只含统计量。
   - 验证：人工核对行数、缺失数、断档数与父任务 `research/dataset-profile.md` 一致；`grep -c '^2023-\|^2024-\|^2025-' docs/datasets/cement-yangquan.md` 输出 0（无原始行）
10. [x] `docs/datasets/README.md`：目录约定、协议索引、私有数据规则。
11. [x] 收尾检查：`bash tools/check_private_data.sh`；`python tools/datasets/verify_manifest.py docs/datasets/manifest.toml`；运行 `trellis-check` 对照 prd.md 验收项。

## 风险与回滚点

- 第 7 步需要安装 torch（数 GB）；若 sm_120 的 torch wheel 不可用，记录错误并在 P1 开始前解决，不阻塞第 8–10 步。
- 本任务只新增文档与脚本；回滚为回退本任务提交。

## 启动前检查

- [x] `implement.jsonl` / `check.jsonl` 已列出计划书与父任务研究记录。
- [x] 用户已确认最终规划摘要。
