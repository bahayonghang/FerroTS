# datasets-protocol 技术设计

## 1. 产物布局

```text
docs/datasets/
  README.md              # 数据目录约定、协议 ID 索引、私有数据规则
  manifest.toml          # 公共文件清单（可提交）
  ett.md                 # 协议 ett-lsf-v1 + 参考数值
  custom-lsf.md          # 协议 custom-lsf-v1（Weather/Exchange/ECL/Traffic/ILI）+ 参考数值
  cement-yangquan.md     # 数据卡 + 协议 cement-twin-v1（只含统计量）
tools/datasets/
  verify_manifest.py     # 标准库实现：按 manifest 校验字节数与 SHA256
  profile_cement.py      # 由父任务 research/profile_cement.py 迁入，扩展运行段统计
tools/check_private_data.sh
tools/reference/         # 与 P0 共用 uv 项目；增加 TSLib/iTransformer 官方代码运行脚本
.gitignore               # 增加 runs/、tools/reference/.cache/
```

## 2. `manifest.toml` 格式

```toml
schema_version = 1

[[source]]
id = "hf-tslib"
repo = "thuml/Time-Series-Library"
repo_type = "dataset"
revision = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"

[[file]]
path = "data/ETT-small/ETTh1.csv"
source = "hf-tslib"
upstream_path = "ETT-small/ETTh1.csv"
bytes = 2589657
sha256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
```

私有数据不写入 `manifest.toml`；其字节数与 SHA256 只写在 `cement-yangquan.md`。

## 3. 协议文档结构

每个协议文档包含同一组小节，P1 按此转写为 DataSpec TOML：

| 小节         | ett-lsf-v1                                 | custom-lsf-v1                                | cement-twin-v1                                           |
| ------------ | ------------------------------------------ | -------------------------------------------- | -------------------------------------------------------- |
| 文件与时间列 | `date`，`%Y-%m-%d %H:%M:%S`                | `date`；Exchange 为 `%Y/%-m/%-d %-H:%M`      | `DateTime`，ISO 8601 带微秒                              |
| 预处理       | 无                                         | 无                                           | 1 min 均值；每分钟有效样本 ≥ 3；运行段规则；受限因果填充 |
| 切分         | 固定行边界（小时 8640/2880/2880；分钟 ×4） | 70/10/20 比例                                | 按日期 70/10/20，边界落在段间隔                          |
| L / H        | 96 / {96,192,336,720}                      | 取自官方 iTransformer 脚本；ILI 取官方专用值 | 96 / {15,30,60}（min）                                   |
| 变量         | 7 个全部                                   | 全部                                         | 18 个全部                                                |
| 时间特征     | 与官方脚本 `--freq`、`timeenc` 一致        | 同左                                         | 与 1 min 频率对应的 `timeF`                              |
| 指标         | 标准化 MSE/MAE（主），原始单位             | 同左                                         | 分变量原始单位 MAE/RMSE（主），标准化 MSE/MAE            |
| 验收         | D4 门槛（P1–P3）                           | P4 报告，无门槛                              | 与基线并列报告                                           |

L/H 与时间特征参数必须从固定 SHA 的官方脚本中读取并在文档中引用具体文件与行号，不凭记忆填写。

## 4. `cement-twin-v1` 运行段规则

- 在 1 min 聚合后的序列上计算：`窑头煤反馈 > 5` 的连续分钟构成候选段；候选段长度 ≥ 30 min 记为运行段。
- 运行段内连续缺失 > 5 min 时断段。
- 窗口 `[o-L, o+H)` 必须完全落在同一运行段内。
- 切分边界按日期确定后，对齐到最近的段间隔；数据卡报告各切分的运行段数、总分钟数、有效窗口数。
- 默认值（最少样本数 3、断段 5 min、陈旧时间 10 min）若在统计中出现明显不合理（例如某切分有效窗口数为 0），在文档中记录调整与理由。

## 5. 官方基准运行环境（`tools/reference/`）

- `uv` 项目，固定 Python 3.12 与支持 sm_120 的 torch 版本（CUDA wheel）；版本号写入 `pyproject.toml` 与 `docs/environment.md`。
- `fetch_upstream.py`：把 TSLib 与 iTransformer 按固定 SHA 克隆到 `tools/reference/.cache/`（不提交）。
- `run_tslib.py`：包装官方 `run.py`，`--root_path` 指向本仓库 `data/`，参数取自官方脚本；输出写入 `runs/reference/`。
- 本任务只验证 DLinear/ETTh1/H=96 可运行；完整 3 种子基准在 P1–P3 各自执行。

## 6. 私有数据检查脚本

`tools/check_private_data.sh`：

1. `git ls-files | grep -E '^(data|runs)/'` 有输出 → 退出码 1。
2. 对 `git ls-files 'tests/fixtures/**.csv'` 的每个文件读取首行，包含 `窑头煤反馈` 或 `DateTime,窑喂料反馈` → 退出码 1。
3. 其余情况退出码 0。

## 7. 取舍

- 公共文件校验使用 Python 标准库脚本：跨 Windows/WSL 一致，不增加 Rust 构建依赖。P1 的 Rust 数据层另外按数据指纹工作，两者互不依赖。
- 协议以 Markdown 文档固定参数：DataSpec 格式由 P1 设计，避免本任务提前锁定 TOML schema。
