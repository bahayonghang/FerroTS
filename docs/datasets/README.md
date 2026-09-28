# 数据集与评测协议

模型子任务通过协议 ID 引用本目录的切分、窗口与指标定义。P1 把协议参数转写为 DataSpec 配置文件。

## 协议索引

| 协议 ID          | 数据                                 | 文档                 | 验收                           |
| ---------------- | ------------------------------------ | -------------------- | ------------------------------ |
| `ett-lsf-v1`     | ETTh1、ETTh2、ETTm1、ETTm2           | `ett.md`             | P1–P3 门槛（父任务 D4）        |
| `custom-lsf-v1`  | Weather、Exchange、ECL、Traffic、ILI | `custom-lsf.md`      | P4 报告，不设门槛（父任务 D5） |
| `cement-twin-v1` | 阳泉回转窑孪生样本（私有）           | `cement-yangquan.md` | 与基线并列报告                 |

协议参数发生变化时增加版本号（例如 `ett-lsf-v2`），不修改已发布版本的含义。

## 目录约定

| 路径                      | 内容                                          | 是否提交 |
| ------------------------- | --------------------------------------------- | -------- |
| `data/`                   | 公共数据与私有数据                            | 否       |
| `runs/`                   | 实验输出，包括 `runs/reference/` 官方基准运行 | 否       |
| `tools/reference/.cache/` | 按固定 SHA 获取的官方代码                     | 否       |
| `docs/datasets/`          | 清单、协议、数据卡                            | 是       |

- 公共数据文件路径与上游目录名一致，例如 `data/ETT-small/ETTh1.csv`。来源与校验值见 `manifest.toml`。
- 私有数据的字节数与 SHA256 只写在数据卡中，不写入 `manifest.toml`。

## 私有数据规则

- `data/` 与 `runs/` 不进入 Git 索引。
- `tests/fixtures/` 只放合成数据或公共数据；不放含水泥数据列名的 CSV。
- 数据卡与报告只写统计量，不写原始数据行。
- 提交前运行 `bash tools/check_private_data.sh`。违反前两条规则时退出码为 1。

## 工具

| 命令                                                                                               | 作用                                                           |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `python tools/datasets/verify_manifest.py docs/datasets/manifest.toml`                             | 按清单校验公共文件的字节数与 SHA256                            |
| `bash tools/check_private_data.sh`                                                                 | 检查私有数据与实验输出没有进入 Git 索引                        |
| `PYTHONIOENCODING=utf-8 uv run --with polars --with numpy python tools/datasets/profile_cement.py` | 生成 `cement-yangquan.md` 中的统计量                           |
| `uv run --project tools/reference python tools/reference/fetch_upstream.py`                        | 按固定 SHA 获取 Time-Series-Library 与 iTransformer 官方代码   |
| `uv run --project tools/reference python tools/reference/run_tslib.py --seed <n> -- <run.py 参数>` | 在本地数据上运行 TSLib 官方 `run.py`，输出到 `runs/reference/` |

### 官方基准运行环境（`tools/reference/`）

- Python 3.12；torch 2.14.0（CUDA 13.0 构建，支持 sm_120）；其余依赖版本取自 TSLib `requirements.txt`。版本固定在 `tools/reference/pyproject.toml` 与 `uv.lock`。
- `uv sync` 需要访问 PyPI 与 `download.pytorch.org`。网络需要代理时设置 `HTTPS_PROXY`。
- `run_tslib.py` 的 `--root_path` 相对仓库根目录，例如 `data/ETT-small/`。数据文件不存在时脚本停止，防止 TSLib 从 Hugging Face 下载数据。
- 3 种子基准：对同一组参数分别以 3 个 `--seed` 运行。TSLib 官方种子为 2021。
- 本环境的 numpy 为 2.1.2。iTransformer 官方代码 `utils/tools.py:34` 使用 numpy 2.0 已删除的 `np.Inf`，在本环境中运行会报 `AttributeError`。P2 运行 iTransformer 官方代码前需要处理该问题。

本机验证运行（2026-09-28，RTX 5090 Laptop，Windows 11）：

| 模型    | 数据  | H   | 种子 | 测试 MSE | 测试 MAE | 训练/验证/测试样本 | 用时    |
| ------- | ----- | --- | ---- | -------- | -------- | ------------------ | ------- |
| DLinear | ETTh1 | 96  | 2021 | 0.3962   | 0.4108   | 8449 / 2785 / 2785 | 767.5 s |

参数取自 TSLib `scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh:5`。训练在第 9 轮早停。
