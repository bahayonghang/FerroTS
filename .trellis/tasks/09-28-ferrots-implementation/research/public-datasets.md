# 公共数据集来源与校验（2026-09-28）

## 来源

- 用户已执行 `hf download thuml/Time-Series-Library --repo-type=dataset`。
- 本地缓存：`C:\Users\lyh\.cache\huggingface\hub\datasets--thuml--Time-Series-Library`（默认 `HF_HUB_CACHE`；`HF_HOME`、`HF_HUB_CACHE` 均未设置）。
- 快照 revision：`2b66e59ee19dac8f6f19fb5d4997f289fdfea357`（`refs/main`）。
- 缓存文件为指向 `blobs/` 的符号链接；使用 `cp -L` 解引用拷贝到 `data/`，保持上游目录名与文件名。
- 只拷贝长期预测数据集。未拷贝：分类数据（`.ts`，如 EthanolConcentration、Heartbeat）、异常检测数据（MSL、SMAP、SMD、PSM、SWaT）、短期预测 M4。

## 文件清单

| 路径                                   | 字节数      | 行数（不含表头） | 列数（含 date） | 起止时间                            | SHA256                                                             |
| -------------------------------------- | ----------- | ---------------- | --------------- | ----------------------------------- | ------------------------------------------------------------------ |
| `data/ETT-small/ETTh1.csv`             | 2,589,657   | 17,420           | 8               | 2016-07-01 00:00 – 2018-06-26 19:00 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| `data/ETT-small/ETTh2.csv`             | 2,417,960   | 17,420           | 8               | 同上                                | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| `data/ETT-small/ETTm1.csv`             | 10,360,719  | 69,680           | 8               | 2016-07-01 00:00 – 2018-06-26 19:45 | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` |
| `data/ETT-small/ETTm2.csv`             | 9,677,236   | 69,680           | 8               | 同上                                | `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd` |
| `data/electricity/electricity.csv`     | 95,581,762  | 26,304           | 322             | 2016-07-01 02:00 – 2019-07-02 01:00 | `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad` |
| `data/exchange_rate/exchange_rate.csv` | 637,800     | 7,587            | 9               | 1990/1/1 0:00 – 2010/10/10 0:00     | `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842` |
| `data/illness/national_illness.csv`    | 67,620      | 966              | 8               | 2002-01-01 – 2020-06-30             | `93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a` |
| `data/traffic/traffic.csv`             | 136,478,119 | 17,544           | 863             | 2016-07-01 02:00 – 2018-07-02 01:00 | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` |
| `data/weather/weather.csv`             | 7,235,425   | 52,696           | 22              | 2020-01-01 00:10 – 2021-01-01 00:00 | `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63` |

拷贝后逐文件比对 SHA256，与缓存源文件全部一致。

## 对实现的影响

- ETTh 行数 17,420 = 12/4/4 个月切分所需 8,640 + 2,880 + 2,880 = 14,400 行之外还有 3,020 行；TSLib 的 ETT 切分只使用前 14,400 行（ETTm 为 57,600 行）。P1 窗口数测试必须复现该行为。
- `exchange_rate.csv` 时间格式为 `1990/1/1 0:00`（非零填充的 `/` 分隔），其他文件为 `YYYY-MM-DD HH:MM:SS`。P1 的时间列解析必须支持可配置格式。
- `weather.csv` 列名含空格、括号与单位（如 `p (mbar)`）；`illness` 列名含 `%` 与空格。DataSpec 列名按原样匹配。
