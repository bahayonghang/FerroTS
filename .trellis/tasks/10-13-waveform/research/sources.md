# WaveForM 一手来源与实现风险

核验日期2026-10-13，以下事实用于规划，未运行该模型。

## 身份

- [AAAI 2023论文页](https://ojs.aaai.org/index.php/AAAI/article/view/26276)：WaveForM: Graph Enhanced Wavelet Learning for Long Sequence Forecasting of Multivariate Time Series。
- [论文PDF](https://ojs.aaai.org/index.php/AAAI/article/download/26276/26048)。
- [固定作者仓库](https://github.com/alanyoungCN/WaveForM/tree/6d307ffe77f2b626b9f4993613d8a234bb80ea72)。

## 已核对事实

| 事实 | 一手来源 | 规划影响 |
| --- | --- | --- |
| Haar多尺度分解、学习图、膨胀卷积、双向MixHop和IDWT | [模型](https://github.com/alanyoungCN/WaveForM/blob/6d307ffe77f2b626b9f4993613d8a234bb80ea72/models/WaveForM.py)、[GP](https://github.com/alanyoungCN/WaveForM/blob/6d307ffe77f2b626b9f4993613d8a234bb80ea72/models/GP.py) | 分模块CUDA前后向验证 |
| 论文70/20/10，代码70/10/20；scaler仅fit训练段 | 论文PDF与[loader](https://github.com/alanyoungCN/WaveForM/blob/6d307ffe77f2b626b9f4993613d8a234bb80ea72/data_provider/data_loader.py) | 同协议实现对齐与论文表复现分开 |
| loader各split drop_last，validation shuffle；入口未主动固定seed | [factory](https://github.com/alanyoungCN/WaveForM/blob/6d307ffe77f2b626b9f4993613d8a234bb80ea72/data_provider/data_factory.py)、[入口](https://github.com/alanyoungCN/WaveForM/blob/6d307ffe77f2b626b9f4993613d8a234bb80ea72/run.py) | 显式适配种子与全部评价窗口 |
| batch32/lr0.001/dropout0.30/Haar J3/top-k6/GNN3；默认最多100epoch/patience3 | [脚本](https://github.com/alanyoungCN/WaveForM/blob/6d307ffe77f2b626b9f4993613d8a234bb80ea72/run.sh)、入口 | 不套用P2的10epoch预算 |
| 当前完整tree无LICENSE，亦无README引用的requirements.txt | [固定完整tree](https://api.github.com/repos/alanyoungCN/WaveForM/git/trees/6d307ffe77f2b626b9f4993613d8a234bb80ea72?recursive=1) | 实施前先决定许可与依赖依据 |
| Candle0.11.0 ConvTranspose1D反向不支持 | [固定版本backprop](https://docs.rs/crate/candle-core/0.11.0/source/src/backprop.rs) | IDWT先验可导解析实现 |

作者模型的symmetric DWT/默认IDWT边界、DWT(L+H)-DWT(L)系数长度和无轴squeeze，要求单独覆盖奇数、非8倍数、水泥H15/30/60及B=1。不能以一次前向成功证明模型可训练。top-k图索引不可导，但选中边权必须保留到节点embedding的梯度。

## 本任务建议与未确认项

采用项目既有2021/2022/2023种子、全部验证/测试窗口与项目切分；ETT96+Weather24+水泥9共129次为规划提案，尚未执行。论文完整数据还包含Solar/Temperature，本轮不下载或声称全覆盖。实际依赖版本、合法参考执行方式和精确训练日程是步骤1–2待完成项。

候选PatchTST来源为[作者仓库](https://github.com/yuqinie98/PatchTST)/[论文](https://arxiv.org/abs/2211.14730)；TimeMixer来源为[作者仓库](https://github.com/kwuking/TimeMixer)/[论文](https://openreview.net/forum?id=7oLshfEIC2)。这两者仅为候选，仍需独立固定版本与任务规划，TimeMixer不等于TSMixer。
