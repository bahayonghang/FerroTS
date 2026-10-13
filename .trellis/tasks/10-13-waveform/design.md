# WaveForM 设计

## 来源与首关

论文为[AAAI 2023 WaveForM](https://ojs.aaai.org/index.php/AAAI/article/view/26276)。固定作者仓库SHA为6d307ffe77f2b626b9f4993613d8a234bb80ea72，证据见research/sources.md。当前仓库无LICENSE、README引用的requirements.txt也不存在。首先确定可接受的实现/参考执行依据和依赖；无法确定则不进入源码复用及正式复现。论文独立实现与作者源码复用须区分，不自行为上游附许可。

## 模型边界

沿用ModelSpec/ForecastModel静态派发和现有TrainingSession。先实现Haar数学变换与边界规则，再共享图/膨胀卷积/MixHop及预测系数头，最后IDWT重建。只增模型自身需要的配置（固定可支持的wavelet、level、图top-k、模型宽度等），不先建通用小波插件或图框架。

Candle 0.11.0的ConvTranspose1D反向不支持；候选IDWT为可导Haar解析组合，先验证所有切片、拼接、交错、缩放路径的梯度和边界；若无法满足参考语义，再评审最小专用CUDA前后向算子。DWT采用symmetric边界，不能凭常见padding替代。预测系数长度由参考DWT(L+H)-DWT(L)语义确定，重建只取明确的未来H步，不引入未来真实值。

图构造学习节点embedding，top-k离散索引固定tie规则，选中边权保持梯度；图可为C×C，记录O(C²)成本。先以C=7/18/21验证公共和水泥场景；Traffic的862节点成本留给P4预算。B=1不得删除batch轴，尾批全覆盖。

## 协议与日程

论文切分70/20/10与官方loader70/10/20冲突。Rust和参考runner统一使用项目已冻结协议，明确这是同协议实现对齐；训练scaler只fit训练段。官方未主动固定seed且验证/测试drop_last，参考适配显式设置2021/2022/2023；验证/测试覆盖全部窗口、不shuffle，差异列入模型卡。不得把修改后的数值称论文原表复现。

参考脚本Haar J=3、batch32、lr0.001、dropout0.30、top-k6、3层GNN；run.py默认最多100epoch、patience3。来源关核对每个字段与调度后冻结配置，不擅自套P2的10epoch。水泥工程profile单独注明；先测1–3 CUDA epoch估计129次预算，再批准完整运行。

## 持久化与验收

沿用P2的权重/Adam/调度/RNG/游标合同；若实际模型有持久buffer，接入模型命名buffer合同，不另建持久化栈。新增随机site独立标识，验证不消耗训练随机流。模型输入只读历史，监督mask沿用数据/损失边界。

| AC | 机制 | 实施步骤 |
| --- | --- | --- |
| AC1 | 来源/许可/依赖与参考执行决定 | 1 |
| AC2 | 协议适配manifest、行边界/窗口计数比较 | 2 |
| AC3/AC4 | 可导Haar及图模块CUDA前后向fixture、边界矩阵 | 3–4 |
| AC5 | smoke与独立进程全状态恢复 | 5 |
| AC6 | ETT96+Weather24、40项同协议对齐门 | 6 |
| AC7 | 水泥9次、按变量名聚合及独立复算 | 7 |
| AC8 | 分阶段性能与最终身份/质量审阅 | 5–8 |

正式suite绑定所有数据/配置/来源/工具/二进制指纹。研究阶段未解决问题保留具体阻塞，不以额外CPU训练替代；也不阻塞可独立推进且已批准的其他任务。
