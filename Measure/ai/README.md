# 构型初态、虚时演化与弱测量实验

本目录只保留四份 Markdown，按需要阅读：

| 文件 | 内容 |
|---|---|
| [RESULTS.md](RESULTS.md) | 全部结果与图、时间窗口校准、数据与复现命令 |
| [SCALING_GUIDE.md](SCALING_GUIDE.md) | 全部构型图、投影与对称性判据、scaling 判断方法、数值公式 |
| [neq.md](neq.md) | 原始任务，保持原文 |
| 本页 | 文件导航和基本约定 |

采用周期 HoneyComb120、实数等强最近邻 hopping、半填充、单种自由费米子的自然对数 Rényi-2 熵。“平衡”指基态准备后测量，“非平衡”指有限虚时演化后测量；改变初态也改变自身空位测量集合。

**能投影到基态、有限窗口拟合稳定、渐近角系数相等是三个不同问题。** 新的 [β=0 初态专项](RESULTS.md#beta0-study) 比较 16 个候选，并对其中 10 个补齐统一的大尺寸时间确认网格；[β=0 初态判断与对称性](SCALING_GUIDE.md#beta0-initial-state) 解释为何可达性还不够，以及哪些互补构型具有严格相同的熵。

此前 [三类构型的 β 扫描](RESULTS.md#results-occupancy-classes) 集中展示九个可达例；[第二类追加四构型](RESULTS.md#occupancy-second-class-extra) 包括两个新可达 4×4 构型，以及条纹交替、棋盘交替的有限时间与可达激发态极限熵。受阻交替态的参数不能称为基态角系数。[3×3 混合构型归第三类](RESULTS.md#occupancy-near02-irregular3)，[全部六种 2×2 的受阻证明](RESULTS.md#occupancy-pure02-m2-blocked) 单列。[irregular4 专项复核](RESULTS.md#irregular-followup) 检查异常曲率、扩大尺寸外推和计算实现。原有密集 β 扫描见 [uniform A](RESULTS.md#results-uniform-a)；全部构型图及判断方法见 [指南](SCALING_GUIDE.md)。

代码、原始数据、拟合和图保留原位。[复现命令与数据索引](RESULTS.md#reproduce) 集中维护，[旧文稿压缩归档](data/documentation_archive_20261019.zip) 只供历史追溯。仅更新结果页可在仓库根目录运行：

```powershell
python src/Measure/ai/write_results_report.py
```
