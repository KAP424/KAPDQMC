# KAPDQMC

[![Build Status](https://github.com/KAP424/KAPDQMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KAP424/KAPDQMC.jl/actions/workflows/CI.yml?query=branch%3Amain)

KAPDQMC 是一个用 Julia 编写的行列式量子蒙特卡洛（DQMC）研究代码库，采用辅助场采样和虚时间投影，计算相互作用费米子模型的物理量、二阶 Rényi 纠缠熵及区域熵差。部分模型支持相互作用参数随虚时间变化的淬火过程。

## 主要功能

- **模型模块**：`tU`、`tV`、`tUV`、`SO3`、`VBS` 和 `M_VBS`，通过各自的 `*_Hubbard_Para` 构造模型参数。
- **物理量采样**：`Initial_s` 初始化辅助场，`phy_update` 执行扫描并测量能量、关联函数等；具体观测量依模型而定。
- **纠缠量的增量算法**：`ctrl_SCEEicr` 计算两个区域的二阶 Rényi 熵差；`tU` 还提供 `ctrl_EEicr`，`tU`、`tV` 提供无序算符增量接口 `ctrl_SCDOPicr`。
- **带测量算符的 tU 模型**：`phy_update_M` 和 `ctrl_SCEEicr_M` 接收测量格点 `Eidx` 与强度 `beta`；`beta=0` 时测量算符退化为单位算符。
- **公共数值工具**：晶格与区域索引、试探态构造、Green 矩阵、Grover 矩阵，以及基于 BLAS/LAPACK 的矩阵更新与稳定化。

## 代码结构

- [`src/KAPDQMC.jl`](src/KAPDQMC.jl)：包入口与公共接口。
- [`src/public/`](src/public/)：几何、缓冲区和矩阵运算工具。
- `src/tU/`、`src/tV/`、`src/tUV/`、`src/SO3/`、`src/VBS/`、`src/M_VBS/`：各模型的参数、Green 矩阵、采样与纠缠量实现。
- [`test/`](test/)：测试与模拟示例；带测量算符的 tU 示例见 [`test/tU_Measure/`](test/tU_Measure/)。
- [`DP_Lib.py`](DP_Lib.py)：CSV 数据处理、误差估计与绘图工具。

## 基本使用

项目使用 Julia 1.12。在仓库根目录安装依赖并启动项目环境：

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=.
```

以蜂窝晶格上的 tU 模型为例：

```julia
using KAPDQMC, Random

model = tU_Hubbard_Para(
    Ht=1.0, Hu1=3.8, Hu2=3.8,
    Θrelax=5.0, Θquench=0.0,
    Lattice="HoneyComb120", site=[3, 3],
    Δt=0.1, BatchSize=10, Initial="V",
)
s = Initial_s(model, MersenneTwister(1234))
path = "output/tU"
mkpath(path)
phy_update(path, model, s, 100, false)  # 热化示例，不记录数据
phy_update(path, model, s, 100, true)   # 记录物理量到 CSV
```

示例中的 sweep 数仅用于演示，实际计算需要检查热化、统计误差与投影长度收敛性。采样函数会原地更新辅助场，记录数据时采用追加写入。

纠缠增量计算需分别运行 `λ = 0, 1/Nλ, …, (Nλ-1)/Nλ`。`DP_Lib.py` 中的 `File_Process2` 按 λ 汇总观测量，并返回 `-log(∏λ ⟨Oλ⟩)` 及误差估计；用于 `ctrl_SCEEicr` 输出时，该量是区域熵差。

## 待完善
