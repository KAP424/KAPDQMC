# Born 测量的严格恒等式、上界与数涨落角量

这份研究备忘录区分三件事：已有的固定尺寸强测量展开；本次可严格推出的测量恒等式与界；仍然不能推出的 Rényi-2 角系数结论。以下数学推导不以 MC 拟合为前提，也不声称已经完成文献新颖性审查。用于检验的代码是同目录 `check_theory.py`，结果是 `theory_checks.json`。

## 1. 任意粒子数、任意固定输出数

设总格点数为 M，初始态是固定 n 粒子的归一化纯 Slater 态，其占据轨道矩阵 Q 满足 Q†Q=I，相关矩阵记为 G₀=QQ†。E 是受抑制的输出集合，固定其大小为 k；S 是 E 的补集。取

$$
t=e^{-2\beta},\qquad q=t^2,\qquad M_E=t^{N_E},\qquad
w_E=\langle\Psi|q^{N_E}|\Psi\rangle.
$$

在固定 n 粒子子空间内，完备和是标量：

$$
\sum_{|E|=k}M_E^\dagger M_E
=Z_{M,n,k}(q)I,
\qquad
Z_{M,n,k}(q)=\sum_j {n\choose j}{M-n\choose k-j}q^j.
$$

证明只需作用在任意 Fock 构型 F 上：在 F 中选 j 个格点、在补集中选 k−j 个格点组成 E，其数量与 F 的具体形状无关。因此条件 Born 分布是 w_E/Z；这不是任意修改过的采样权重。

令 P₀(F)=|⟨F|Ψ⟩|²，则精确的辅助联合分布为

$$
P(F,E)=\frac{P_0(F)q^{|E\cap F|}}{Z_{M,n,k}},\qquad
P(j)=\frac{{n\choose j}{M-n\choose k-j}q^j}{Z_{M,n,k}}.
$$

j 与 F 独立；给定 F、j 后，分别均匀选择 j 个 F 格点和 k−j 个补集格点。它提供精确独立抽样，也可用于下面的证明。**辅助 F 并未被物理测量；j 一般也不是观测到 E 后可读取的变量。计算量子熵仍须对原始 Ψ 施加 M_E。** 此独立抽样结构已经在旧 `BornSampler.jl` 中实现，不是本轮新发现。

## 2. 保持所有对角均值；总方差精确分解

对任意 Fock 基对角算符 A，所有 M_E 与 A 对易，故

$$
\mathbb E_E\langle A\rangle_E
=\frac1Z\sum_E\langle\Psi|M_E A M_E|\Psi\rangle
=\langle A\rangle_0.
$$

对任意空间区域 R，定义 X=N_R，初始均值 m 和量子数涨落 V₀，以及每个测量态的 m_E、V_E：

$$
m=\langle X\rangle_0,\quad V_0=\langle X^2\rangle_0-m^2,
\qquad m_E=\langle X\rangle_E,\quad V_E=\langle X^2\rangle_E-m_E^2.
$$

同时保持 X 和 X² 的平均便得到

$$
\boxed{\mathbb E_E V_E=V_0-\operatorname{Var}_E(m_E)\le V_0.}
$$

这也是辅助联合分布的全方差公式：测量态 Fock 概率恰为 P(F|E)，所以 m_E=E[X(F)|E]、V_E=Var[X(F)|E]。它适用于任意固定 k、n、任何零振幅构型，甚至不要求初态为 Gaussian。下面从数涨落推到 Rényi-2 熵则需要纯 Gaussian 初态和滤波后仍为 Gaussian。

## 3. Gaussian Rényi-2 熵的统一面积阶上界

测量后 Q_E 仍定义纯 Slater 投影。若区域相关矩阵的本征值为 ν，令 v=ν(1−ν)，则 0≤v≤1/4，单模二阶 Rényi 熵为 −ln(1−2v)。由 −ln(1−2v)/v 在此区间单调增加，有

$$
2v\le-\ln(1-2v)\le4\ln2\,v.
$$

求和及 Born 平均得到

$$
\boxed{
2\mathbb E_E V_E\le\mathbb E_E S_2(R;E)
\le4\ln2\,\mathbb E_E V_E
\le4\ln2\,V_0(R).}
$$

**若初态区域数涨落 V₀(R)=O(L)，则对所有 β，甚至任意随 L 变化的 β(L)，该条件 Born 协议的平均 S₂(R) 都是 O(L)；它不能产生平均熵的真正体积律。** 这是统一上界，不是小 β 展开。固定 k 的条件化没有破坏证明；也不需要以条件化后的 LOCC 熵单调性作为假设。

这里 O(L) 的前提必须在所研究的初态族中独立成立。二维 Dirac 基态满足的数涨落面积阶与有 Fermi 面时可能出现的 L ln L 必须区分。本界也适用于有微小质量的初态。

**单个区域的熵界不能直接对 A、B、C 相减。** 因此这一结论不能排除 D 为 O(L)，更不能证明 D 的对数系数等于基态。

## 4. 只使用可观测 E 的 MMSE 上界

上面的全方差右边是 Bayesian 后验均方误差。对任何仅依赖 E 的估计量 X̂(E)，均有

$$
\mathbb E_E V_E\le\mathbb E_{F,E}[X(F)-\widehat X(E)]^2.
$$

下面的估计量只使用 K=|E∩R| 或 |S∩R|，没有偷用不可观测的 j。

### 一般 n、k 的精确超几何矩

写 h=M−n，r=|R|。给定 X=x、j 后，K 是两个独立超几何计数之和：

$$
K=K_1+K_0,\qquad
K_1\sim\operatorname{Hypergeom}(n,x,j),\qquad
K_0\sim\operatorname{Hypergeom}(h,r-x,k-j).
$$

设 J=E[j]、W=Var(j)，并定义

$$
d=\frac Jn-\frac{k-J}{h},\qquad
u=\frac{\mathbb E[j(n-j)]}{n^2(n-1)},\qquad
v=\frac{\mathbb E[(k-j)(h-k+j)]}{h^2(h-1)}.
$$

以下显式公式取 n,h>1；等于 1 的退化超几何因子直接使用零方差。

$$
\begin{aligned}
\mathbb E K&=\frac{k-J}{h}r+dm,\\
\operatorname{Cov}(X,K)&=dV_0,\\
V_K=\operatorname{Var}(K)
&=u[m(n-m)-V_0]+v[(r-m)(h-r+m)-V_0]+d^2V_0\\
&\quad+W\left[\left(\frac1n+\frac1h\right)^2V_0
+\left(\left(\frac1n+\frac1h\right)m-\frac rh\right)^2\right].
\end{aligned}
$$

用可见的补集计数 T=r−K，最优线性估计量及误差为

$$
\widehat X_{\rm lin}(E)=m-\frac{dV_0}{V_K}(T-\mathbb E T),
\qquad
\boxed{\mathbb E_E V_E\le B_{\rm lin}=V_0-\frac{d^2V_0^2}{V_K}.}
$$

当 V_K=0 时 K 没有可用波动，取常数估计量 X̂=m，其界为 V₀。上式的减号来自 Cov(X,T)=−dV₀。

### 当前半填充、固定半输出协议

现在 M=2a、n=k=a，并先假定 m=r/2。令

$$
\eta=1-\frac{2J}{a},\qquad
b=\frac{\mathbb E[j(a-j)]}{a^2(a-1)},\qquad R_*=r(M-r).
$$

则

$$
\operatorname{Cov}(X,T)=\eta V_0,\qquad
V_K=[1-(4a-2)b]V_0+\frac b2 R_*,
\qquad B_{\rm lin}=V_0-\frac{\eta^2V_0^2}{V_K}.
$$

更直观的估计量 X̂=T=|S∩R| 给出完全显式的界：

$$
\boxed{\mathbb E_E V_E
\le B_{\rm count}
=b\left[\frac{R_*}{2}-2V_0\right]
+\frac{4\mathbb E[j^2]}{a^2}V_0.}
$$

最佳线性界不大于这个计数估计界。若 m≠r/2，计数界中两个 V₀ 替换为 V₀+(m−r/2)²；最佳线性界可直接使用一般公式。

固定 a 的 q→0 极限中，E[j]=E[j²]=a²q+O(q²)、b=q+O(q²)，所以

$$
B_{\rm count}=\frac q2\{r(M-r)+4V_0\}+O(q^2),\qquad
\mathbb E S_2(R)\le2\ln2\,q\{r(M-r)+4V_0\}+O(q^2).
$$

这个上界不要求任何 Fock minor 非零；因此也约束有节点时的强测量首项。它不是均匀大 L 展开。

## 5. 平均单体相关的精确退相干公式

两个 n 粒子 Fock 构型相差 d 次粒子交换时，对固定 k 的 Born 输出求平均得到的密度矩阵给它们之间的相干乘以

$$
\Lambda_d=\frac{[z^k](1+qz)^{n-d}(1+tz)^{2d}(1+z)^{M-n-d}}
{Z_{M,n,k}(q)}.
$$

因为 d=1 的两个构型共有 n−1 个占据格点、两个单独占据格点和 M−n−1 个共同空格点，生成函数的三个因子逐一给出此式。单体非对角算符只连接 d=1 的构型，因而不受初态振幅节点影响：

$$
\boxed{\mathbb E_E G_{E,ij}=\lambda_1G_{0,ij}\quad(i\ne j),\qquad
\mathbb E_E G_{E,ii}=G_{0,ii},}
$$

$$
\lambda_1=1-(1-t)^2\frac{Z_{M-2,n-1,k-1}(q)}{Z_{M,n,k}(q)}.
$$

这还给出与 M、n、k 无关的严格正下界。令共同多项式为 (1+qz)ⁿ⁻¹(1+z)ᴹ⁻ⁿ⁻¹，其 k、k−1、k−2 次系数为非负的 A、B、C，则

$$
\lambda_1=\frac{A+2tB+t^2C}{A+(1+t^2)B+t^2C}
\ge\frac{2t}{1+t^2}=\operatorname{sech}(2\beta).
$$

k=0 或 k=M 时滤波在固定粒子数扇区内是标量，直接取 λ=1。满占据和真空初态的空间熵恒为零，单独处理即可。

其证明还可直接使用生成函数差：

$$
(1+qz)(1+z)-(1+tz)^2=(1-t)^2z.
$$

半填充、固定半输出时，记 Zₐ=Σⱼ C(a,j)²t²ʲ，便得到

$$
\boxed{\lambda_a=1-(1-t)^2\frac{Z_{a-1}(t^2)}{Z_a(t^2)}.}
$$

### λ 的严格尺寸界和热力极限

矩序列表示为

$$
Z_a(t^2)=\frac1{2\pi}\int_0^{2\pi}|1+te^{i\phi}|^{2a}\,d\phi.
$$

非负随机变量的矩满足 Zₐ²≤Zₐ₋₁Zₐ₊₁，所以 Zₐ₋₁/Zₐ 随 a 不增；又有 |1+teⁱφ|²≤(1+t)²。因此

$$
\boxed{\operatorname{sech}(2\beta)=\lambda_1\le\lambda_a
\le\operatorname{sech}^2\beta,\qquad \lambda_a\text{ 随 }a\text{ 不减}.}
$$

左式 λ₁ 在本行专指 a=1 的值，勿与通用“单交换相干因子”的下标混淆。对固定有限 β（t>0），Laplace 展开给出

$$
Z_a(t^2)\sim\frac{(1+t)^{2a+1}}{2\sqrt{\pi at}},\qquad
\boxed{\lim_{a\to\infty}\lambda_a=\frac{4t}{(1+t)^2}=\operatorname{sech}^2\beta.}
$$

### RMS 关联与区域数涨落的下界

Jensen 不等式给出每对格点的严格界：

$$
\boxed{\mathbb E_E|G_{E,ij}|^2\ge\lambda_a^2|G_{0,ij}|^2\quad(i\ne j).}
$$

因纯 Slater 态满足 V_E(R)=Σᵢ∈R,ⱼ∉R |G_E,ij|²，故

$$
\boxed{\lambda_a^2V_0(R)\le\mathbb E_E V_E(R)\le V_0(R).}
$$

与第 3 节联立，得到任意固定 k 协议的区域熵界：

$$
\boxed{2\lambda_1^2V_0(R)\le\mathbb E_E S_2(R;E)\le4\ln2\,V_0(R).}
$$

以初态熵表达则为

$$
\boxed{\frac{\lambda_1^2}{2\ln2}S_{2,0}(R)
\le\mathbb E_E S_2(R;E)\le2\ln2\,S_{2,0}(R).}
$$

因此固定有限 β 时，任何非零的区域熵领先增长阶（例如 L、L ln L、L²）在这个条件 Born 平均下都保持 Θ 意义上的相同阶；前提是初态和输出态为纯、守粒子数 Gaussian 态。这不等于领先系数不变，更不等于二维面积项下面的角对数系数保持。

**增长阶保持本身并非 Born 特有：下一段给出更普遍的逐态滤波界。** Born 结构的附加价值是 β 无关的平均上界、更强的相干下界、总方差恒等式及数涨落角量的几何界。当 β 随 L 增大时，下界常数可以趋零，不能再援用固定有限 β 的 Θ 结论。

### 一般有界局域 Gaussian 滤波的逐态谱稳定性

对任意输出或任意对角单体滤波 D，若 tI≤D≤I，定义

$$
A=Q_R^\dagger D_R^2Q_R,\quad B=Q_{R^c}^\dagger D_{R^c}^2Q_{R^c},\qquad
A_0=Q_R^\dagger Q_R,\quad B_0=I-A_0.
$$

有 t²A₀≤A≤A₀、t²B₀≤B≤B₀。区域纠缠本征值的 odds r=ν/(1−ν) 是矩阵束 (A,B) 的广义本征值；Rayleigh 商及 min-max 原理给出有序谱的夹逼

$$
t^2r_{0,k}\le r_{D,k}\le t^{-2}r_{0,k},\qquad
|\epsilon_{D,k}-\epsilon_{0,k}|\le4\beta,\quad\epsilon=-\ln r.
$$

可逆的 D 保持 A、B 的秩，因此 ν=0、1 的零熵模保持其数目；奇异矩阵束可在共同有效支持或加正则后取极限处理。纠缠能 ε 的单模二阶 Rényi 熵为 s₂(ε)=ln(1+sech ε)，并且

$$
\left|\frac{d\ln s_2}{d\epsilon}\right|
=\frac{x|\tanh\epsilon|}{(1+x)\ln(1+x)}\le1,\qquad x=\operatorname{sech}\epsilon.
$$

故直接得到比借助数涨落更紧的逐态熵界：

$$
\boxed{e^{-4\beta}S_{2,0}(R)\le S_{2,D}(R)\le e^{4\beta}S_{2,0}(R).}
$$

这对强制选择的任何构型也成立，不需 Born 权重。它只控制单区域总熵和谱变化范围，仍然不能将含负权重的角相减组合直接夹逼。Born 平均可以进一步联合为

$$
\max\left(e^{-4\beta},\frac{\lambda_1^2}{2\ln2}\right)S_{2,0}
\le\mathbb E_E S_2
\le\min(e^{4\beta},2\ln2)S_{2,0}.
$$

若初态具有代数衰减的单体相关，则固定有限 β 下 Born 平均的相关模平方不可能在所有长距离都被纯指数截断。这个陈述针对 ensemble RMS，**不证明典型 trajectory 也有相同长程关联，更不确定 S₂ 角系数。** 稀有 trajectory 可以贡献平均值。

## 6. 数涨落角量具有非负的精确几何核

仍使用 L×L 原胞环面，所有区域包含被选原胞内的全部亚格点。取整数 w=pL，0<p≤1/2：A=w×L，C=L×w，B=w×(L−w)，定义

$$
D_V=(1-p)V(A)+pV(C)-V(B).
$$

对同一纯态，把这三个区域同时作全部 L² 次原胞平移，再平均。对有序格点对 i=(xᵢ,yᵢ,α)、j=(xⱼ,yⱼ,γ)，取 dₓ=(xⱼ−xᵢ) mod L、dᵧ 类似，定义

$$
h_p(d)=\frac{\min(d,L-d,w)}L.
$$

则有逐态恒等式

$$
\boxed{\overline D_V^{\,\rm full}(G)
=\sum_{i,j}|G_{ij}|^2h_p(d_x)h_p(d_y)\ge0.}
$$

证明：令 f_w(d) 是一维周期区间与其位移后的交叠格点数，则 f_{L−w}(d)=f_w(d)+L−2w。三区域跨界有序对计数的组合正好为

$$
w^2-w f_w(d_x)-w f_w(d_y)+f_w(d_x)f_w(d_y)
=[w-f_w(d_x)][w-f_w(d_y)].
$$

除以 L² 就是上面的核。它只依赖原胞成员关系，所以**同时覆盖四种 α、γ 亚格点配对；原矢是否正交、实空间亚格点偏移是多少，都不改变此离散恒等式。** 真正切开原胞或只选某一亚格点的区域不能直接套用该核。沿坐标轴的格点对至少一个 h 为零，说明相应直边局域项确实相消；斜向、长程格点对保留，形成角量。

若 p=1/m，使用 m² 个瓦片平移也有更强的逐态非负核。把原胞坐标按 w 分成 m 段、原点与区域一致，令 bₓ(x)=⌊x/w⌋，则

$$
\boxed{\overline D_V^{\,H}(G)=\frac1{m^2}\sum_{i,j}|G_{ij}|^2
\mathbf1[b_x(x_i)\ne b_x(x_j)]\mathbf1[b_y(y_i)\ne b_y(y_j)]\ge0.}
$$

它来自同一瓦片的两个点被宽 w 条带同时包含的概率 (1/m)1[same tile]，以及宽 L−w 条带是某个瓦片的补集。这里的两点计数包含所有原胞平移子群记录；单个原点 D_V 不必具有这个非负核。

将非负几何核与逐格点 Jensen 界相乘后求和，即得到

$$
\boxed{\mathbb E_E\overline D_{V,E}^{\,H\text{ 或 full}}
\ge\lambda_a^2\overline D_{V,0}^{\,H\text{ 或 full}}.}
$$

这个不等式甚至不要求初态平移对称。若初态及 Born 分布平移对称，平均原点观测、子群平均和全平移平均的期望相同。

**条件性的对数下界：** 若所选临界初态族满足 Dᵥ,₀(L)=cᵥ ln L+O(1)、cᵥ>0，则固定有限 β 时

$$
\boxed{\liminf_{L\to\infty}\frac{\mathbb E_E D_{V,E}(L)}{\ln L}
\ge\operatorname{sech}^4\beta\,c_V>0.}
$$

这允许对数以上的增长，**并非已证明纯 log scaling，更不是已证明系数与初态相同。** 当前 μ=10⁻⁸ 若固定后先取真正 L→∞，最终有质量截断；要把此式作为严格 Dirac 临界结论，需要定义 μ=0 的一致零模处置，或先取 μ→0、或指定 μL→0 的极限。有限数值尺寸上的近临界窗口不能替代这一极限说明。

对当前无自旋蜂窝模型的两 Dirac 锥，原矢平行四边形有两个 60°、两个 120° 角。已知临界数涨落角函数 u(θ)=1+(π−θ)cot θ、两锥电导 σ=1/8 给出

$$
c_{V,0}=\frac{\sigma}{\pi^2}\{2u(\pi/3)+2u(2\pi/3)\}
=\frac{4+2\pi/(3\sqrt3)}{8\pi^2}\simeq0.06597528.
$$

这属于已有 CFT 数涨落基线，见 [Cornering the universal shape of fluctuations, Eq. (9)](https://www.nature.com/articles/s41467-021-27727-1)，不是本轮的新发现。相应原文还指出 Dirac 情形更早已有推导。新 `compute_corner_baseline.py` 的 μ=0 一致零模选态、L≤1536 FFT 基线与这个值相符；数据见 `baseline/ground_corners.csv` 和 `baseline/summary.json`。它为上述条件下界提供现成的正系数，但没有把下界变成等号。

## 7. 不能把数涨落角下界直接换成 Rényi-2 角下界

本次先由实际随机搜索找到、再由有理数运算给出严格证书的反例是

$$
D_{S_2}^{\rm full}<2D_V^{\rm full}
$$

的纯 Gaussian 态。L=2、p=1/2、每原胞两个格点、半填充，用一个整数 8×4 矩阵 X 定义精确投影 G=X(XᵀX)⁻¹Xᵀ，对同一 G 做全部原胞平移后的结果是

$$
D_{S_2}=0.023970329722176893,\qquad D_V=0.012465217242688740,
\qquad D_{S_2}-2D_V=-0.000960104763200586.
$$

证书在 `exact_corner_counterexample.py` 与 `exact_corner_counterexample.json`。先用 Fraction 精确验证 G²=G，再把 Dₛ₂ 写为 ln ℛ/8；ℛ、Dᵥ 都是精确有理数。对 x=ℛ−1，证书直接验证 0<x<1 且 x−x²/2+x³/3<16Dᵥ；利用 ln(1+x)≤x−x²/2+x³/3 即严格证明负差，不依赖浮点误差条。该反例使用非均匀 Gaussian 态；它推翻的是对任意 Gaussian 态直接相减得到的普遍不等式，不声称它已经推翻某个指定蜂窝 Born ensemble 的更特殊定理。最值得进一步测量的是同一 E 下的 Dᵥ 与 Dₛ₂，不能用前者的证明替代后者。最早的浮点搜索态另保留在 `corner_inequality_counterexample.npz`，数值不同，不要与此有理证书混用。

## 8. 已有强测量展开与节点修正：可以补充界，不能算新发现

旧 `analysis/born_scaling_strong_measurement.py` 已含完整的固定尺寸 full-support 展开和 rank-one 节点修正。对满足 ⟨Nᵣ⟩=r/2 的区域，full-support 系数为（不要求每个单独格点都恰占据 1/2）

$$
C_R^{\rm bright}=r(M-r)+4V_0(R).
$$

若某输出 S 的 Q_S 恰缺一秩，令 W_S 是非零奇异值平方的乘积、u 是 S 中的空穴、v 是 E 中的粒子，并设 x、y 分别为它们落在 R 的概率。实际首项替换量为

$$
\Delta C_R(S)=W_S\{f(x)+f(y)-2[x+y-2xy]\},\qquad
f(x)=-\ln[x^2+(1-x)^2].
$$

缺秩至少二的输出只从 q² 或更高阶贡献。旧 L=3 数据已有

$$
C_D=16.2998418126967-1.66209042368616=14.6377513890106,
$$

而 β=3 的精确平均 D/q 为 14.6345686137172。此一致性不能被重新包装成新发现。

可附加的严格界是：对任意 x,y∈[0,1]，

$$
-2\le f(x)+f(y)-2[x+y-2xy]\le2\ln2-1.
$$

下界在 (x,y)=(0,1) 取等；上界在 (1/2,1/2) 取等。上界由 f(x)≤4 ln2 x(1−x) 及 x+y−2xy=x(1−x)+y(1−y)+(x−y)² 得到。修正不具有一般的固定符号。

更有整体意义的是第 4 节 MMSE 界蕴含的、对所有节点结构有效的首项界

$$
0\le C_R\le2\ln2\,C_R^{\rm bright}.
$$

它针对单区域熵系数，同样不能相减后当作 D 的界。对初始 Fock 乘积态，实际熵始终为零，节点修正恰好抵消所谓 bright 计数项，说明节点不能省略。

## 9. 完整局域协议与固定 k 条件协议的精确关系

完整二元局域 Kraus 算符可取

$$
K_0=\frac{\operatorname{diag}(1,\sqrt t)}{\sqrt{1+t}},\qquad
K_1=\frac{\operatorname{diag}(\sqrt t,1)}{\sqrt{1+t}}.
$$

令 E 是选 K₀ 的格点集合。在固定 n 粒子扇区内，该完整局域输出算符等于

$$
K_E=(1+t)^{-M/2}t^{(M-n-|E|)/2}M_E.
$$

因此

$$
P_{\rm local}(|E|=k)=\frac{t^{M-n-k}Z_{M,n,k}(t^2)}{(1+t)^M},
\qquad P_{\rm local}(E\mid |E|=k)=\frac{w_E}{Z_{M,n,k}}.
$$

输出数的边缘分布与固定 n 扇区内的初始态无关。任意 E 下的归一化测量态相同，但全局域平均是所有 k 条件平均的上述加权和，**不能默认等于 k=M/2 的条件平均。**

完整局域协议的单体相干因子恰为 sech²β；半填充固定半输出的 λₐ 不大于它，且固定有限 β、大 a 时趋于它。这严格比较了平均单体相干，不能据此排序两种协议的平均 S₂ 或角系数。对角均值、全方差与熵上界在两种协议中均成立。

## 10. 已执行检验和可检验命题

`check_theory.py` 已完成 24 组所有输出的精确枚举，包含 M=8,n=4 和 M=7,n=3，多个固定 k 与 β=0、0.3、1、3。分别直接检查输出概率归一化、平均相关矩阵、Nᵣ 与 Nᵣ²、全方差、超几何方差／协方差、可见补集计数估计误差以及最佳线性上界。

完整平移与瓦片子群几何检查包含 L=2、3、4、6，p=1/2、1/3；逐个有序格点对核与直接区域数涨落均一致，包含全部四个亚格点配对。最大数涨落收缩差为 3.34×10⁻¹⁶。另用 300 个复 Gaussian 态和随机有界局域滤波，检查了有序纠缠能移动范围及更紧逐态熵界。反例搜索第 12 个测试态就发现上节的负差，故未继续把“未找到反例”当证据。

本次最可靠的下一步不是再选择一个熵拟合窗口，而是在同一 Born 样本中同时保存以下量：每个区域 V_E、m_E，Dᵥ，Dₛ₂，以及按原胞位移分箱的 |G_E,ij|²。它们必须满足上述总方差、MMSE、λ 和非负核界。再比较 Dᵥ、Dₛ₂ 的局部斜率及两者随尺寸的比例，才有机会辨别 Rényi-2 角漂移是高阶谱信息、稀有 trajectory，还是共同的长程涨落交叉。

这些恒等式和界是可以写成严谨 lemma 的结构；发表价值还取决于它们在本协议中的新颖性核查、由此提出的有区分力的物理结论，以及足够稳定的后续数据。
