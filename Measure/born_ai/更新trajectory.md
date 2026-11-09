> 2026-10-27 检查说明：下文单 Slater 态的交换公式已通过直接行列式与长链更新核验。实际采样概率是权重除以固定大小集合上的归一化常数；它是固定正负输出数目的条件测量分布。实测接受率、构型概率、数值风险和独立抽样方法统一见 [测量概率与采样检查](SAMPLING_RESULTS.md)。开头辅助场表达式的推广不属于本次单 Slater 验证范围。

$$
\Psi=\sum_s P_s \Psi_s
\\
S_2 (\Psi) = -\ln \{\sum_{s_1,s_2}  P_{s_1}P_{s_2} \det[ G_{s_1}G_{s_2}+(I-G_{s_1})(I-G_{s_2}) ]  \}
\\
\Downarrow Measure \\
\text{以波恩概率}w(E)\text{概率坍缩到波函数} M_E \Psi
\\
\mathbb{E}_{b} S_2 = \sum_E w_E S_2(M_E \Psi)
$$

# trajectory 中测量构型的单点交换

下面只讨论固定粒子数的 Slater 态。目标是从所有含 \(a\) 个“占据目标”格点的构型中，用单点交换做重要性采样。

## 1. 统一记号

设总格点数为

$$
N_s=2a,
$$

初态为 \(a\) 粒子的 Slater 态

$$
|\Psi\rangle=|\Phi(P)\rangle,
\qquad P\in\mathbb C^{N_s\times a}.
$$

这里 \(P\) 的列是单粒子轨道；列之间不要求正交。定义初态的单粒子密度矩阵

$$
G\equiv P(P^\dagger P)^{-1}P^\dagger,
\qquad
G_{rt}=\frac{\langle\Psi|\hat c_t^\dagger\hat c_r|\Psi\rangle}
              {\langle\Psi|\Psi\rangle}.
$$

它满足

$$
G^\dagger=G,
\qquad G^2=G,
\qquad \operatorname{tr}G=a.
$$

把目标构型写成两个有序数组

$$
S=(s_1,s_2,\ldots,s_a),\qquad
E=(e_1,e_2,\ldots,e_a),
$$

其中

$$
s_i=S[i]\quad\text{是目标占据格点},\qquad
e_j=E[j]\quad\text{是目标空格点}.
$$

这里 \(i,j=1,\ldots,a\) 只表示数组位置；实际交换的是 \(S[i]\) 和 \(E[j]\)。以后 \(k,l\) 表示 \(E\) 中的位置，\(r,t\) 表示物理格点。

## 2. 先看计算逻辑

### 2.1 测量算符和构型权重

记

$$
\hat N_S=\sum_{r\in S}\hat n_r,
\qquad
\hat N_E=\sum_{r\in E}\hat n_r.
$$

原来的坍缩算符可以写成

$$
M_{S,E}^{\rm full}
=\exp\!\left[\beta(\hat N_S-\hat N_E-a)\right].
$$

我们始终在 \(N_{\mathrm{part}}=a\) 的粒子数扇区内，因此

$$
\hat N_S+\hat N_E=a,
$$

后面统一使用

$$
\boxed{M_E=e^{-2\beta\hat N_E}}.
$$

有限 \(\beta\) 时这是一个滤波器。定义

$$
w(E)\equiv
\frac{\langle\Psi|M_E^\dagger M_E|\Psi\rangle}
     {\langle\Psi|\Psi\rangle},
$$

它是重要性采样的权重；所有 \(E\) 的权重之和一般不是 \(1\)，但 Metropolis 只需要权重之比。

这里固定大小集合上的真实概率及归一化为

$$
p_a(E)=\frac{w(E)}{Z_a(\beta)},\qquad
Z_a(\beta)=\sum_{j=0}^a\binom aj^2e^{-4\beta j}.
$$

因此开头的 Born 平均中，若使用本文固定大小的构型空间，应以这个归一化后的概率为权重。以下更新仍只需未归一化权重之比。令

$$
\lambda\equiv1-e^{-4\beta},
\qquad 0\leq\lambda<1.
$$

### 2.2 当前构型需要保存什么

取 \(E\) 的主子块

$$
(G_E)_{kl}=G_{e_ke_l},
$$

并定义

$$
W_E\equiv I_a-\lambda G_E,
\qquad
X\equiv W_E^{-1}.
$$

当前构型的权重为

$$
w(E)=\det W_E.
$$

因此采样时只需维护当前的 \(E\) 和 \(X\)。

### 2.3 一次单点交换怎样计算

随机选取数组位置

$$
i\in\{1,\ldots,a\},\qquad
j\in\{1,\ldots,a\},
$$

交换物理格点 \(S[i]=s_i\) 和 \(E[j]=e_j\)：

$$
S'=S\text{，但 }S'[i]=e_j,
\qquad
E'=E\text{，但 }E'[j]=s_i.
$$

令

$$
\widehat E=E\setminus\{e_j\},
$$

并保持剩余元素的原顺序。带下标 \(\widehat E\) 的矩阵都表示按这个顺序取出的主子块。

先计算

$$
g_k=G_{e_ks_i},
\qquad
z_k=\sum_{l=1}^aX_{kl}g_l.
$$

然后对 \(k\neq j\) 计算

$$
\widetilde z_k
=z_k-\frac{X_{kj}}{X_{jj}}z_j.
$$

定义

$$
\Delta_{ij}
=1-\lambda G_{s_is_i}
 -\lambda^2\sum_{k\neq j}G_{s_ie_k}\widetilde z_k.
$$

权重比为

$$
\boxed{\mathrm{ratio}
=\frac{w(E')}{w(E)}
=\Delta_{ij}X_{jj}}.
$$

如果 \(S[i]\) 和 \(E[j]\) 分别均匀选取，正、反提议概率相同，所以接受率为

$$
\boxed{\alpha=\min(1,\mathrm{ratio})}.
$$

若接受，仍把新格点 \(s_i\) 放在 \(E\) 的第 \(j\) 个位置。对 \(k,l\neq j\)，

$$
\boxed{
X'_{kl}
=X_{kl}-\frac{X_{kj}X_{jl}}{X_{jj}}
 +\frac{\lambda^2}{\Delta_{ij}}
   \widetilde z_k\widetilde z_l^*}
$$

并且

$$
\boxed{
X'_{kj}=\frac{\lambda\widetilde z_k}{\Delta_{ij}},
\qquad
X'_{jk}=\frac{\lambda\widetilde z_k^*}{\Delta_{ij}},
\qquad
X'_{jj}=\frac1{\Delta_{ij}}.}
$$

实矩阵时可以去掉复共轭。一次提议只需要一次 \(Xg\)，因此计算量为 \(O(a^2)\)；接受后的更新也是 \(O(a^2)\)。

用于检查实现的等价形式是

$$
\boxed{
\mathrm{ratio}
=\left(1-\lambda G_{s_is_i}-\lambda^2g^\dagger Xg\right)X_{jj}
 +\lambda^2|z_j|^2.}
$$

实际采样使用 \(\Delta_{ij}X_{jj}\) 更简洁，因为它和逆矩阵更新共用 \(\widetilde z\)。

## 3. 核心公式的证明

### 3.1 单粒子算符作用在 Slater 态上

令

$$
\widehat A=\hat c^\dagger T\hat c
=\sum_{rt}\hat c_r^\dagger T_{rt}\hat c_t.
$$

用 \(u\) 表示单粒子基底指标。利用反对易关系，

$$
[\widehat A,\hat c_u^\dagger]
=\sum_r\hat c_r^\dagger T_{ru}.
$$

重复使用这个关系并求 Baker--Campbell--Hausdorff 级数，得到

$$
e^{\widehat A}\hat c_u^\dagger e^{-\widehat A}
=\sum_r\hat c_r^\dagger(e^T)_{ru}.
$$

写成

$$
|\Phi(P)\rangle
=\prod_{m=1}^a
  \left(\sum_rP_{rm}\hat c_r^\dagger\right)|0\rangle,
$$

便有

$$
\boxed{e^{\hat c^\dagger T\hat c}|\Phi(P)\rangle
=|\Phi(e^TP)\rangle.}
$$

对本题，令 \(D_E\) 为 \(E\) 上的单粒子投影：

$$
(D_E)_{rt}
=\begin{cases}
1,&r=t\in E,\\
0,&\text{其他情况}.
\end{cases}
$$

于是

$$
M_E=e^{\hat c^\dagger(-2\beta D_E)\hat c},
\qquad
B_E=e^{-2\beta D_E},
$$

从而

$$
M_E|\Phi(P)\rangle=|\Phi(B_EP)\rangle.
$$

### 3.2 权重的行列式形式

Slater 态的交叠满足

$$
\langle\Phi(Q)|\Phi(P)\rangle=\det(Q^\dagger P).
$$

这是因为把 bra、ket 中的 \(a\) 个线性组合逐个反对易后，每个配对给出 \(Q^\dagger P\) 的一个矩阵元，所有排列的符号和正好组成行列式。

因此

$$
w(E)
=\frac{\det\!\left(P^\dagger B_E^2P\right)}
       {\det(P^\dagger P)}.
$$

由于

$$
B_E^2=e^{-4\beta D_E}=I_{N_s}-\lambda D_E,
$$

有

$$
w(E)
=\frac{\det\!\left(P^\dagger(I-\lambda D_E)P\right)}
       {\det(P^\dagger P)}.
$$

用 Sylvester 行列式恒等式 \(\det(I-AB)=\det(I-BA)\)，并代入 \(G=P(P^\dagger P)^{-1}P^\dagger\)，得到

$$
w(E)=\det(I_{N_s}-\lambda D_EG).
$$

把行列和列同时按 \((E,S)\) 重排，矩阵成为块上三角形式：

$$
I_{N_s}-\lambda D_EG
\sim
\begin{pmatrix}
I_a-\lambda G_E & -\lambda G_{ES}\\
0&I_a
\end{pmatrix}.
$$

所以

$$
\boxed{w(E)=\det(I_a-\lambda G_E)=\det W_E.}
$$

由于 \(G_E\) 的本征值属于 \([0,1]\)，且 \(0\leq\lambda<1\)，\(W_E\) 正定，权重没有符号问题。

### 3.3 单点交换的比值

把当前矩阵 \(W_E\) 按 \((\widehat E,e_j)\) 排列，记

$$
A=I_{\widehat E}-\lambda G_{\widehat E},
\qquad
b=W[\widehat E,e_j],
\qquad
d=W_{e_je_j}.
$$

于是

$$
W_E=\begin{pmatrix}A&b\\b^\dagger&d\end{pmatrix}.
$$

定义旧构型的 Schur 补

$$
\sigma=d-b^\dagger A^{-1}b.
$$

分块行列式公式给出

$$
\det W_E=\det A\,\sigma.
$$

另一方面，\(X=W_E^{-1}\) 的第 \(j,j\) 个元素满足

$$
X_{jj}=\sigma^{-1}.
$$

提议构型只把第 \(j\) 个格点从 \(e_j\) 换成 \(s_i\)，所以它与当前构型共享同一个 \(\widehat E\) 子块：

$$
W_{E'}=
\begin{pmatrix}A&y\\y^\dagger&d'\end{pmatrix},
$$

其中

$$
y=W[\widehat E,s_i]=-\lambda G[\widehat E,s_i],
\qquad
d'=W_{s_is_i}=1-\lambda G_{s_is_i}.
$$

新矩阵的 Schur 补为

$$
\Delta_{ij}=d'-y^\dagger A^{-1}y.
$$

于是

$$
\det W_{E'}=\det A\,\Delta_{ij}.
$$

两式相除，得到

$$
\boxed{\frac{w(E')}{w(E)}=\Delta_{ij}X_{jj}.}
$$

现在把 \(A^{-1}\) 写成当前 \(X\) 的量。分块求逆给出

$$
X_{\widehat E\widehat E}
=A^{-1}+A^{-1}b\,\sigma^{-1}b^\dagger A^{-1},
\qquad
X_{\widehat E,j}=-A^{-1}b\,\sigma^{-1}.
$$

若 \(g_k=G_{e_ks_i}\)、\(z=Xg\)，则

$$
\left(z_k-\frac{X_{kj}}{X_{jj}}z_j\right)_{k\neq j}
=A^{-1}G[\widehat E,s_i].
$$

这正是前面定义的 \(\widetilde z\)。因为 \(y=-\lambda G[\widehat E,s_i]\)，

$$
\Delta_{ij}
=1-\lambda G_{s_is_i}
 -\lambda^2G[s_i,\widehat E]A^{-1}G[\widehat E,s_i]
=1-\lambda G_{s_is_i}
 -\lambda^2\sum_{k\neq j}G_{s_ie_k}\widetilde z_k.
$$

因此 ratio 的计算只需要当前的 \(X\) 和 \(G\)。

### 3.4 直接形式的来由

如果先把 \(s_i\) 加入 \(E\)，再删去 \(e_j\)，令

$$
T=E\cup\{s_i\}.
$$

对 \(W_T\) 先按 \(E\) 与 \(s_i\) 分块。其 Schur 补为

$$
\tau_{ij}
=W_{s_is_i}-W[s_i,E]XW[E,s_i]
=1-\lambda G_{s_is_i}-\lambda^2g^\dagger Xg.
$$

于是

$$
\det W_T=\det W_E\,\tau_{ij}.
$$

余子式公式给出

$$
\det W_{E'}
=\det W_T\,(W_T^{-1})_{e_je_j}.
$$

分块求逆的 \(e_j,e_j\) 元为

$$
(W_T^{-1})_{e_je_j}
=X_{jj}+\frac{\lambda^2|z_j|^2}{\tau_{ij}}.
$$

因此

$$
\frac{w(E')}{w(E)}
=\left(1-\lambda G_{s_is_i}-\lambda^2g^\dagger Xg\right)X_{jj}
 +\lambda^2|z_j|^2,
$$

它与 \(\Delta_{ij}X_{jj}\) 的等价性可以直接写出来。记

$$
\widehat X
=X_{\widehat E\widehat E}
 -\frac{X_{\widehat E,j}X_{j,\widehat E}}{X_{jj}}
=A^{-1},
$$

并把 \(g\) 按 \((\widehat E,E[j])\) 分成

$$
g=\begin{pmatrix}g_{\widehat E}\\g_j\end{pmatrix},
\qquad
z_j=X_{j,\widehat E}g_{\widehat E}+X_{jj}g_j.
$$

由上面 \(\widehat X\) 的定义，直接展开二次型可得

$$
g^\dagger Xg
=g_{\widehat E}^\dagger\widehat Xg_{\widehat E}
 +\frac{|z_j|^2}{X_{jj}}.
$$

因此

$$
\Delta_{ij}
=1-\lambda G_{s_is_i}
 -\lambda^2g_{\widehat E}^\dagger\widehat Xg_{\widehat E}
$$

满足

$$
\Delta_{ij}
=\left(1-\lambda G_{s_is_i}-\lambda^2g^\dagger Xg\right)
 +\frac{\lambda^2|z_j|^2}{X_{jj}}.
$$

两边乘以 \(X_{jj}\)，就得到

$$
\boxed{
\Delta_{ij}X_{jj}
=\left(1-\lambda G_{s_is_i}-\lambda^2g^\dagger Xg\right)X_{jj}
 +\lambda^2|z_j|^2.}
$$

所以两个式子不是两种不同的 ratio，而是同一个行列式比值的两种计算方式。

### 3.5 逆矩阵的增量更新

先删去第 \(j\) 行和第 \(j\) 列。由分块逆矩阵公式，

$$
\widehat X\equiv A^{-1}
=X_{\widehat E\widehat E}
 -\frac{X_{\widehat E,j}X_{j,\widehat E}}{X_{jj}}.
$$

再把新格点 \(s_i\) 插回第 \(j\) 个位置：

$$
W_{E'}=
\begin{pmatrix}
A&-\lambda G[\widehat E,s_i]\\
-\lambda G[s_i,\widehat E]&1-\lambda G_{s_is_i}
\end{pmatrix}.
$$

其 Schur 补就是 \(\Delta_{ij}\)，所以

$$
X'_{\widehat E\widehat E}
=\widehat X
 +\frac{\lambda^2}{\Delta_{ij}}
  \widetilde z\,\widetilde z^\dagger,
$$

$$
X'_{\widehat E,j}=\frac{\lambda}{\Delta_{ij}}\widetilde z,
\qquad
X'_{j,\widehat E}=\frac{\lambda}{\Delta_{ij}}\widetilde z^\dagger,
\qquad
X'_{jj}=\frac1{\Delta_{ij}}.
$$

把 \(\widehat X\) 换回旧 \(X\)，就是第 2.3 节列出的逐元素更新式。整个更新没有重新求逆。
