
$$
|\Psi_M\rangle=M|\Psi\rangle
$$

$\Psi$ 为初始波函数，$\Psi_M$ 为测量后坍缩的波函数，$M$ 为测量坍缩算符

对于粒子数测量：$\{n_i=0/1\}$，总格点数为 $N_s$，总粒子数为 $a=N_s/2$

如果我要坍缩到测量结果为

$$
S=\mathrm{nidx1}=\{i:n_i=1\},\qquad
E=\mathrm{nidx2}=\{i:n_i=0\},
$$

其中 $S$ 为占据集合，$E$ 为空格点集合。令

$$
\hat N_S=\sum_{i\in S}\hat n_i,\qquad \hat N_R=\sum_{i\in E}\hat n_i, \qquad N_s=2a.
$$

目标构型满足

$$
N_E=0,\qquad N_S=a.
$$

适合目标构型的测量坍缩算符应写成

$$
\boxed{ M_\beta= \exp\left\{ \beta\left( \hat N_S-\hat N_E-a \right) \right\} }
$$

其中 $a$ 表示 $a\hat I$。对于目标构型，指数为零，因此 $M_\beta$ 的本征值为 $1$；对于其他构型，指数严格小于零。

所以在完整 Fock 空间中，它是一个标量乘以粒子数守恒的高斯算符。

在固定粒子数 $N_p=a$ 的子空间内，

$$
\hat N_S=a-\hat N_E,
$$

因此

$$
\boxed{ M_\beta\big|_{N_p=a} = e^{-2\beta\hat N_E} }
$$

其中 $P$ 是 $N_s\times a$ 的轨道矩阵。引入单粒子空间中的投影矩阵

$$
D_L= \begin{pmatrix} [I_a]_{E\times E} &0\\ 0& [0]_{S\times S} \end{pmatrix}.
$$

那么

$$
M_\beta = e^{\hat c^\dagger T_\beta\hat c}, \qquad T_\beta=-2\beta D_L.
$$

对应的单粒子传播矩阵为

$$
B_\beta=e^{T_\beta} = \begin{pmatrix} [e^{-2\beta}I_a]_{E\times E}&0\\ 0&[I_a]_{S\times S} \end{pmatrix}.
$$

因此：

$$
\boxed{ |\Psi\rangle\to M_\beta|\Psi\rangle = |B_\beta \Psi\rangle }
$$

同时坍缩到这个结果的概率为：

$$
P(\{S,E\})=\frac{\langle\Psi|M_\beta M_\beta |\Psi\rangle}{\langle\Psi|\Psi\rangle}
$$



-----------

现在我要计算更新的测量结果，在格点很大的体系里，穷举遍历 $\{S,E\}$ 是很困难的，于是我想要用马尔科夫链重要性采样去遍历。比如现在我提议一个单点更新，分别随机选取 S 和 E 中的一个格点 index i 和 j 进行交换。

那么测量坍缩算符 $M_\beta \to M^\prime_\beta$ ，从而坍缩概率也会改变 $P(\{S,E\}) \to P(\{S_{i\to j},E_{j\to i}\})$。于是接受这个提议的概率为：
$$
ratio=\frac{P(\{S_{i\to j},E_{j\to i}\})}{P(\{S,E\})}
$$
你帮我进一步详细推导这个 ratio 的更新表达式。

---

# 附：单点交换 Metropolis 比值的完整推导

## 0. 记号与约定

约定轨道矩阵 $P$ 为**实**矩阵（DQMC 中的常见情形），于是 $G=G^{\mathsf T}$、$W=W^{\mathsf T}$。

占据集合 $S=\{i:n_i=1\}$，空格点集合 $E=\{i:n_i=0\}$，$|S|=|E|=a$，$N_s=2a$。

**等时单粒子格林函数**（初始态 $|\Psi\rangle=|\Phi(P)\rangle$ 的单粒子密度矩阵）：

$$
\boxed{\;G\equiv P(P^{\dagger}P)^{-1}P^{\dagger},\qquad
G_{ij}=\frac{\langle\Psi|\hat c_i^{\dagger}\hat c_j|\Psi\rangle}{\langle\Psi|\Psi\rangle}\;}
$$

它是**占据轨道空间上的投影算符**：$G^2=G$，$G^{\dagger}=G$，$\operatorname{tr}G=a$。

定义两个常数：

$$
\boxed{\;\lambda\equiv 1-e^{-4\beta}\in(0,1)\;},\qquad
\boxed{\;\eta\equiv e^{4\beta}-1=\frac{\lambda}{1-\lambda}\;}
$$

简记：$G_E\equiv G\big|_{E\times E}$（取 $E$ 指标的子块），$W\equiv I_{N_s}-\lambda G$，$E=\{e_1,e_2,\dots,e_a\}$。

**关于 $M_\beta$ 的两种写法的一致性。** 前面写过 $M_\beta=\exp\{\beta(\hat N_S-\hat N_E-a)\}$，也写过 $M_\beta=e^{\hat c^{\dagger}T_\beta\hat c}$（$T_\beta=-2\beta D_L$）。二者仅在 $N_p=a$ 子空间内相等：那里 $\hat N_S=a-\hat N_E$，故 $\hat N_S-\hat N_E-a=-2\hat N_E$。由于初态与所有坍缩后态都严格在 $N_p=a$ 子空间内，下文一律取

$$
M_\beta=e^{-2\beta\hat N_E}\Big|_{N_p=a},\qquad
M_\beta^{\dagger}M_\beta=e^{-4\beta\hat N_E}=e^{\hat c^{\dagger}(2T_\beta)\hat c}.
$$

## 1. 两个引理

### 引理 1（单粒子算符的伴随作用）

$$
e^{\hat c^{\dagger}T\hat c}\,\hat c_i^{\dagger}\,e^{-\hat c^{\dagger}T\hat c}
=\sum_j \hat c_j^{\dagger}\,(e^{T})_{ji}
\qquad\Longleftrightarrow\qquad
\hat c^{\dagger}\;\longrightarrow\;\hat c^{\dagger}e^{T}.
$$

**证明.** 先算一次对易子：
$$
\Big[\hat c^{\dagger}T\hat c,\;\hat c_i^{\dagger}\Big]
=\sum_{kl}T_{kl}\Big[\hat c_k^{\dagger}\hat c_l,\hat c_i^{\dagger}\Big]
=\sum_{kl}T_{kl}\,\hat c_k^{\dagger}\underbrace{\{\hat c_l,\hat c_i^{\dagger}\}}_{\delta_{li}}
=\sum_k \hat c_k^{\dagger}T_{ki}.
$$
即 $\mathrm{ad}_A\hat c_i^{\dagger}=\sum_k\hat c_k^{\dagger}T_{ki}$，于是 $\mathrm{ad}_A^{\,n}\hat c_i^{\dagger}=\sum_k\hat c_k^{\dagger}(T^n)_{ki}$（对 $n$ 归纳即得）。代入 Baker–Campbell–Hausdorff 级数
$$
e^{A}\hat c_i^{\dagger}e^{-A}=\sum_{n\ge0}\frac{1}{n!}\,\mathrm{ad}_A^{\,n}\hat c_i^{\dagger}
=\sum_k\hat c_k^{\dagger}\sum_{n\ge0}\frac{(T^n)_{ki}}{n!}
=\sum_k\hat c_k^{\dagger}(e^{T})_{ki}.\qquad\blacksquare
$$

（对易子级数在 $T$ 的幂次上逐项终止地给出矩阵指数，对任意 $T$ 都成立，不要求 $T$ 厄米或幺正。）

### 推论 1（单粒子算符对 Slater determinant 的作用）

$$
\boxed{\;e^{\hat c^{\dagger}T\hat c}\,|\Phi(P)\rangle=|\Phi(e^{T}P)\rangle\;}
$$

**证明.** 记 $|\Phi(P)\rangle=\prod_{m=1}^{a}\Big(\sum_i P_{im}\hat c_i^{\dagger}\Big)|0\rangle$。因 $e^{-\hat c^{\dagger}T\hat c}|0\rangle=|0\rangle$，有
$$
e^{\hat c^{\dagger}T\hat c}|\Phi(P)\rangle
=\prod_{m=1}^{a}\Big(\sum_i P_{im}\,e^{\hat c^{\dagger}T\hat c}\hat c_i^{\dagger}e^{-\hat c^{\dagger}T\hat c}\Big)|0\rangle
=\prod_{m=1}^{a}\Big(\sum_j \hat c_j^{\dagger}(e^{T}P)_{jm}\Big)|0\rangle=|\Phi(e^{T}P)\rangle.\quad\blacksquare
$$
这正是文中 $|\Psi\rangle\to|B_\beta\Psi\rangle=|\Phi(B_\beta P)\rangle$ 的依据。

### 引理 2（Slater determinant 交叠）

$$
\boxed{\;\langle\Phi(Q)|\Phi(P)\rangle=\det\!\big(Q^{\dagger}P\big)\;}
$$

## 2. 坍缩概率的行列式形式

### 2.1 概率 = 坍缩后态矢的模方

$$
P(\{S,E\})
=\frac{\langle\Psi|M_\beta^{\dagger}M_\beta|\Psi\rangle}{\langle\Psi|\Psi\rangle}
=\frac{\langle\Psi|e^{-4\beta\hat N_E}|\Psi\rangle}{\langle\Psi|\Psi\rangle}
=\frac{\big\langle \Phi(B_\beta P)\big|\Phi(B_\beta P)\big\rangle}{\langle\Phi(P)|\Phi(P)\rangle}
=\boxed{\;\frac{\det\!\big(P_\beta^{\dagger}P_\beta\big)}{\det\!\big(P^{\dagger}P\big)}\;}
$$
其中
$$
P_\beta\equiv B_\beta P=e^{-2\beta D_E}P .
$$
（用引理 2 与推论 1；$M_\beta^{\dagger}=M_\beta$。）

### 2.2 化为 $a\times a$ 行列式

因 $B_\beta^{2}=e^{-4\beta D_E}=I-\lambda D_E$（$\lambda=1-e^{-4\beta}$），故
$$
P(\{S,E\})
=\frac{\det\!\big(P^{\dagger}(I-\lambda D_E)P\big)}{\det(P^{\dagger}P)}
=\det\!\Big(I_a-\lambda\,M_E\Big),\qquad M_E\equiv(P^{\dagger}P)^{-1}P^{\dagger}D_E P .
$$
用 $\det(I-AB)=\det(I-BA)$（取 $A=(P^{\dagger}P)^{-1}P^{\dagger}D_E$，$B=P$）：
$$
P(\{S,E\})=\det\!\big(I_{N_s}-\lambda\,D_E G\big).
$$
$D_EG$ 只有 $E$ 中的行非零，其余行是单位向量 $e_k$；沿这些行展开即只剩 $E$ 指标的子块：
$$
\boxed{\;P(\{S,E\})=\det\!\big(I_{|E|}-\lambda\,G_E\big)=\det\!\big(W_E\big)\;}
$$

### 2.3 三种等价写法

**(i) 占据数期望值形式。** 因 $\hat n_i\in\{0,1\}$，
$$
e^{-4\beta\hat n_i}=1+(e^{-4\beta}-1)\hat n_i=1-\lambda\hat n_i
$$
$$
\Longrightarrow\quad
\boxed{\;P(\{S,E\})=\Big\langle\Psi\Big|\prod_{i\in E}(1-\lambda\hat n_i)\Big|\Psi\Big\rangle\Big/\langle\Psi|\Psi\rangle\;}
$$
（把行列式理解为费米子高斯积分即可直接得到。）

**(ii) 互补块形式（Jacobi 恒等式）。**

先把记号写清楚。设 $W$ 为 $N_s\times N_s$ 可逆阵，$E,S$ 是一对**互补**指标集（本题中 $|E|=|S|=a=N_s/2$）。记 $W[E,E]$ 为取 $E$ 中的行、$E$ 中的列拼成的 $a\times a$ 子块（即前面的 $W_E$），$W^{-1}[S,S]$ 同理。

**引理（Jacobi 互补余子式恒等式，方阵互补情形）。**

$$
\boxed{\;\det W[E,E]=\det W\;\cdot\;\det W^{-1}[S,S]\;}
$$

一句话概括：**$W$ 在 $E$ 上的 $a$ 阶主子式，等于 $\det W$ 乘上 $W^{-1}$ 在互补指标集 $S$ 上的 $a$ 阶主子式。**（取行列式的是 $W^{-1}$ 的子块，不是 $W$ 的子块的逆——这一点见下面的注记。）

**证明（Schur 补）。** 把指标重排成"$E$ 在前、$S$ 在后"，即用同一个置换 $\Pi$ 同时作用在行与列上：

$$
\Pi W\Pi^{\mathsf T}=\begin{pmatrix} A & B\\ C & D\end{pmatrix},
\qquad
\begin{aligned}
A&=W[E,E], \qquad B&=W[E,S],\\
C&=W[S,E], \qquad D&=W[S,S].
\end{aligned}
$$

由于行、列用的是**同一个**置换，$\det(\Pi W\Pi^{\mathsf T})=\det(\Pi)^2\det W=\det W$，重排不改变行列式（关键：若行、列用不同置换就会多出符号，见"一般形式"）。由分块行列式公式

$$
\det W=\det A\cdot\det\!\big(D-CA^{-1}B\big),
$$

（此式在 $A$ 可逆时成立。在 $\det W\neq0$ 的开集内 $A$ 可逆的点稠密，而等式两边都是该开集上的连续函数，故对一切可逆 $W$ 成立。）另一方面，分块求逆公式给出 $W^{-1}$ 的右下块恰为 Schur 补之逆：

$$
W^{-1}[S,S]=\big(D-CA^{-1}B\big)^{-1}
\qquad\Longrightarrow\qquad
\det W^{-1}[S,S]=\frac{1}{\det\!\big(D-CA^{-1}B\big)} .
$$

两者相乘即得

$$
\det W\cdot\det W^{-1}[S,S]
=\det A\cdot\det\!\big(D-CA^{-1}B\big)\cdot\frac{1}{\det\!\big(D-CA^{-1}B\big)}
=\det A=\det W[E,E].
\qquad\blacksquare
$$

**一般形式（带符号）。** 对任意 $\alpha,\beta$，只要 $|\alpha|=|\beta|=k$，就有

$$
\det W[\alpha,\beta]=(-1)^{p(\alpha)+p(\beta)}\det W\;\cdot\;\det W^{-1}[\beta^{c},\alpha^{c}],
\qquad p(\gamma)\equiv\sum_{i\in\gamma}i .
$$

本题中 $\alpha=\beta=E$，$p(E)+p(E)=2p(E)$ 为偶数，符号为 $+1$，回到上面的无符号形式。

**经典叙述（伴随阵）。** 用 $\mathrm{adj}(W)=\det(W)\,W^{-1}$，引理等价于

$$
\det\!\big(\mathrm{adj}(W)[S,S]\big)=\det(W)^{\,a-1}\,\det W[E,E],
$$

即通常所说的 Jacobi 定理：伴随阵的 $k$ 阶余子式与 $W$ 的 $(n-k)$ 阶余子式互补对应。验证：$\det(\mathrm{adj}\,W[S,S])=\det(W)^{a}\det W^{-1}[S,S]=\det(W)^{a-1}\det W[E,E]$。

**一个 $2\times2$ 的检验。** 取 $E=\{1\}$、$S=\{2\}$，

$$
W=\begin{pmatrix} w_{11} & w_{12}\\ w_{21} & w_{22}\end{pmatrix},
\qquad
W^{-1}=\frac{1}{\det W}\begin{pmatrix} w_{22} & -w_{12}\\ -w_{21} & w_{11}\end{pmatrix},
$$

于是 $\det W[E,E]=w_{11}$，$\det W^{-1}[S,S]=w_{11}/\det W$，相乘得 $w_{11}$，等式成立。

**注记：$W^{-1}[S,S]\neq(W[S,S])^{-1}$。** 这是最容易搞混的一点，二者由 Schur 补联系：

$$
W^{-1}[S,S]=\Big(W[S,S]-W[S,E]\,W[E,E]^{-1}\,W[E,S]\Big)^{-1},
$$

上式正是证明中用到的分块求逆公式。**但在本题中这个区别无关紧要**，因为 $W=I-\lambda G$、$W^{-1}=I+\eta G$ 都写成 $I$ 与 $G$ 的组合，取 $S$ 子块时是直接逐块取的。

**代回本题。** 由 $G$ 的本征值为 $1$（$a$ 重）与 $0$（$N_s-a=a$ 重），
$$
\det W=\det(I-\lambda G)=(1-\lambda)^{a}=e^{-4\beta a},
\qquad
W^{-1}=(I-\lambda G)^{-1}=I+\frac{\lambda}{1-\lambda}G=I+\eta G .
$$

而 $W^{-1}[S,S]=(I+\eta G)[S,S]=I_S+\eta\,G_S$（单位阵取子块仍是单位阵），所以

$$
\boxed{\;P(\{S,E\})=\det W[E,E]=e^{-4\beta a}\,\det\!\big(I_S+\eta\,G_S\big)\;\propto\;\det\!\big(I_S+\eta\,G_S\big)\;}
$$

与 (i) 完全等价（比例常数 $e^{-4\beta a}$ 与构型无关，在 Metropolis 比值里约掉）。两个极端值得注意：

- $E$ 中的格点在初态里**确实为空**（$G_E=0$）$\Rightarrow P=1$；
- $E$ 中的格点在初态里**全部被占**（$G_E=I_a$）$\Rightarrow P=(1-\lambda)^a=e^{-4\beta a}\to0$。

行为正确：$\beta$ 越大越倾向于把 $E$ 选在初态的空位上。

**(iii) 归一化。** $\sum_{|E|=a}P(\{S,E\})$ 在有限 $\beta$ 下**不等于 1**（例如 $N_s=4,a=2$ 时 $\beta=0.1$ 给出 $\approx4.13$）；只有 $\beta\to\infty$ 时
$$
\sum_{|E|=a}\det\!\big(I_E-G_E\big)=1 .
$$
所以这是"软"测量（滤波），$M_\beta$ 不是 POVM 元；对 Metropolis 而言只需要相对权重，归一化常数不影响 ratio。

## 3. 单点交换的 Metropolis 比值

### 3.1 目标

随机取 $i\in S$、$j\in E$ 交换：$i$ 从"目标占据"移到"目标空"，$j$ 反之。于是
$$
S'=S\setminus\{i\}\cup\{j\},\qquad
E'=E\setminus\{j\}\cup\{i\}=(E\cup\{i\})\setminus\{j\}.
$$
$$
ratio=\frac{P(\{S',E'\})}{P(\{S,E\})}
=\frac{\det(W_{E'})}{\det(W_E)} .
$$

### 3.2 先"加 $i$"再"删 $j$"：Schur 补 + 余子式

**为什么要拆成两步？** $E$ 与 $E'$ 是**不同的指标集**（差一个元素），所以 $\det(W_{E'})/\det(W_E)$ 不是"同一个矩阵的秩一更新"，Sherman–Morrison 那类公式直接用不上。但"加一个指标"和"删一个指标"各自都有干净的公式，于是拆成两步相乘：

$$
\frac{\det(W_{E'})}{\det(W_E)}
=\frac{\det(W_T)}{\det(W_E)}\cdot\frac{\det(W_{E'})}{\det(W_T)},
\qquad
T\equiv E\cup\{i\},\qquad E'=T\setminus\{j\}.
$$

左因子靠 Schur 补，右因子靠余子式。

**一个预备说明（这里不存在符号歧义）。** 对指标集 $X$，$\det(W_X)$ 指"行、列按 $X$ 的**同一个**排列"拼出的子块的行列式。把 $X$ 换一个排列，只是对子块做相似变换 $P(\cdot)P^{\mathsf T}$，而 $\det(P)^{2}=1$，所以**取值与排列无关**。因此下文可以放心地把 $i$ 排到最后、把 $j$ 划掉，不必担心多出符号。（这与 §2.3(ii) 里"行、列必须用同一个置换"是同一件事。）

**第一步（加 $i$）。** 令 $T=E\cup\{i\}$，把 $i$ 排在最后，按 $(E;i)$ 分块：

$$
W_T=\begin{pmatrix} W_E & w_{Ei}\\ w_{iE} & W_{ii}\end{pmatrix},
\qquad
(w_{Ei})_k=W_{e_k i}\ \ (a\times1),
\qquad
(w_{iE})_k=W_{i e_k}\ \ (1\times a).
$$

对分块矩阵做 LDU 分解——两个外侧因子是单位对角的分块三角阵，行列式都是 $1$：

$$
\begin{pmatrix} A & B\\ C & D\end{pmatrix}
=\begin{pmatrix} I & 0\\ CA^{-1} & I\end{pmatrix}
\begin{pmatrix} A & 0\\ 0 & D-CA^{-1}B\end{pmatrix}
\begin{pmatrix} I & A^{-1}B\\ 0 & I\end{pmatrix}.
$$

两边取行列式即得**分块行列式公式**

$$
\det\begin{pmatrix} A & B\\ C & D\end{pmatrix}=\det A\cdot\det\!\big(D-CA^{-1}B\big).
$$

在本题里 $A=W_E$、$B=w_{Ei}$、$C=w_{iE}$、$D=W_{ii}$（标量），故 $D-CA^{-1}B$ 是一个**标量**，即 **Schur 补**

$$
\boxed{\;S_i\equiv W_{ii}-w_{iE}\,W_E^{-1}\,w_{Ei}\;}
$$

于是

$$
\boxed{\;\det(W_T)=\det(W_E)\cdot S_i\;}
$$

（$S_i$ 的含义：把 $E$ 块"积分掉"之后格点 $i$ 上的等效对角元；由下面的块求逆公式还知道 $S_i=1/(W_T^{-1})_{ii}$。）

**第二步（删 $j$）。** 对可逆阵 $M$ 有 $M^{-1}=\mathrm{adj}(M)/\det M$，而伴随阵的对角元就是余子式：

$$
\big(\mathrm{adj}\,M\big)_{jj}=C_{jj}=(-1)^{\,j+j}\det\big(M_{\hat j}\big)=\det\big(M_{\hat j}\big),
$$

其中 $M_{\hat j}$ 表示划掉第 $j$ 行与第 $j$ 列。于是得到**余子式恒等式**

$$
\boxed{\;(M^{-1})_{jj}=\frac{\det\big(M_{\hat j}\big)}{\det M}\;}
$$

取 $M=W_T$、$j=e_p$（$E$ 中第 $p$ 个格点）：划掉 $j$ 之后剩下的指标集恰好是 $T\setminus\{j\}=E'$，所以

$$
(W_T^{-1})_{jj}=\frac{\det(W_{E'})}{\det(W_T)} .
$$

**合并两步。** 把 $\det(W_T)=\det(W_E)\,S_i$ 代进去：

$$
\frac{\det(W_{E'})}{\det(W_E)}
=\frac{\det(W_T)\,(W_T^{-1})_{jj}}{\det(W_E)}
=\frac{\det(W_E)\,S_i\,(W_T^{-1})_{jj}}{\det(W_E)}
=\boxed{\;S_i\,(W_T^{-1})_{jj}\;}
$$

**第三步（块求逆）。** 还差 $W_T^{-1}$ 的 $(jj)$ 元。对上面的 LDU 分解取逆（顺序反过来）：

$$
\begin{pmatrix} A & B\\ C & D\end{pmatrix}^{-1}
=\begin{pmatrix} I & -A^{-1}B\\ 0 & I\end{pmatrix}
\begin{pmatrix} A^{-1} & 0\\ 0 & \big(D-CA^{-1}B\big)^{-1}\end{pmatrix}
\begin{pmatrix} I & 0\\ -CA^{-1} & I\end{pmatrix},
$$

展开左上块：

$$
W_T^{-1}=\begin{pmatrix}
W_E^{-1}+W_E^{-1}w_{Ei}\,S_i^{-1}\,w_{iE}W_E^{-1} & -W_E^{-1}w_{Ei}\,S_i^{-1}\\[1mm]
-S_i^{-1}\,w_{iE}W_E^{-1} & S_i^{-1}
\end{pmatrix}.
$$

因为 $j\in E$，它落在左上块里；取 $(jj)$ 元（记 $X\equiv W_E^{-1}$，$S_i$ 是标量可以提出去）：

$$
(W_T^{-1})_{jj}=X_{jj}+\big(Xw_{Ei}\big)_j\,S_i^{-1}\,\big(w_{iE}X\big)_j .
$$

**代回。** 乘上 $S_i$ 时，$S_i$ 与 $S_i^{-1}$ 恰好约掉：

$$
ratio=S_i\,(W_T^{-1})_{jj}
=S_i\,X_{jj}+\big(Xw_{Ei}\big)_j\big(w_{iE}X\big)_j,
\qquad
\boxed{\;X\equiv W_E^{-1}=(I_E-\lambda G_E)^{-1}\;}
$$

这就是最紧凑的形式：只用到当前 $E$ 块的逆 $X$ 以及 $W$ 的一行一列，不必碰 $N_s\times N_s$ 的大矩阵。注意 $S_i$ 与 $S_i^{-1}$ 的约掉不是巧合——它正是"加 $i$ 再删 $j$"这个循环抵消的体现。

**$a=1$ 的显式检验。** 取 $E=\{j\}$、$S=\{i\}$，此时 $W_T=W$ 就是 $2\times2$：

$$
S_i=W_{ii}-W_{ij}W_{jj}^{-1}W_{ji}=\frac{\det W}{W_{jj}},
\qquad
(W^{-1})_{jj}=\frac{W_{ii}}{\det W},
$$

相乘得 $ratio=W_{ii}/W_{jj}$；而直接算 $\det(W_{E'})/\det(W_E)=W_{ii}/W_{jj}$，一致。

**注记（等价地在 $S$ 侧做）。** 用 §2.3(ii) 的互补形式 $P\propto\det(W'_S)$，其中 $W'=I+\eta G=I-(-\eta)G$，交换后 $S'=(S\cup\{j\})\setminus\{i\}$。完全相同的两步给出

$$
ratio=S'_j\,\Big(\big(W'_{S\cup\{j\}}\big)^{-1}\Big)_{ii},
\qquad
S'_j=W'_{jj}-w'_{jS}\,(W'_S)^{-1}\,w'_{Sj},
$$

即"把 $\lambda\to-\eta$、$E\to S$、$i\leftrightarrow j$"后的同一个公式。两种写法严格等价，实现时任选其一。

### 3.3 代入 $W=I-\lambda G$

因为 $i\notin E$，
$$
W_{ii}=1-\lambda G_{ii},\qquad
w_{iE}=-\lambda\,g_{iE},\qquad
w_{Ei}=-\lambda\,g_{Ei},
$$
其中 $g_{iE}$ 是行向量 $(G_{i e_1},\dots,G_{i e_a})$，$g_{Ei}$ 是列向量 $(G_{e_1 i},\dots,G_{e_a i})^{\mathsf T}$。定义
$$
\boxed{\;u_j\equiv\big(Xg_{Ei}\big)_j=\sum_{l=1}^{a}X_{jl}\,G_{e_l i}\;},\qquad
\boxed{\;w_j\equiv\big(g_{iE}X\big)_j=\sum_{k=1}^{a}G_{i e_k}\,X_{kj}=u_j\;},
$$
$$
\boxed{\;q_i\equiv g_{iE}Xg_{Ei}=\sum_{k,l=1}^{a}G_{i e_k}X_{kl}\,G_{e_l i}\in\mathbb R\;}
$$
（$u_j=w_j$ 用到 $G=G^{\mathsf T}$、$X=X^{\mathsf T}$）。于是 $S_i=1-\lambda G_{ii}-\lambda^{2}q_i$，最终

$$
\boxed{\;
ratio=\Big(1-\lambda G_{ii}-\lambda^{2}q_i\Big)\,X_{jj}+\lambda^{2}\,u_j^{2}
\;}
$$

> **实现提示.** 这里 $u_j$、$q_i$、$S_i$ 都用**完整 $E$** 算，是推导的中间结果，采样时不必算它们。实际用的等价形式是 §4.5 的
> $$ratio=\hat S_i\,X_{jj},$$
> 其中 $\hat S_i$ 是更新 $X$ 时顺带得到的量（用**缩减集** $\hat p=E\setminus\{j\}$，$j=e_p$ 即被换掉的那个元素）。两者相等由 §4.5(a) 的两条桥接恒等式保证。

写全指标：
$$
ratio=\Big(1-\lambda G_{ii}-\lambda^{2}\sum_{k,l=1}^{a}G_{i e_k}X_{kl}G_{e_l i}\Big)X_{jj}
+\lambda^{2}\Big(\sum_{l=1}^{a}X_{jl}G_{e_l i}\Big)^{2}.
$$

**极限检验**

- $\beta\to0$：$\lambda\to0$，$X\to I_E$，$ratio\to1$。没有测量，权重均匀。✔
- $G$ 对角（初态是格点占据数本征态）：$u_j=q_i=0$，
$$
ratio=\frac{1-\lambda G_{ii}}{1-\lambda G_{jj}} .
$$
  若 $i$ 在初态被占（$G_{ii}=1$）而被提议改为"空"、$j$ 在初态为空（$G_{jj}=0$）而被提议改为"占"，则 $ratio=e^{-4\beta}<1$，被压制；反过来的提议 $ratio=e^{4\beta}>1$，被促进。✔

### 3.4 等价形式：滤波态 + 占据数

把 $N_{E'}=N_E-\hat n_j+\hat n_i$（注意 $j\in E$、$i\notin E$，各项对易）代入：
$$
e^{-4\beta\hat N_{E'}}=e^{-4\beta\hat N_E}\,e^{4\beta\hat n_j}\,e^{-4\beta\hat n_i}
=e^{-4\beta\hat N_E}\,(1+\eta\hat n_j)(1-\lambda\hat n_i)
$$
定义**滤波态**
$$
|\Psi_E\rangle\equiv\frac{e^{-2\beta\hat N_E}|\Psi\rangle}{\|e^{-2\beta\hat N_E}|\Psi\rangle\|}
\;\propto\;|\Phi(B_\beta P)\rangle,
\qquad
\mathcal G\equiv P_\beta(P_\beta^{\dagger}P_\beta)^{-1}P_\beta^{\dagger},
$$
则（因 $e^{-4\beta\hat N_E}=(e^{-2\beta\hat N_E})^{\dagger}(e^{-2\beta\hat N_E})$，且 $\hat n_i,\hat n_j$ 与 $\hat N_E$ 对易）
$$
\boxed{\;ratio=\big\langle\Psi_E\big|(1+\eta\hat n_j)(1-\lambda\hat n_i)\big|\Psi_E\big\rangle\;}
$$
对 Slater determinant 用 Wick 定理（$\langle\hat n_i\hat n_j\rangle=\langle\hat n_i\rangle\langle\hat n_j\rangle-|\mathcal G_{ij}|^{2}$）：
$$
\boxed{\;
ratio=1+\eta\,\mathcal G_{jj}-\lambda\,\mathcal G_{ii}
-\eta\lambda\big(\mathcal G_{ii}\mathcal G_{jj}-|\mathcal G_{ij}|^{2}\big)
\;}
$$
这个形式只用 $\mathcal G$ 的四个元素，物理意义一目了然：**把"空目标"格点 $j$ 变成"占据目标"要付出 $\eta\langle n_j\rangle$ 的代价，把"占据目标"格点 $i$ 变成"空目标"要付出 $\lambda\langle n_i\rangle$ 的代价，末项是配对关联修正。**

## 4. 实现：$X=(I_E-\lambda G_E)^{-1}$ 的增量更新

接受交换后 $E\to E'$，需把 $X$ 更新为 $X'$，不必重新求逆。下面只用到一个观察，**全程只用 $X$、$G$、$\lambda$**。

### 4.1 关键观察：$M_{E'}$ 与 $M_E$ 只差第 $p$ 行、第 $p$ 列

记
$$
\boxed{\;M_E\equiv I_E-\lambda G_E,\qquad X=M_E^{-1}\;}
$$
（$I_E$ 是 $E$ 上的单位阵，$G_E=G[E,E]$）。约定 **$E'$ 与 $E$ 同序，只把第 $p$ 位上的 $j=e_p$ 换成 $i$**：
$$
E'=\big(e_1,\dots,e_{p-1},\;i,\;e_{p+1},\dots,e_a\big).
$$
按位置对齐后，$M_{E'}$ 与 $M_E$ 在 $k,l\neq p$ 处**完全相同**（那里 $E'[k]=e_k$、$E'[l]=e_l$），只有第 $p$ 行、第 $p$ 列被换成了 $W$ 的 $i$ 行、$i$ 列：
$$
(M_{E'})_{pl}=(M_{E'})_{lp}=W[i,e_l]\quad(l\neq p),\qquad (M_{E'})_{pp}=W[i,i].
$$
（$i\notin E$ 保证 $W[i,e_l]=-\lambda G_{ie_l}$。）

所以整个问题就是：**把一个已知矩阵的第 $p$ 行、第 $p$ 列换掉，它的逆怎么变。** 分两步：先把第 $p$ 行/列**删掉**，再把新的第 $p$ 行/列**插回去**。

### 4.2 第一步：删掉第 $p$ 行、第 $p$ 列

记 $\hat p\equiv E\setminus\{e_p\}$（保持原序的缩减指标集）。

**引理.** 设 $N=M^{-1}$，则 $\big(M_{\hat p\hat p}\big)^{-1}=N_{\hat p\hat p}-\dfrac{N_{\hat p\,p}N_{p\,\hat p}}{N_{pp}}$。

*证明.* 把 $N$ 按 $(p,\hat p)$ 分块：$N=\begin{pmatrix}d&c^{\mathsf T}\\ b&A\end{pmatrix}$。由分块求逆公式，$M=N^{-1}$ 的右下块是 $\big(A-bd^{-1}c^{\mathsf T}\big)^{-1}$，即 $M_{\hat p\hat p}=\big(A-bd^{-1}c^{\mathsf T}\big)^{-1}$。两边取逆（$d=N_{pp}$ 是标量）即得。$\blacksquare$

取 $M=M_E$、$N=X$：
$$
\boxed{\;\hat X\equiv\big(M_E[\hat p,\hat p]\big)^{-1}=X_{\hat p\hat p}-\frac{X_{\hat p\,p}\,X_{p\,\hat p}}{X_{pp}}\;}
$$
即"把 $X$ 的第 $p$ 行/列删掉，再减去由它们张成的一个外积"。注意 $M_{E'}[\hat p,\hat p]=M_E[\hat p,\hat p]$，所以 $\hat X$ 也正是 $X'$ 的 $\hat p$ 块的"母矩阵"之逆。

反过来把 $W_E$ 按 $(p,\hat p)$ 分块，记
$$
b\equiv W[\hat p,j]\in\mathbb R^{a-1},\qquad \sigma\equiv W_{jj}-b^{\mathsf T}\hat X b ,
$$
分块求逆给出后面要反复引用的四条关系：
$$
\boxed{\;X_{pp}=\frac1\sigma,\qquad
X_{p\hat p}=-\frac1\sigma b^{\mathsf T}\hat X,\qquad
X_{\hat p p}=-\frac1\sigma \hat X b,\qquad
X_{\hat p\hat p}=\hat X+\frac1\sigma\hat X bb^{\mathsf T}\hat X\;}
$$
（第一条即 $\sigma=1/X_{pp}$。）

### 4.3 第二步：把 $i$ 的第 $p$ 行/列插回去

按"$\hat p$ 块 + 第 $p$ 位"分块：
$$
M_{E'}=\begin{pmatrix}M_E[\hat p,\hat p] & y\\ y^{\mathsf T} & W_{ii}\end{pmatrix},
\qquad y\equiv W[\hat p,i]=-\lambda\,g_{\hat p i},
$$
其中 $g_{\hat p i}$ 是 $G$ 的 $i$ 列在 $\hat p$ 上的那一部分。对可逆块用分块求逆
$$
\begin{pmatrix}A&b\\ b^{\mathsf T}&d\end{pmatrix}^{-1}
=\begin{pmatrix}A^{-1}+\frac{1}{\sigma}A^{-1}bb^{\mathsf T}A^{-1} & -\frac{1}{\sigma}A^{-1}b\\[1mm]
-\frac{1}{\sigma}b^{\mathsf T}A^{-1} & \frac{1}{\sigma}\end{pmatrix},
\qquad \sigma\equiv d-b^{\mathsf T}A^{-1}b,
$$
代入 $A^{-1}=\hat X$、$b=y$、$d=W_{ii}$，并记
$$
\boxed{\;\tilde u\equiv \hat X\,g_{\hat p i},\qquad \hat q_i\equiv g_{i\hat p}\tilde u,\qquad
\hat S_i\equiv W_{ii}-y^{\mathsf T}\hat X y=1-\lambda G_{ii}-\lambda^{2}\hat q_i\;}
$$
（于是 $-\hat X y=\lambda\tilde u$，$\sigma=\hat S_i$），得到
$$
\boxed{\;
X'=\bar X+\frac{\lambda^{2}}{\hat S_i}\,\bar u\,\bar u^{\mathsf T}
\;}
$$
其中 $\bar X$ 是"把 $\hat X$ 摆回原来的位置、第 $p$ 行/列留零"，$\bar u$ 是"把 $\tilde u$ 摆回原来的位置、第 $p$ 位填 $1/\lambda$"：
$$
\bar u_k=\begin{cases}\tilde u_k,&k\neq p\\[1mm] 1/\lambda,&k=p\end{cases}
$$
**新的第 $p$ 行/列就放在第 $p$ 位——这正是同序约定，不需要任何置换。**

### 4.4 写开

$$
\boxed{\;X'_{kl}=(\hat X)_{kl}+\frac{\lambda^{2}\tilde u_k\tilde u_l}{\hat S_i}\quad(k,l\neq p),\qquad
X'_{kp}=X'_{pk}=\frac{\lambda\,\tilde u_k}{\hat S_i}\quad(k\neq p),\qquad
X'_{pp}=\frac{1}{\hat S_i}\;}
$$
矩阵对称，全部只用到 $X$、$G$、$\lambda$。$\tilde u$ 与秩一修正各需 $O(a^{2})$，所以整个提议 + 更新是 $O(a^{2})$ 的。

### 4.5 与 $ratio$ 共用的量：统一命名

#### (a) 关键恒等式：$ratio=\hat S_i\,X_{jj}$

上一节为**更新** $X$ 而算出的 $\hat S_i$，同时就是 §3 的**接受概率**里的那个标量。只需把 §4.2 的四条分块关系代入 §3.3。（$j=e_p$，故 $X_{jj}=X_{pp}$，以下两个写法混用。）记
$$
\beta\equiv b^{\mathsf T}\tilde u-G_{ji}\in\mathbb R
\qquad\big(\Rightarrow\ b=-\lambda g_{\hat p j},\quad W_{ji}=-\lambda G_{ji}\big).
$$

**恒等式 1（§3.3 的 $u_j$）.** 用 $X_{p\hat p}=-\frac1\sigma b^{\mathsf T}\hat X$、$X_{pp}=\frac1\sigma$、$\tilde u=\hat Xg_{\hat p i}$：
$$
u_j=\big(Xg_{Ei}\big)_j=X_{p\hat p}\,g_{\hat p i}+X_{pp}\,G_{ji}
=-\frac1\sigma b^{\mathsf T}\tilde u+\frac1\sigma G_{ji}
=-\frac{\beta}{\sigma}
=\boxed{-X_{jj}\,\beta}
$$

**恒等式 2（§3.3 的 $S_i$）.** 用 $S_i=W_{ii}-W[i,E]XW[E,i]$，代入分块形式并整理（记 $b_i\equiv W[\hat p,i]=-\lambda g_{\hat p i}$，$\delta\equiv b^{\mathsf T}\hat Xb_i=-\lambda b^{\mathsf T}\tilde u$）：
$$
S_i=\underbrace{W_{ii}-b_i^{\mathsf T}\hat Xb_i}_{=\ \hat S_i}-\frac{(\delta-W_{ji})^{2}}{\sigma}
=\hat S_i-\frac{\lambda^{2}\beta^{2}}{\sigma}
=\boxed{\ \hat S_i-\lambda^{2}X_{jj}\,\beta^{2}\ }
$$

把两条恒等式代回 §3.3 的比值公式，$\beta$ 被完全消掉：
$$
ratio=S_iX_{jj}+\lambda^{2}u_j^{2}
=\big(\hat S_i-\lambda^{2}X_{jj}\beta^{2}\big)X_{jj}+\lambda^{2}X_{jj}^{2}\beta^{2}
=\boxed{\ \hat S_i\,X_{jj}\ }
$$

> **结论：$ratio$ 不需要任何独立计算。** §3.3 的 $u_j$、$q_i$、$S_i$ 是用**完整 $E$** 算的中间量，实现时一律不出现；只要算本节这一组（用**缩减集 $\hat p$**），再乘一个 $X_{jj}$，就得到接受概率。

#### (b) 统一命名

实现中每个提议 $(i,p)$ 只维护下面这几个量：

| 符号 | 定义 | 维数 | 每步代价 | 用途 |
|---|---|---|---|---|
| $\hat p$ | $E\setminus\{e_p\}$ | — | — | 缩减指标集 |
| $\tilde g$ | $G[E,i]$，把第 $p$ 位清零 | $a$ | — | — |
| $z$ | $z\equiv X\tilde g$ | $a$ | $O(a^{2})$ | ratio |
| $\tilde u$ | $\tilde u\equiv z_{\hat p}-\dfrac{X_{\hat p p}}{X_{pp}}\,z_p\ \big(=\hat X g_{\hat p i}\big)$ | $a-1$ | $O(a)$ | **ratio 与更新共用** |
| $\hat S_i$ | $\hat S_i\equiv 1-\lambda G_{ii}-\lambda^{2}\,g_{i\hat p}\tilde u$ | 标量 | $O(a)$ | **ratio 与更新共用** |
| $ratio$ | $ratio=\hat S_i\,X_{jj}$ | 标量 | $O(1)$ | Metropolis 判据 |

$\tilde u$ 的两条写法等价：$z=X\tilde g$ 就是"把 $g_{Ei}$ 的第 $p$ 位清零"之后的 $Xg_{Ei}$，由 $X_{\hat p\hat p}=\hat X+\frac1\sigma\hat Xbb^{\mathsf T}\hat X$ 与 $X_{\hat p p}=-\frac1\sigma\hat Xb$ 立即可得
$$
z_{\hat p}-\frac{X_{\hat p p}}{X_{pp}}z_p=\hat Xg_{\hat p i}=\tilde u .
$$
这样 $\tilde u$ 和 $\hat S_i$ **完全不需要先构造 $\hat X$**。

#### (c) 最终流程（每个提议只做一次 $O(a^2)$ 矩阵-向量乘）

1. $z=X\tilde g$ —— $O(a^{2})$，**唯一的矩阵-向量乘**；
2. $\tilde u=z_{\hat p}-\dfrac{X_{\hat p p}}{X_{pp}}z_p$，$\hat S_i=1-\lambda G_{ii}-\lambda^{2}g_{i\hat p}\tilde u$ —— $O(a)$；
3. $ratio=\hat S_iX_{jj}$，与 $[0,1)$ 均匀随机数比较 —— $O(1)$；
4. **若拒绝**：什么都不改，本节结束（总代价一次 $O(a^{2})$ 乘）；
5. **若接受**：直接写回 $X$（不必单独生成 $\hat X$ 数组；右端全部用旧值）
$$
X_{\hat p\hat p}\leftarrow X_{\hat p\hat p}-\frac{X_{\hat p p}X_{p\hat p}}{X_{pp}}+\frac{\lambda^{2}}{\hat S_i}\tilde u\tilde u^{\mathsf T},\qquad
X_{\hat p p}=X_{p\hat p}\leftarrow\frac{\lambda}{\hat S_i}\tilde u,\qquad
X_{pp}\leftarrow\frac{1}{\hat S_i}.
$$

#### (d) 计算量对比

按 $O(a^{2})$ 量级的操作数计（$a=N_s/2$）：

| 做法 | 拒绝 | 接受 |
|---|---|---|
| 分开算（§3.3 的 $ratio$ + 4.2–4.4 的更新） | 1 次矩阵-向量乘 | 2 次矩阵-向量乘 + 2 次外积 |
| **共用（本节）** | 1 次矩阵-向量乘 | **1 次矩阵-向量乘 + 2 次外积** |

接受时省掉一整次 $O(a^{2})$ 矩阵-向量乘（即 §3.3 的 $Xg_{Ei}$），并且不再需要 §3.3 的 $u_j$、$q_i$、$S_i$ 那一整套中间量，$\hat X$ 也不必单独存成数组。拒绝路径不变（本来也只有一次乘）。

**数值校验.** 随机 $P$（$N_s\in\{6,8,10\}$、$a=N_s/2$、$\beta\in[0.05,3]$、各 20 组种子）下逐项验证（相对误差）：
$$
\underbrace{M_{E'}\text{ 与 }M_E\text{ 只在第 }p\text{ 行/列不同}}_{1.0\times10^{-16}},\qquad
\underbrace{\text{4.2 的引理}}_{4\times10^{-16}\sim1\times10^{-12}},\qquad
\underbrace{\text{4.4 的三条分量}}_{4\times10^{-16}\sim5\times10^{-12}}
$$
$$
\underbrace{ratio=\hat S_iX_{jj}}_{6\times10^{-16}\sim5\times10^{-12}},\qquad
\underbrace{\text{两条桥接恒等式}}_{1\times10^{-16}\sim3\times10^{-11}},\qquad
\underbrace{\tilde u,\hat S_i\text{ 的 }z\text{ 写法}}_{0\sim6\times10^{-12}}
$$
（$\beta=3$ 时 $W$ 近奇异、$|X|_{\max}\sim7\times10^{4}$，属浮点舍入。）

**注记（符号问题）**：$\det(W_E)$ 可以变号（$G_E$ 的子块本征值可复、可正可负），因此权重 $\det(W_E)$ 不是恒正的。若出现负权重，需按 $\operatorname{sgn}\det(W_E)$ 做符号重加权，或在 $\beta$ 不太大时使用；$\beta\to\infty$ 时 $\det(I_E-G_E)\ge0$ 恒成立。

## 5. 数值验证

用随机 $P$（$N_s=8,a=4$，多个 $\beta\in[0.05,3]$、多组随机种子、各 100 次随机交换）与穷举 $\det(W_{E'})/\det(W_E)$ 对比，三种表达式的最大相对误差 $\sim10^{-11}$（$\beta=3$ 时 $W$ 接近奇异，量级已到 $10^{-12}$，属浮点舍入）：

| 表达式 | 最大相对误差 |
|---|---|
| §3.2 一般形式 $S_iX_{jj}+(Xw_{Ei})_j(w_{iE}X)_j$ | $1.6\times10^{-12}$ |
| §3.3 显式形式 $(1-\lambda G_{ii}-\lambda^2q_i)X_{jj}+\lambda^2u_j^2$ | $2.0\times10^{-12}$ |
| §3.4 滤波态形式 $1+\eta\mathcal G_{jj}-\lambda\mathcal G_{ii}-\eta\lambda(\mathcal G_{ii}\mathcal G_{jj}-\mathcal G_{ij}^2)$ | $1.9\times10^{-11}$ |
| **§4.5 共用形式 $ratio=\hat S_iX_{jj}$** | $4.6\times10^{-12}$ |

§4 的增量更新与 §4.5 的共用量（相对误差，$\beta\in\{0.05,0.5,1,2,3\}$、$N_s\in\{6,8,10\}$、各 20 组种子）：

| 检查项 | 最大相对误差 |
|---|---|
| $M_{E'}$ 与 $M_E$ 只在第 $p$ 行/列不同 | $1.0\times10^{-16}$ |
| §4.2 引理 $\hat X=\big(M_E[\hat p,\hat p]\big)^{-1}$ | $3.9\times10^{-16}\sim1.3\times10^{-12}$ |
| §4.4 三条分量（$\hat p\hat p$、$\hat p p$、$pp$） | $4.0\times10^{-16}\sim4.6\times10^{-12}$ |
| §4.5 桥接恒等式 $u_j=-X_{jj}\beta$、$S_i=\hat S_i-\lambda^2X_{jj}\beta^2$ | $1.9\times10^{-15}\sim2.7\times10^{-11}$ |
| §4.5 的 $z$ 写法给出同一个 $\tilde u$、$\hat S_i$ | $0\sim5.8\times10^{-12}$ |

同时验证了 §2 的各条恒等式：
$\det(W_E)=\det(I_E-\lambda G_E)=e^{-4\beta a}\det(I_S+\eta G_S)=\det(P_\beta^{\dagger}P_\beta)/\det(P^{\dagger}P)=\langle\prod_{i\in E}(1-\lambda n_i)\rangle$，误差 $\sim10^{-15}$。



















