
$$
\ket{\Psi_M}=M\ket{\Psi}
$$

$\Psi$ 为初始波函数，$\Psi_M$ 为测量后坍缩的波函数，$M$ 为测量坍缩算符

对于粒子数测量：$\{n_i=0/1\}$，总格点数为 $N_s$，总粒子数为 $a=N_s/2$

如果我要坍缩到测量结果为：
$$
占据集合
S=\mathrm{nidx1}=\{i:n_i=1\},
\\
空格点集合

E=\mathrm{nidx2}=\{i:n_i=0\}.
$$
令

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
\boxed{ \ket{\Psi}\to M_\beta\ket{\Psi} = \ket{B_\beta \Psi} }
$$

同时坍缩到这个结果的概率为：

$$
P(\{S,E\})=\frac{\bra{\Psi}M_\beta M_\beta \ket{\Psi}}{\bra{\Psi}\ket{\Psi}}
$$

-------------

在格点很大的体系里，穷举遍历 $\{S,E\}$ 是很困难的，于是我想要用马尔科夫链重要性采样去遍历。比如现在我提议一个单点更新，分别随机选取 S 和 E 中的一个格点 index i 和 j 进行交换。

那么测量坍缩算符 $M_\beta \to M^\prime_\beta$ ，从而坍缩概率也会改变 $P(\{S,E\}) \to P(\{S_{i\to j},E_{j\to i}\})$。于是接受这个提议的概率为：
$$
ratio=\frac{P(\{S_{i\to j},E_{j\to i}\})}{P(\{S,E\})}
$$
你帮我进一步详细推导这个 ratio 的更新表达式。



















