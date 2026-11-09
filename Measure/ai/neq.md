1、 先确定一个构型的原胞大小，比如1x1,2x2,3x3；构型原胞里要求一半是占据1个费米子，另一半是空的。然后再通过平移不变得到整个晶格的构型，这就要求计算的晶格尺寸是构型原胞和3的公倍数。

```
function Config_set(site, subsite, config)
    """
    E = Config_set([6, 6], [3, 3], [1 1 -1; 1 1 1; -1 1 1])
    L = 6
    E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 0) - Int(x % 3 == 0 && y % 3 == 1) for y in 1:L for x in 1:L]
    sort(collect(E)) == sort(collect(E_idx))
    """
    subE = zeros(Int, prod(subsite))
    count = 1
    for i in 1:subsite[1]
        for j in 1:subsite[2]
            if config[i, j] == 1
                subE[count] = 2 * (i + (j - 1) * site[1])
            else
                subE[count] = 2 * (i + (j - 1) * site[1]) - 1
            end
            count += 1
        end
    end
    E = []
    for i in 1:div(site[1], subsite[1])
        for j in 1:div(site[2], subsite[2])
            E = vcat(E, subE .+ 2 * subsite[1] * (i - 1) .+ 2 * subsite[2] * site[2] * (j - 1))
        end
    end
    @assert length(E) == prod(site) "E_idx 长度不正确"
    return E
end
```

这个构型获得函数是一个special的，构型原胞里每个原胞都是一个占据一个空缺，更一般的没有这个要求，只要保证整个构型原胞是一半占据一半空缺的就行。

2、 以整个晶格的构型构建初始波函数 P，然后用自由费米子的hopping哈密顿量进行虚时投影，$$\ket{\psi(\theta)}=e^{-\theta H_0}\ket{P}$$

3、 再对投影后的波函数进行晶格构型的弱测量 $$\ket{\Psi(\beta,\theta)}=M_E(\beta) \ket{\psi(\theta)}$$

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

我们始终在 $$N_{\mathrm{part}}=a$$ 的粒子数扇区内，因此

$$
\hat N_S+\hat N_E=a,
$$

后面统一使用

$$
\boxed{M_E=e^{-2\beta\hat N_E}}.
$$

这里有几个要注意的事项需要你确认：

1. $$\ket{\psi(\theta)}=e^{-\theta H_0}\ket{P}$$  从初始构型开始虚时投影能在 $$\theta$$ 很大时候收敛，并且选择不同初始构型应该也收敛到相同的波函数
2. 你除了要尝试 Config_set 里special 的初始构型，你还要尝试我上面说的更一般的构型，看看结论是否一致

然后我需要你做下面几个任务：

1. 看平衡态的测量结果：让 $$\theta$$ 很大到虚时投影收敛，然后看看基态在不同 $$\beta$$ 大小测量坍缩后的纠缠熵随着系统尺寸 L 的scaling 结果会怎么变化。scaling是否和初始构型相关？初始构型的种类这么多，最终能得到什么规律吗？
2. 看看非平衡态的结果：不让 $$\theta$$ 很大收敛到基态，但同样要做 $$\beta$$ 的测量坍缩，看看在不同  $$\beta$$ 下得到是纠缠熵随着 $$\theta$$ 的scaling 行为是否与系统尺寸相关。目前已知在 $$\beta=0$$ 时候，短时 $$\theta$$ 的scaling 行为和 L 无关（过于小尺寸的除外）。然后如果和系统尺寸无关的话，能否从随着 $$\theta$$  的scaling 行为中提取出平衡态一样的 scaling 系数。
