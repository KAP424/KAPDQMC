
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots
import numpy as np
import sys
sys.path.append("C:/Users/admin/Desktop/JuliaDQMC/code/")
from DP_Lib import *
# 清除所有自定义设置
rcParams.update(plt.rcParamsDefault)
plt.style.use(['science','ieee'])

# 字体设置
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['mathtext.fontset'] = 'dejavusans'  # 数学字体使用 DejaVu Sans
rcParams['text.usetex'] = False  # 不使用 LaTeX 渲染文本
rcParams['font.size'] = 7  # 默认字号7.5pt
rcParams['axes.titlesize'] = 8
rcParams['axes.labelsize'] = 7
rcParams['xtick.labelsize'] = 6  # x轴刻度字号6pt
rcParams['ytick.labelsize'] = 6  # y轴刻度字号6pt

# 线条和边框宽度设置为0.75pt
rcParams['lines.linewidth'] = 0.5  # 线条宽度0.75pt
rcParams['axes.linewidth'] = 0.5  # 边框宽度0.75pt
rcParams['xtick.major.width'] = 0.3  # x轴主刻度线宽度0.75pt
rcParams['ytick.major.width'] = 0.3  # y轴主刻度线宽度0.75pt

rcParams['axes.labelpad'] = 1  # x/y标签与轴线的距离2pt
rcParams['lines.markersize'] = 4.0
# 刻度线设置
rcParams['xtick.direction'] = 'in'  # x轴刻度线向内
rcParams['ytick.direction'] = 'in'  # y轴刻度线向内
rcParams['xtick.major.pad'] = 1  # x轴刻度标签与轴线的距离
rcParams['ytick.major.pad'] = 1  # y轴刻度标签与轴线的距离

rcParams['xtick.minor.visible'] = False
rcParams['ytick.minor.visible'] = False
# 图例设置
rcParams['legend.frameon'] = False  # 图例不需要边框

def create_plot(m, n, col,sharex=False, sharey=False):

    if col==1:
        if m==1 and n==1:
            fig, ax = plt.subplots(m, n, figsize=(3.375, 2.2), dpi=300, constrained_layout=True,sharex=sharex, sharey=sharey)
        if m==2 and n==2:
            fig, ax = plt.subplots(m, n, figsize=(3.375, 3.375* 0.718), dpi=300, constrained_layout=True,sharex=sharex, sharey=sharey)
        if m==1 and n==2:
            fig, ax = plt.subplots(m, n, figsize=(3.375, 2.), dpi=300, constrained_layout=True,sharex=sharex, sharey=sharey)
        # fig, ax = plt.subplots(m, n, figsize=(width1, height1), dpi=300, constrained_layout=True)
    else:
        if m==1 and n==2:
            fig, ax = plt.subplots(m, n, figsize=(6.2, 2.55), dpi=300, constrained_layout=True,sharex=sharex, sharey=sharey)
        if m==2 and n==2:
            fig, ax = plt.subplots(m, n, figsize=(6.75, 6.75* 0.658), dpi=300, constrained_layout=True,sharex=sharex, sharey=sharey)
        elif m==1 and n==3:
            fig, ax = plt.subplots(m, n, figsize=(6.75, 2.15), dpi=300, constrained_layout=True,sharex=sharex, sharey=sharey)


    # fig, ax = plt.subplots(m, n, figsize=(col*width_1, col*height), dpi=300, constrained_layout=True)
    ax_arr = np.array(ax).reshape(-1)

    return fig, ax_arr

def Plot_fixTheta(U,Data,skip):
    t=Data['t'].values.astype(float)
    L=Data["L"].values.astype(float)
    EE=Data["EE"].values.astype(float)
    dEE=Data["dEE"].values.astype(float)
    Lu=np.unique(L)
    tlist=np.unique(Data.t.values)
    xticks=tlist
    
    fig=plt.figure(figsize=(12,8), dpi=200)
    # 调整 suptitle 与图内容的距离：提高 y 可增大间距
    fig.suptitle(f"neq SCEE of U={U} From AFM to Ground State", y=0.98)
    # 如需进一步下移图内容以留出空间，可调整 top（0-1，越小留白越多）
    fig.subplots_adjust(top=0.95)
    ax=plt.subplot(2,2,1)

    # Fix Theta
    Avg=[]
    dAvg=[]
    # figTheta=plt.figure(figsize=(8,6))
    for i, Θ in enumerate(tlist):
        idx = Data['t'] == Θ
        x = Data.loc[idx, 'L'].values
        y = Data.loc[idx, 'EE'].values
        dy = Data.loc[idx, 'dEE'].values

        # 找到最近邻数据点
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        dy = dy[sorted_indices]

        data=[(x[i],y[i],dy[i]) for i in range(len(x))]

        idx=unit_neardata(data)
        avg = np.mean(y[idx])
        Avg.append(avg)
        dAvg.append(np.mean(dy[idx]))

        plot_data(ax, x, y, yerr=dy,
            marker=markers[i%len(markers)], color=colors[i%len(colors)], label=rf'$\Theta={Θ}$')

        plt.axhline(avg, color=colors[i%len(colors)], linestyle='--')
        plt.text(min(L)-0.5, avg, f'{avg:.2f}', ha='center', va='bottom')

        for j in idx:
            plt.text(x[j], y[j], "[  ]", color='blue',ha='center',va='center',fontsize=10)
    plt.legend(bbox_to_anchor=(0.95, 0.85), loc='upper left', borderaxespad=0.)
    plt.xlabel('L')
    plt.ylabel(r'$\Delta S_2$')
    plt.xticks(Lu,Lu)
    plt.xlim(min(L)-1,max(L)+1)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Avg=np.array(Avg)
    dAvg=np.array(dAvg)
    para,ds=linear_fit(np.log(tlist[skip:]),Avg[skip:],dAvg[skip:])
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ax=plt.subplot(2,2,2)
    plt.xscale("log")
    plt.ylabel(r"$\Delta S_2$")
    plt.xlabel(r"$ \Theta$ (log scale)")

    # 遍历处理每个L值
    for l in Lu:
        # 获取当前L值的索引
        idx = np.where(np.array(L) == l)[0]

        if len(idx) > 0:
            # 提取数据并排序
            x = (t[idx])  # 原代码对t取对数
            y = EE[idx]
            dy = dEE[idx]
            
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            dy_sorted = dy[sorted_indices]
            
            i=np.where(Lu==l)[0][0]
            plot_data(ax, x_sorted, y_sorted, yerr=dy_sorted,
                marker=markers[i%len(markers)], color=colors[i%len(colors)], label=f"L={int(l)}")

    x=np.linspace(tlist[skip],tlist[-1],100)
    y=para[0]*np.log(x)+para[1]
    plt.plot(x, y,linewidth=3,alpha=0.4,color='red', label=f'$ y= ( {para[0]:.3f}\pm {ds:.3f} ) \ln \Theta + {para[1]:.3f}$')
    # 调整图例位置（右上角）
    plt.legend(bbox_to_anchor=(0.05, 0.9), loc='upper left', borderaxespad=0.)
    # 绘制基准线
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(xticks, xticks)
    plt.minorticks_off()
    plt.grid(True)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ax=plt.subplot(2,2,4)
    plot_data(ax, tlist, Avg, yerr=dAvg,
                marker=markers[2], color=colors[0], label=r"neq corner Entangle Entropy",markersize=18)
    # plt.errorbar(tlist,Avg,dAvg,fmt='o', marker=markers[1],markersize=5, capsize=5, ecolor='black',)

    x=np.linspace(tlist[skip],tlist[-1],100)
    y=para[0]*np.log(x)+para[1]
    plt.plot(x, y,linewidth=3,alpha=0.4,color='red', label=f'$ y= ( {para[0]:.3f}\pm {ds:.3f} ) \ln \Theta + {para[1]:.3f}$')
    s=para[0]
    const=para[1]
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"$ \Theta$ (log scale)")
    plt.ylabel(r"$\Delta S_2$")
    # 自定义x轴刻度
    # plt.xlim(0.2,4.5)
    plt.xticks(xticks, xticks)
    plt.minorticks_off()
    plt.grid(True)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ax=plt.subplot(2,2,3)
    for l in Lu:
        # 获取当前L值的索引
        idx = np.where(np.array(L) == l)[0]

        if len(idx) > 0:
            # 提取数据并排序
            x = t[idx]/L[idx]  # 原代码对t取对数
            y = EE[idx]-s*np.log(L[idx])
            dy = dEE[idx]
            
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            dy_sorted = dy[sorted_indices]
            
            # 绘制带误差条的曲线    
            i=np.where(Lu==l)[0][0]
            # 去除 fmt 中的 marker 仅保留线型
            plot_data(ax, x_sorted, y_sorted, yerr=dy_sorted,
                marker=markers[i%len(markers)],  label=f"L={int(l)}",color=colors[i%len(colors)])
            # plt.errorbar(x_sorted, y_sorted, yerr=dy_sorted,
            #         fmt='o--', markersize=5, capsize=5, ecolor='black',label=f"L={int(l)}")

    # xx=np.linspace(0.1,0.5,100)
    xx=np.linspace(min(t/L),max(t/L),100)
    yy=s*np.log(xx)+const
    ax.plot(xx,yy,linewidth=3,alpha=0.4,color='red',
            label=rf'$ y= {s:.3f} \ln x + {const:.3f} $')

    # 调整图例位置（右上角）
    plt.legend(bbox_to_anchor=(0.05, 0.9), loc='upper left', borderaxespad=0.)
    # 绘制基准线
    plt.xscale("log")
    tllist=np.round(np.arange(np.min(t/L),np.max(t/L),0.05),2).tolist()
    plt.xticks(tllist,tllist)
    # f'$ y={para[0]:.3f}  \Theta  {para[1]:.3f}$'
    plt.xlabel(r"$\frac{\Theta}{L}$")
    plt.ylabel(r"$\Delta S_2-s\ln L$")

    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plt.text(x_min+0.4*(x_max-x_min), y_min+0.08*(y_max-y_min),  r'$s \approx$' + f'{s:.4f}'+r"$\pm$"+f'{ds:.4f}', ha='center', va='bottom',bbox=dict(
                facecolor='yellow',  # 背景颜色
                alpha=0.5,           # 透明度
                edgecolor='black',   # 边框颜色
                boxstyle='round,pad=0.5'  # 圆角和边距
            ))
    return fig
 
# 设置数据点大小和误差棒宽度
errorbar_capsize = 2  # 误差棒宽度2pt
errorbar_linewidth = 1  # 误差棒线宽0.75pt
marker_edgewidth = 0.75  # 数据点边框宽度0.75pt
point_size = 5  # 数据点大小3pt
# 数据点形状列表和颜色列表
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
colors =['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

def plot_data(ax, x, y, yerr=None,marker=markers[0], color=colors[0], label=None,markersize=point_size,linestyle='--'):
    ax.errorbar(x, y, yerr=yerr, capsize=errorbar_capsize, markersize=point_size,ecolor='black',linestyle=linestyle,
                marker=marker, markerfacecolor='none', markeredgewidth=marker_edgewidth,color=color,
                elinewidth=errorbar_linewidth, label=label)
