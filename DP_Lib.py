import errno
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import csv
from scipy.stats import sem
from scipy.optimize import curve_fit

import numpy as np

def weighted_average_with_uncertainty(data_points, uncertainties):
    """
    计算加权平均值和加权平均值的不确定度
    
    参数:
    data_points: 一维数组，数据点 y1, y2, ..., yn
    uncertainties: 一维数组，对应的不确定度 δy1, δy2, ..., δyn
    
    返回:
    tuple: (加权平均值, 加权平均值的不确定度)
    """
    # 转换为numpy数组
    data = np.array(data_points)
    unc = np.array(uncertainties)
    
    # 检查输入
    if data.shape != unc.shape:
        raise ValueError("数据点和不确定度的数组长度必须相同")
    
    if len(data) == 0:
        raise ValueError("输入数组不能为空")
    
    # 检查不确定度是否为零或负数
    if np.any(unc <= 0):
        raise ValueError("不确定度必须是正数")
    
    # 计算权重：w = 1/δy²
    weights = 1.0 / (unc ** 2)
    
    # 计算加权平均值：μ = Σ(wi*yi) / Σ(wi)
    weighted_mean = np.sum(weights * data) / np.sum(weights)
    
    # 计算加权平均值的不确定度：σ_μ = 1/√(Σ(wi))
    weighted_uncertainty = 1.0 / np.sqrt(np.sum(weights))
    
    return weighted_mean, weighted_uncertainty


def parse_filename(filename):
    name = filename.rstrip('.csv')
    result = {}

    # 1. t 值
    t_match = re.search(r'_t([0-9]*\.?[0-9]+)', name)
    if t_match:
        t_val = float(t_match.group(1))
        result['t'] = int(t_val) if t_val.is_integer() else t_val

    # 2. U 或 V
    uv_match = re.search(r'([UV])([0-9]*\.?[0-9]+)(?:_([0-9]*\.?[0-9]+))?', name)
    if uv_match:
        prefix = uv_match.group(1)  # 'U' or 'V'
        val1 = float(uv_match.group(2))
        val2 = uv_match.group(3)
        if val2 is not None:
            val2 = float(val2)
            v1 = int(val1) if val1.is_integer() else val1
            v2 = int(val2) if val2.is_integer() else val2
            result[prefix] = [v1, v2]
        else:
            v1 = int(val1) if val1.is_integer() else val1
            result[prefix] = v1

    # 3. size：支持任意数量的整数
    size_match = re.search(r'size\[(.*?)\]', name)
    if size_match:
        inner = size_match.group(1)
        if inner.strip():  # 非空
            nums = [int(x.strip()) for x in inner.split(',')]
            result['size'] = nums
        else:
            result['size'] = []

    # 4. Δt
    dt_match = re.search(r'Δt([0-9]*\.?[0-9]+)', name)
    if dt_match:
        result['Δt'] = float(dt_match.group(1))

    # 5. Θ
    theta_match = re.search(r'Θ([0-9]*\.?[0-9]+)(?:_([0-9]*\.?[0-9]+))?', name)
    if theta_match:
        th1 = float(theta_match.group(1))
        th2 = theta_match.group(2)
        if th2 is not None:
            th2 = float(th2)
            result['Θ'] = [th1, th2]
        else:
            result['Θ'] = th1

    # 6. N
    n_match = re.search(r'N(\d+)', name)
    if n_match:
        result['N'] = int(n_match.group(1))

    # 7. BS
    bs_match = re.search(r'BS(\d+)', name)
    if bs_match:
        result['BS'] = int(bs_match.group(1))

    return result

def linear_fit(x,y,dy):
    A=np.ones([len(x),2]) 
    A[:,0]=x
    
    para=np.linalg.inv(A.conj().T@A)@A.conj().T@y

    if dy is False:
        yy=para[0] * x + para[1]
        dslope=np.sqrt( np.sum((yy-y)**2)/(len(x)-2)/np.sum((x-np.mean(x))**2) )
    else:
        
        xbar=np.sum(x/dy**2)/np.sum(1/dy**2)
        dslope=1/np.sqrt(  np.sum((x - xbar)**2 / dy**2)  )

    return para, dslope 

def powerfit(x,y):
    initial_guess = [0.3, 1.0, 1.0]

    def model_func(x, a, b, c):
        return a + b * x**(c)
    
    params, _ = curve_fit(
        model_func, x, y,
        p0=initial_guess,
        # bounds=None,
        maxfev=100000
    )
    return params

def sweep_stable(period, file):
    filename = os.path.basename(file)              # 'DP_Lib.py'

    data = []
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
            
    data=combine_data(data)
    vals = data[:,1:].astype(float)

    N = data.shape[1] // period
    
    S2 = np.zeros(N)
    dS = np.zeros(N)

    for i in range(N):
        subEE = np.mean(vals[:, 0:(i+1)*period], axis=1)
        dsubEE = np.std(vals[:, 0:(i+1)*period], axis=1, ddof=1) / np.sqrt((i+1)*period)
        S2[i] = -np.log(np.prod(subEE))
        dS[i] = np.sqrt(np.sum((dsubEE/subEE)**2))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=200)
    ax[0].errorbar(range(1, N+1), S2, yerr=dS, fmt='o-', markersize=2, color='blue', ecolor='black', capsize=5)
    ax[0].set_xlabel(f'Sweeps (unit: {period})')
    ax[0].set_ylabel(r'$\Delta S_2$')

    params= parse_filename(file)

    text = f'Final: {S2[-1]:.4f} ± {dS[-1]:.4f}'
    ax[0].text(0.98, 0.02, text,
            transform=ax[0].transAxes,  # 轴坐标(0~1)
            ha='right', va='bottom',
            fontsize=8,
            bbox=dict(boxstyle='round', fc='w', alpha=0.7))

    ax[1].plot(range(1, N+1), dS, 'o-', markersize=2, color='red', label='Error')
    ax[1].set_xlabel(f'Sweeps (unit: {period})')
    ax[1].set_ylabel('Error bar')


    if 'U' in params:
        ax[0].set_title(f"U={params['U']}, L={params['size']}, θ={params['Θ']}, N={params['N']:.0f}, BS={params['BS']:.0f}")
        ax[1].set_title(f"U={params['U']}, L={params['size']}, θ={params['Θ']}, N={params['N']:.0f}, BS={params['BS']:.0f}")
    else:
        ax[0].set_title(f"V={params['V']}, L={params['size']}, θ={params['Θ']}, N={params['N']:.0f}, BS={params['BS']:.0f}")
        ax[1].set_title(f"V={params['V']}, L={params['size']}, θ={params['Θ']}, N={params['N']:.0f}, BS={params['BS']:.0f}")
        # fig.savefig(os.path.join("C:/Users/admin/Desktop/",f"U={U}L={L}θ={t}N={params[-2]}.png"))
    return fig

def File_Process2(file,folder_path,throw=0,subEEfig=False):
    full_path = os.path.join(folder_path, file)
    
    data = []
    with open(full_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
            
    data=combine_data(data,throw)

    λ = data[:,0]
    # 确保为浮点数并计算样本数 n（data 的第一列是标签）
    vals = data[:,1:].astype(float)
    subEE = np.mean(vals, axis=1)
    n = vals.shape[1]
    # 对样本数太小的情况做保护（n<=1 时无法计算样本标准差）
    if n <= 1:
        # 无法计算 SEM，返回 NaN 向量以便上层捕获或忽略
        dsubEE = np.full(subEE.shape, np.nan)
    else:
        # 使用样本标准差（ddof=1），标准误 = s / sqrt(n)
        s = np.std(vals, axis=1, ddof=1)
        dsubEE = s / np.sqrt(n)

    # 排序处理
    sorted_indices = np.argsort(λ)

    λ = λ[sorted_indices]
    subEE = subEE[sorted_indices]
    dsubEE = dsubEE[sorted_indices]

    EE=-np.log(np.prod(subEE))
    dEE=np.sqrt(np.sum((dsubEE/subEE)**2))

    # 绘图逻辑
    if subEEfig:
        params=parse_filename(file)
        fig=plt.figure()
        plt.errorbar(λ, subEE, yerr=dsubEE, fmt='o-', markersize=5, color='blue', ecolor='red', capsize=5)
        plt.xlim(-0.2, 1)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$EE_{icr}$')
        plt.title(f'$\Delta S_2={EE:.4f} \pm {dEE:.4f}$')

        figpath=os.path.join(folder_path,"fig")
        if not os.path.exists(figpath):
            os.makedirs(figpath)
            print(f"Created directory: {figpath}")
        if 'U' in params:
            fig.savefig(os.path.join(figpath,f"U={params['U']}L={params['size']}θ={params['Θ']}N={params['N']}subEE.png"))
        if 'V' in params:
            fig.savefig(os.path.join(figpath,f"V={params['V']}L={params['size']}θ={params['Θ']}N={params['N']}subEE.png"))
        plt.close()

    return EE,dEE

def File_Process(folder_paths, throw=0, subEEfig=False):
    Data = pd.DataFrame([ ], columns=['t', 'L', 'EE', 'dEE'])
    
    for count, folder in enumerate(os.listdir(folder_paths)):
        folder_path = os.path.join(folder_paths, folder)
        if not os.path.isdir(folder_path):
            continue

        U,data=File_Process1(folder_path,throw,subEEfig)
        Data = pd.concat([Data, data], ignore_index=True)

    Data.to_csv(os.path.join(folder_paths, "datafit.csv"), index=False)

    return U,Data

def File_Process1(folder_path,throw=0,subEEfig=False):
    Data = []

    for file in os.listdir(folder_path):
        if not file.endswith(".csv") or "EE" not in file:
            continue
        print(file)

        params=parse_filename(file)
        L=params['size'][0] if 'size' in params and len(params['size'])>0 else None
        if isinstance(params['Θ'], list):
            t=params['Θ'][-1]
        else:   
            t=params['Θ']
        
        EE,dEE=File_Process2(file,folder_path,throw,subEEfig)
        Data.append([t, L, EE , dEE])

    Data = pd.DataFrame(Data, columns=['t', 'L', 'EE', 'dEE'])
    U=params['U'] if 'U' in params else params['V']
    return U,Data

# def extract_numbers(s):
#     """从字符串提取所有数值（支持负数和小数）"""
#     return [float(num) for num in re.findall(r'-?\d+\.?\d*', s)]

def trim_extreme_in_csv(file_path, mode='min', inplace=True, output_path=None, n=None):
    """
    按首列数值分组，将每组内“从第二个数起”的所有数据视为一类，
    对每一类找到其中的 n 个极值（最小或最大），并将这些极值替换为该类的均值，
    最终覆盖写回原文件（或写入指定输出）。

    参数:
    - file_path: 输入 CSV 文件路径。
    - mode: 'min' 处理最小值，'max' 处理最大值。
    - inplace: True 时原地覆盖写回；False 时写入 output_path。
    - output_path: 非原地模式时的输出路径；未提供则在同目录生成带 _trim 后缀的文件。
    - n: 每“类别”替换的极值个数，必须为非负整数（0 表示不替换）。

    返回:
    - 写出文件的路径。
    """
    if mode not in ('min', 'max'):
        raise ValueError("mode 必须为 'min' 或 'max'")
    if n is None:
        raise ValueError("n 参数为必填，且必须为非负整数")
    if not isinstance(n, int) or n < 0:
        raise ValueError("n 必须为非负整数")

    # 读入所有行
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))

    # 解析标签：尽量按数值分组，失败则按原字符串
    def parse_label(cell):
        if cell is None:
            return ''
        s = str(cell).strip()
        if s == '':
            return ''
        try:
            return float(s)
        except (ValueError, TypeError):
            return s

    # 构建：标签 -> 该标签的行索引列表
    groups = {}
    for i, row in enumerate(rows):
        if not row or all(str(c).strip() == '' for c in row):
            continue
        label_key = parse_label(row[0]) if len(row) >= 1 else ''
        groups.setdefault(label_key, []).append(i)

    # 对每个组，用均值替换 n 个极值
    for label_key, idx_list in groups.items():
        # 收集该组内所有可比较的数值项: (row_index, col_index, value)
        def collect_numeric_entries():
            entries = []
            for ri in idx_list:
                row = rows[ri]
                # 从第二列开始
                for cj in range(1, len(row)):
                    try:
                        v = float(row[cj])
                    except (ValueError, TypeError):
                        continue
                    entries.append((ri, cj, v))
            return entries

        entries = collect_numeric_entries()
        if not entries:
            continue

        # 循环替换 n 次（若不足 n，则尽可能替换）
        replaced_positions = set()
        for _ in range(n):
            if not entries:
                break
            # 每次替换前重新计算均值（基于当前已更新的数据）
            group_mean = float(np.mean([e[2] for e in entries]))

            # 计算当前极值（跳过已替换位置）
            candidates = [e for e in entries if (e[0], e[1]) not in replaced_positions]
            if not candidates:
                break

            if mode == 'min':
                target_val = min(e[2] for e in candidates)
            else:
                target_val = max(e[2] for e in candidates)

            # 找到第一个等于极值的项目并替换为均值
            replace_entry = None
            for idx, e in enumerate(entries):
                if (e[0], e[1]) in replaced_positions:
                    continue
                if e[2] == target_val:
                    replace_entry = (idx, e)
                    break
            if replace_entry is None:
                break

            replace_idx, (ri, cj, _) = replace_entry
            row = rows[ri]
            if 0 <= cj < len(row):
                row[cj] = str(group_mean)

            # 更新条目值并标记已替换，避免重复替换同一位置
            entries[replace_idx] = (ri, cj, group_mean)
            replaced_positions.add((ri, cj))

    # 计算输出路径
    if inplace:
        out_path = file_path
    else:
        if output_path:
            out_path = output_path
        else:
            base, ext = os.path.splitext(file_path)
            out_path = f"{base}_trim{ext or '.csv'}"

    # 写回文件
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)

    return out_path

def combine_data(matrix,throw=0):
    """合并数据表，按标签聚合数据"""
    data_dict = {}
    
    for row in matrix:
        if len(row)==0:
            continue
        label = row[0]   # 保留三位小数
        values = row[throw+1:]
        
        if label not in data_dict:
            data_dict[label] = []
        data_dict[label].extend(values)
    
    # # 构建结果矩阵
    labels = sorted(data_dict.keys())
    max_len = min(len(v) for v in data_dict.values())
    
    combined = np.empty([len(data_dict),max_len+1])
    for index, (key, value) in enumerate(data_dict.items()):
        combined[index,0]=key
        combined[index,1:]=value[:max_len]
    
    return combined

# 数据处理函数
def unit_neardata(data):
    units = []
    for i in range(len(data)):
        neighbors = [j for j in range(len(data)) 
            if max(data[i][1]-data[i][2],data[j][1]-data[j][2]) <= min(data[i][1]+data[i][2],data[j][1]+data[j][2])  ] 
            # if (data[j][1]-data[j][2] <= data[i][1]-data[i][2]<= data[j][1]+data[j][2] or data[j][1]-data[j][2] <= data[i][1]+data[i][2]<= data[j][1]+data[j][2] or
            #     data[i][1]-data[i][2] <= data[j][1]-data[j][2] <= data[i][1]+data[i][2]  or data[i][1]-data[i][2] <= data[j][1]+data[j][2] <= data[i][1]+data[i][2] )] 
        units.append(neighbors)
    LLL=[len(x) for x in units]
    return units[np.where(LLL==np.max(LLL))[0][-1] ]

def phy_PD(folder_path):
    Data = []
    L = []
    t = []
    V = []
    EK , dEK , EV , dEV , R0 , dR0 , R1 , dR1 = [],[],[],[],[],[],[],[] 

    for file in os.listdir(folder_path):
        if not file.endswith(".csv") or "phy" not in file:
            continue
        print(file)

        params = parse_filename(file)
        V.append(params['V'] if 'V' in params else params['U'])
        L.append(params['size'][0] if 'size' in params and len(params['size'])>0 else None)
        t.append(params['Θ'][-1] if isinstance(params['Θ'], list) else params['Θ'])

        full_path = os.path.join(folder_path, file)
        data = np.array(pd.read_csv(full_path, header=None))
        EK.append( np.mean(data[:,0]) )
        EV.append( np.mean(data[:,1]) )
        dEK.append( sem(data[:,0]) )
        dEV.append( sem(data[:,1]) )

        R = np.mean(data[:,2:6],axis=0)
        dR = sem(data[:,2:6],axis=0)
        Rdelta = np.mean(data[:,6:10],axis=0)
        dRdelta = sem(data[:,6:10],axis=0)

        # R0.append(R[0]+R[1])
        R0.append(R[0]+R[1]-R[2]-R[3])
        dR0.append( np.sqrt(np.sum(dR**2)) )
        # R1.append(Rdelta[0]+Rdelta[1])
        R1.append(Rdelta[0]+Rdelta[1]-Rdelta[2]-Rdelta[3])
        dR1.append( np.sqrt(np.sum(dRdelta**2)) )
        Data.append([V[-1],t[-1], L[-1], EK[-1] , dEK[-1],EV[-1] , dEV[-1] , R0[-1] , dR0[-1] , R1[-1] , dR1[-1] ])

        if np.any(np.isnan(Data[-1])):
            raise ValueError("NaN value found in the file: " + file)

    Data = pd.DataFrame(Data, columns=['V','t', 'L', 'EK', 'dEK', 'EV', 'dEV', 'R0', 'dR0', 'R1', 'dR1'])
    return t[1]/L[1],Data
       
def EEfit_F(x, y, fix, arg):
    # y=a*x*lnx + b*x + c*log(x) + d
    if fix == 0:
        A=np.ones((len(x),4))
        A[:,0]=x*np.log(x)
        A[:,1]=x
        A[:,2]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y
    
    elif fix == 1:
        y_adj = y - arg * x * np.log(x)
        A=np.ones((len(x),3))
        A[:,0]=x
        A[:,1]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    
    elif fix == 2:
        y_adj = y - arg * x
        A=np.ones((len(x),3))
        A[:,0]=x*np.log(x)
        A[:,1]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj

    elif fix == 3:
        y_adj = y - arg * np.log(x)
        A=np.ones((len(x),3))
        A[:,0]=x*np.log(x)
        A[:,1]=x
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    elif fix == [1,3]:
        y_adj = y - arg[0] * x * np.log(x) - arg[1] * np.log(x)
        A=np.ones((len(x),2))
        A[:,0]=x
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    elif fix == [2,3]:
        y_adj = y - arg[0] * x - arg[1] * np.log(x)
        A=np.ones((len(x),2))
        A[:,0]=x*np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    elif fix == [1,2]:
        y_adj = y - arg[0] * x *np.log(x) - arg[1] * x
        A=np.ones((len(x),2))
        A[:,0]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj

def QuenchFile_Process(folder_paths, throw=0,subEEfig=False):
    Data = pd.DataFrame([ ], columns=['t','R', 'L', 'EE', 'dEE'])
    
    for count, folder in enumerate(os.listdir(folder_paths)):
        folder_path = os.path.join(folder_paths, folder)
        if not os.path.isdir(folder_path):
            continue

        data=QuenchFile_Process1(folder_path,throw,subEEfig)
        Data = pd.concat([Data, data], ignore_index=True)

    Data.to_csv(os.path.join(folder_paths, "datafit.csv"), index=False)

    return Data

def QuenchFile_Process1(folder_path,throw=0,subEEfig=False):
    Data = []

    for file in os.listdir(folder_path):
        if not file.endswith(".csv") or "EE" not in file:
            continue
        print(file)

        params = parse_filename(file)
        dU=params['U'][1]-params['U'][0] if 'U' in params else params['V'][1]-params['V'][0]
        L=params['size'][0] if 'size' in params and len(params['size'])>0 else None
        t=params['Θ'][-1] if isinstance(params['Θ'], list) else params['Θ']
        R=np.abs(dU)/t

        
        EE,dEE=File_Process2(file,folder_path,throw,subEEfig)
        Data.append([t,R, L, EE , dEE])

    Data = pd.DataFrame(Data, columns=['t','R', 'L', 'EE', 'dEE'])
    return Data