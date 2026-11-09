"""Audited beta-zero section for the existing RESULTS.md generator."""
from pathlib import Path


def build(report):
    root = report.ROOT
    out = root / "analysis_beta0_study"
    summary = report.read_json(out / "study_summary.json")
    data = report.read_json(out / "final_data_validation.json")
    fit = report.read_json(out / "fit_validation.json")
    for receipt in (summary, data, fit):
        assert receipt["status"] == "PASS" and receipt["stage"] == "final"
        for name, checksum in receipt["files"].items():
            report.verify(root / name, checksum)
    assert summary["counts"]["missing_required_keys"] == 0
    assert all(x == "complete" for x in summary["source_run_status"].values())
    assert data["new_rows"] == 530 and data["target_rows"] == 950
    assert fit["decimal_digits"] >= 80
    assert fit["counts"]["confirmation_patterns"] == 10
    assert fit["counts"]["pilot_only_patterns"] == 6
    design = report.read_json(out / "study_design.json")
    plan = report.read_json(out / "confirmation_plan.json")
    physics = report.read_json(out / "physics_summary.json")
    assert physics["status"] == "PASS"
    for name, checksum in physics["files"].items():
        report.verify(root / name, checksum)
    complement = report.read_json(out / "complement_audit.json")
    assert complement["status"] == "PASS" and complement["checked_distinct_pairs"] == 136
    for name, checksum in complement["files"].items():
        report.verify(root / name, checksum)
    zero_mode = report.read_json(out / "zero_mode_bound.json")
    assert zero_mode["status"] == "PASS"
    for name, checksum in zero_mode["files"].items():
        report.verify(root / name, checksum)
    reference = report.read_toml(root / "data/beta0_study_validation.toml")
    assert reference["status"] == "PASS" and reference["states"] == 99
    rows = report.read_csv(out / "primary_coefficients.csv")
    names = design["patterns"]
    assert len(rows) == len(names) == 16
    by_name = {r["pattern"]: r for r in rows}
    assert set(by_name) == set(names)
    selected = set(plan["patterns"])
    assert selected == set(summary["confirmation_patterns"])
    metadata = {r["name"]: r for r in physics["patterns"]}
    report.verify(Path(__file__).resolve(), report.digest(Path(__file__).resolve()))

    def yes(row, key):
        return row.get(key, "").lower() == "true"

    def value(row, key):
        return report.number(row, key)

    def interval(row, prefix):
        return f"[{report.fmt(value(row, prefix + '_min'))}, {report.fmt(value(row, prefix + '_max'))}]"

    def linked(name):
        return f"[`{name}`](#beta0-{name.replace('_', '-')})"

    def group(evidence):
        return [name for name in names if by_name[name]["evidence_class"] == evidence]

    supported = group("finite_window_numerical_support")
    comparison_unresolved = group("time_checks_pass_comparison_unresolved")
    time_unresolved = group("confirmation_time_side_unresolved")
    pilot_only = group("pilot_only_not_sufficiently_confirmed")
    assert len(supported + comparison_unresolved + time_unresolved + pilot_only) == 16
    assert len(pilot_only) == 6
    assert set(supported) == {"uniform_A", "one_B_2", "one_B_3", "special_matched_a", "general_matched_a"}, "Refresh the interpreted cohort paragraph if frozen results change"
    labels = {
        "finite_window_numerical_support": "有限窗口支持一致",
        "time_checks_pass_comparison_unresolved": "时间检查通过，两端比较未定",
        "confirmation_time_side_unresolved": "确认网格已齐，时间侧未稳定",
        "pilot_only_not_sufficiently_confirmed": "仅探索，缺大尺寸确认",
    }
    parts = [
        "本节专门回答：**β=0 时，哪些初态能用非平衡方法提取与平衡端一致的角系数？** "
        "β=0 时没有弱测量，构型只影响初态和它选中的零模基态。这里重新整理了全部旧 β=0 观测，"
        "对 16 个可达候选补了小尺寸密集时间扫描和 L=180、216 的平衡端，再挑选 10 个代表作大尺寸时间确认。"
        "以下结论取代旧的 4–7 窗口在 β=0 上的初步判断；正 β 的旧扫描另列在后文。",
        r"""采用周期 HoneyComb120、半填充、单种自由费米子的自然对数 Rényi-2 熵；分别计算两个无角条带和一个有角区域：

$$
A=(L/3)\times L,\qquad C=L\times(L/3),\qquad B=(L/3)\times(2L/3),
\qquad D=\frac{2S_2(A)+S_2(C)}3-S_2(B).
$$

下文系数是 D 的对数项系数，对应 B 的两个 60° 角与两个 120° 角之和。""",
        "### 结论与适用范围",
        ("**通过本次时间侧独立检查、且与平衡端数值相容的构型：** "
         + "、".join(linked(n) for n in supported) + "。"
         if supported else
         "**本次没有构型同时通过预定的全部检查；不能仅凭两个主拟合系数接近就宣布一致。**"),
        "这里的“相容”要求：时间侧先通过尺寸、移窗、修正阶数检查；平衡侧对尺寸下限和修正阶数稳定；"
        "随后才比较两端主系数，要求相对差不超过 5%，且模型／窗口敏感性区间相交。"
        "这是已计算有限窗口内的数值支持，**不是系数严格相等的数学证明，也不等于原始曲线已经成为无修正的直线。**",
        "**β=0 的平衡端本身具有更强的结论：本模型各可达初态得到的精确 Slater 基态若存在渐近角对数系数，该系数相同。** "
        "因为不同基态只在有限个零模上有区别，任意区域的熵差及 D 的差均有不随 L 增长的上界。"
        "下表不同平衡主拟合值的少量偏差应视为有限尺寸外推问题，不能直接解释为不同的普适角系数；"
        "敏感性区间也不是渐近误差的严格上界。[证明与适用条件](SCALING_GUIDE.md#beta0-initial-state)。",
    ]
    if "uniform_A" in supported:
        parts.append("**uniform B 与 uniform A 的区域 Rényi 熵在 β=0 下严格相同，因此也得到同样支持。** "
                     "更一般地，任一已确认构型的逐格点空／占互补构型也有完全相同的熵曲线。"
                     "这一点来自手征对称性的恒等式，不需要另挑拟合窗口，推导见[指南](SCALING_GUIDE.md#beta0-initial-state)。")
    if comparison_unresolved:
        parts.append("时间侧通过、但平衡稳定性或两端比较仍未满足标准："
                     + "、".join(linked(n) for n in comparison_unresolved) + "。")
    if time_unresolved:
        parts.append("完成大尺寸确认、但当前时间窗仍未稳定："
                     + "、".join(linked(n) for n in time_unresolved)
                     + "。这些结果说明当前窗口不能可靠提取目标系数，不能据此断言其渐近系数一定不同。")
    parts.append("仅完成本次探索网格、未补齐统一大尺寸确认的六个构型："
                 + "、".join(linked(n) for n in pilot_only)
                 + "。它们仍保留全部原始数据及可比时间范围的图，**缺少确认数据不记作物理上的失败，也不列作反例。**")
    parts.append("### 16 个构型的统一比较")
    table = ["| 构型 | 类别 / 超胞 | 平衡 c | 非平衡 c | 两端相对差 | 本次结论 |",
             "|---|---|---:|---:|---:|---|"]
    for name in names:
        row, meta = by_name[name], metadata[name]
        neq = report.fmt(value(row, "c_time_fit")) if name in selected else "—"
        gap = report.pct(value(row, "relative_time_eq_gap")) if name in selected else "—"
        table.append(f"| {linked(name)} | {meta['class_id']} / {meta['m']}×{meta['m']} | "
                     f"{report.fmt(value(row, 'c_eq_fit'))} | {neq} | {gap} | {labels[row['evidence_class']]} |")
    parts.append("\n".join(table))
    parts.append("表中非平衡数值统一指 L=216、θ=4–9 的四参数主拟合；未稳定的参数保留用于诊断，"
                 "不能直接称为渐近角系数。六个仅探索构型不填统一主时间系数。"
                 "类别 1 为每原胞单占，类别 2 为纯空／双占，类别 3 为混合。"
                 "全部构型示意图见[指南构型索引](SCALING_GUIDE.md#patterns)。")
    parts.append("本轮支持名单包含第一类的 uniform A、one_B_2、one_B_3、special_matched_a，"
                 "以及第三类的 general_matched_a。因此，**不能只凭单占原胞就认定当前窗口已稳定；混合构型也能获得数值支持。** "
                 "第一类的 diagonal_B_3 仍未通过全部时间检查；第二类两个完成确认的例子也未通过，"
                 "但另外两个第二类例子只完成探索，不能把这一结果推广成整类禁止。")
    parts.append(report.picture("figs_beta0_study", "coefficient_comparison",
                                "β=0：10 个大尺寸确认构型的两端系数及模型／窗口敏感性范围"))
    parts.append("每个小图单独设置纵轴，便于看到该构型两端的差异，跨小图比较时须按各自纵轴标尺读取数值。"
                 "蓝色空心圆表示平衡端，橙色方块表示非平衡端；区间来自模型与窗口变化，"
                 "不是统计误差棒。橙色空心标记表示时间侧未稳定，即使两端位置接近也不算通过；实心仅代表时间检查通过，两端是否相容还须看表中结论。")
    parts.append("### 这次怎样避免拟合出想要的答案")
    parts.append(r"""先冻结探索网格，完整得到 L=72、108 上 θ=1,1.5,…,10 的时间数据；根据时间侧覆盖和重叠诊断选定 10 个确认构型，再冻结确认网格。选取不使用与平衡系数的接近程度。两份计划及选取理由均保存。

| 检查 | 本次实际设置 |
|---|---|
| 平衡主拟合 | L=36–216，使用每个构型实际相容的尺寸网格；对数、常数、1/L 三项 |
| 平衡敏感性 | 改变下限：4×4 使用 36、48、72，其余使用 36、54、72；另加 1/L² 修正 |
| 时间主比较 | L=180、216，θ=4,4.5,…,9，共 11 个同点；主系数取 L=216 |
| 第三尺寸 | L=144、180、216，在 θ=4,4.5,…,7 的共同网格检查尺寸漂移是否递减 |
| 移窗 | 4–7、4–8、4–9、4.5–9、4–8.5、5–9、5–10；最后一个只在 L=216 作完整移窗检查 |
| 修正项 | 比较 1/θ、1/θ+1/θ²、1/θ+1/θ²+1/θ³；三阶模型至少 11 点才纳入正式阶数比较 |
| 时间尺度 | 拟合点须满足 θ≥4、θ/L≤0.05；探索扫描中不满足的点仅用于观察交叉和有限尺寸效应 |
| 时间侧门槛 | 主系数尺寸漂移≤2%；同点 D 差／拟合对数变化≤1%；移窗和阶数漂移≤5%；无变号、覆盖完整；第三尺寸检查中，180→216 的漂移≤2%且小于 144→180 的漂移 |

$$
D_{\rm eq}(L)=c_{\rm eq}\ln L+b+\frac dL,
\qquad
D_{\rm neq}(\theta)=c_{\rm neq}\ln\theta+b+\frac d\theta+\frac e{\theta^2}.
$$

θ/L 的数值预筛不是尺度分离定理：区域最窄宽度为 L/3，还要考虑速度。主窗口虽然由 7 点扩大为 11 点，仍只有 2.25 倍的时间跨度；因此同时展示原始相邻点对数斜率和同 θ 的尺寸差，不能仅看修正拟合线与数据贴合。**没有强制非平衡系数等于平衡值，也没有用文献值约束拟合。**""")
    diagnostic = ["| 确认构型 | 尺寸漂移 | 同点 D 差 / 对数变化 | 阶数漂移 | 移窗漂移 | 平衡敏感性 |",
                  "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        if name not in selected:
            continue
        row = by_name[name]
        keys = ("size_drift", "D_difference_to_fitted_log_span", "model_drift", "window_drift", "equilibrium_sensitivity")
        diagnostic.append("| " + linked(name) + " | " + " | ".join(report.pct(value(row, k)) for k in keys) + " |")
    parts.append("\n".join(diagnostic))
    r = by_name["random_half_3"]
    d = by_name["diagonal_B_3"]
    n = by_name["pure02_notched_stripe4"]
    parts.append("两个值得单独看的例子：random_half_3 的两端主值仅差 "
                 + report.pct(abs(value(r, "relative_time_eq_gap")))
                 + "，但时间系数的尺寸漂移为 " + report.pct(value(r, "size_drift"))
                 + "、阶数漂移为 " + report.pct(value(r, "model_drift"))
                 + "，所以不能凭主值接近就确认；缺口条带的两端差为 "
                 + report.pct(abs(value(n, "relative_time_eq_gap")))
                 + "，其阶数漂移却达到 " + report.pct(value(n, "model_drift"))
                 + "。diagonal_B_3 则属于接近门槛的情况，主要问题是阶数漂移 "
                 + report.pct(value(d, "model_drift"))
                 + " 略高于预定 5%，不能把这个操作判定解读为突然出现了不同的物理类别。")
    parts.append("### 为什么不能单看占据分类或基态可达性")
    parts.append("能投影到基态只回答无限虚时极限是否落在基态子空间；它不控制进入临界时间区需要多久。"
                 "不同构型在折叠动量空间的最小重叠可以差很多，有能隙模式也可能因初始重叠很小而衰减得很晚。"
                 "本次独立诊断还检查了 Dirac 附近的有效边界子空间：它由整个超胞决定，不能用 A/B 总数或空／双占计数替代。"
                 "详细推导、数值扫描及它们的局限放在[β=0 初态判断指南](SCALING_GUIDE.md#beta0-initial-state)。"
                 "**目前不能把“可达基态”直接升级成“当前 θ 窗口必有相同角系数”，也没有把未稳定构型认定为渐近反例。**")
    parts.append("### 全部左右 scaling 图与时间诊断")
    parts.append("每张图左侧为原始 D 对 ln L，蓝色实线为平衡拟合；右侧为原始 D 对 ln θ，彩色实线连接原始观测，黑色虚线为时间拟合。拟合线只绘制在所标窗口内。"
                 "确认组统一使用上述大尺寸网格；仅探索组明确标注 pilot，其右侧保留已有尺寸的原始曲线，并标明缺少统一的大尺寸拟合，不能与确认组等同。"
                 "各构型下的时间诊断图显示不依赖平衡目标的相邻对数斜率与同时间尺寸差。诊断右图的尺寸差按对应尺寸对、所标窗口的大尺寸原始 D 跨度归一化，各尺寸对分母分别计算。")
    for name in names:
        row = by_name[name]
        parts.append(f"<a id=\"beta0-{name.replace('_', '-')}\"></a>\n#### {name}")
        parts.append(f"**{labels[row['evidence_class']]}。** 平衡敏感性范围："
                     + interval(row, "eq_sensitivity")
                     + ("；时间敏感性范围：" + interval(row, "time_sensitivity") + "。" if name in selected else "。"))
        parts.append(report.picture("figs_beta0_study", name + "_scaling", name + "：β=0 平衡与非平衡 scaling"))
        # Track both diagnostic files in the receipt without rendering another 16 large images.
        for ext in ("png", "pdf"):
            path = root / "figs_beta0_study" / f"{name}_time_diagnostics.{ext}"
            assert path.is_file(), path
            report.USED.add(path)
        parts.append(f"[原始斜率与尺寸差 PNG](figs_beta0_study/{name}_time_diagnostics.png) · "
                     f"[PDF](figs_beta0_study/{name}_time_diagnostics.pdf)")
    parts.append("### 数据来源与独立复核")
    parts.append(f"本次实际新增 **{data['new_rows']} 个观测**；两份冻结计划合计 "
                 f"**{data['target_rows']} 个参数键全部齐全**，其中 {data['reused_rows']} 个复用已有数据。"
                 f"包括旧库后共有 {summary['counts']['pooled_points']} 个唯一 β=0 点，逐点来源、代码快照和计划哈希均保留。"
                 f"独立实空间实现核验了 99 个小系统状态，D 的最大差为 {reference['max_D_error']:.3g}；"
                 f"最终另外用 80 位 QR 重算 {fit['counts']['unique_QR_fits']} 组拟合并检查原始斜率、实际网格和判定标签。"
                 "审计 PASS 表示计算与报告可复核，不表示全部构型都具有稳定 scaling。")
    parts.append("[两端主系数](analysis_beta0_study/primary_coefficients.csv) · "
                 "[合并观测](analysis_beta0_study/study_observations.csv) · "
                 "[探索计划](analysis_beta0_study/study_design.json) · "
                 "[确认计划及选择理由](analysis_beta0_study/confirmation_plan.json) · "
                 "[数据核验](analysis_beta0_study/final_data_validation.json) · "
                 "[独立拟合核验](analysis_beta0_study/fit_validation.json) · "
                 "[物理诊断](analysis_beta0_study/physics_summary.json)。")
    parts.append("从仓库根目录重做分析与更新本节（读取已完成观测，不重新跑 Julia）：\n\n"
                 "```powershell\n"
                 "python src/Measure/ai/validate_beta0_study_completion.py\n"
                 "python src/Measure/ai/analyze_beta0_study.py --stage final --plots\n"
                 "python src/Measure/ai/validate_beta0_study_fits.py\n"
                 "python src/Measure/ai/write_results_report.py --sections BETA0_STUDY\n"
                 "```\n\n"
                 "原始计算的两份冻结任务表和运行参数可按下列命令使用；已有参数键会跳过。"
                 "逐点记录见[探索运行日志](data/beta0_study_pilot.log)、"
                 "[确认运行日志](data/beta0_study_confirmation.log)，代码快照和精确参数见同目录 run TOML。\n\n"
                 "```powershell\n"
                 "julia --startup-file=no src/Measure/ai/beta0_study_experiments.jl --mode=run "
                 "--backend=openblas --threads=8 --strip-threads=1 --require-ground=true "
                 "--job-file=src/Measure/ai/analysis_beta0_study/pilot_jobs.csv "
                 "--plan=src/Measure/ai/analysis_beta0_study/study_design.json\n"
                 "julia --startup-file=no src/Measure/ai/beta0_study_experiments.jl --mode=run "
                 "--backend=openblas --threads=8 --strip-threads=1 --require-ground=true "
                 "--job-file=src/Measure/ai/analysis_beta0_study/confirmation_jobs.csv "
                 "--plan=src/Measure/ai/analysis_beta0_study/confirmation_plan.json\n"
                 "```\n\n"
                 "两份任务表只列本次需要新增的点，复用的旧点及其来源保存在相邻 reused CSV 中；"
                 "完整复现还须保留这些旧源文件。")
    return "\n\n".join(parts)
