"""Refresh audited tables/figures in RESULTS.md; never create separate reports.

The prose outside GENERATED blocks is maintained by hand. This command reads
completed analyses, verifies their receipts, and does not rerun simulations.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import io
import json
import math
import re
import tomllib
import sys

ROOT = Path(__file__).resolve().parent
USED = set()
VERIFIED = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    USED.add(path)
    VERIFIED[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def read_csv(path):
    raw = path.read_bytes()
    USED.add(path)
    VERIFIED[path] = hashlib.sha256(raw).hexdigest()
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def read_toml(path):
    raw = path.read_bytes()
    USED.add(path)
    VERIFIED[path] = hashlib.sha256(raw).hexdigest()
    return tomllib.loads(raw.decode("utf-8-sig"))


def verify(path, expected):
    path = Path(path)
    assert digest(path) == expected, f"Hash mismatch: {path}"
    USED.add(path)
    VERIFIED[path] = expected


def audited_rows(name, expected_rows):
    out = ROOT / f"analysis_{name}"
    summary = read_json(out / "summary.json")
    data = read_json(out / "independent_data_validation.json")
    fit = read_json(out / "independent_fit_validation.json")
    assert summary["stage"] == "final" and summary["required_grid_complete"] and summary["primary_complete"]
    assert all(x["status"] == "PASS" and x["stage"] == "final" for x in (data, fit))
    verify(out / "summary.json", fit["primary_summary_sha256"])
    verify(ROOT / f"analyze_{name}.py", summary["script_sha256"])
    verify(out / "plan.json", summary["plan_sha256"])
    for item in summary.get("dependencies", []):
        verify(Path(item["path"]), item["sha256"])
    for item in fit["source_receipts"]:
        if not item.get("missing"):
            verify(Path(item["path"]), item["source_sha256"])
    for filename, sha in fit.get("table_sha256", fit.get("fit_table_sha256", {})).items():
        verify(out / filename, sha)
    for filename, record in data["outputs"].items():
        verify(ROOT / "data" / filename, record["sha256"])
        if "ledger_sha256" in record:
            verify(ROOT / "data" / (Path(filename).stem + "_provenance.tsv"), record["ledger_sha256"])
    rows = read_csv(out / "primary_coefficients.csv")
    assert len(rows) == expected_rows
    return rows


def audited_occupancy_rows():
    """Require frozen, audited final products before publishing the new batch."""
    out = ROOT / "analysis_occupancy_classes"
    summary_path = out / "summary.json"
    data_path = out / "data_validation.json"
    fit_path = out / "independent_fit_validation.json"
    summary, data, fit = (read_json(p) for p in (summary_path, data_path, fit_path))
    assert summary["stage"] == "final", "Occupation-class analysis is not final; RESULTS is unchanged"
    assert summary["required_grid_complete"] and summary["primary_complete"]
    assert summary["missing_required_keys"] == 0 and summary["primary_rows"] == 54
    assert not summary.get("sources_changed_during_analysis")
    assert not summary.get("dependency_changed_during_analysis")
    assert not summary.get("plan_changed_during_analysis")
    assert data["status"] == fit["status"] == "PASS"
    assert data["stage"] == fit["stage"] == "final"
    assert data["missing_required_rows"] == 0 and not data["missing_keys"]
    assert data["available_required_rows"] == data["required_rows"] == summary["required_keys"]
    assert fit["primary_anchor_rows"] == 54 and fit["fit_count_including_small_pairs"] > 0
    assert fit["precision_digits"] >= 80
    verify(summary_path, fit["summary_sha256"])
    verify(ROOT / "analyze_occupancy_classes.py", summary["script_sha256"])
    assert fit["script_sha256"] == summary["script_sha256"]
    verify(out / "plan.json", summary["plan_sha256"])
    assert data["plan_sha256"] == summary["plan_sha256"]
    verify(ROOT / "validate_occupancy_classes.py", data["validator_sha256"])
    for item in summary.get("dependencies", []):
        verify(Path(item["path"]), item["sha256"])
    for relative, sha in data["sources"].items():
        path = Path(relative)
        verify(path if path.is_absolute() else ROOT / path, sha)
    for item in summary["sources"]:
        assert not item.get("missing"), f"Missing declared source: {item['path']}"
        assert item.get("ignored_incomplete_tail_bytes", 0) == 0
        path = Path(item["path"])
        verify(path, item["sha256"])
        verify(Path(item["parsed_snapshot"]), item["parsed_sha256"])
        assert item["parsed_sha256"] == item["sha256"]
        relative = path.relative_to(ROOT).as_posix()
        assert relative in data["inputs"], f"Source lacks data audit: {relative}"
        assert data["inputs"][relative]["file_sha256"] == item["sha256"]
    required_tables = {"observations", "equilibrium_fits", "time_individual_fits",
                       "time_pair_fits", "raw_entropy_fits", "primary_coefficients"}
    assert set(fit["table_sha256"]) >= required_tables
    for stem, sha in fit["table_sha256"].items():
        filename = stem if stem.endswith(".csv") else stem + ".csv"
        verify(out / filename, sha)
    verify(ROOT / "data" / "occupancy_classes_validation.toml", data["small_system_validation_sha256"])
    # Bind the actual audit receipts as well as the files they audited.
    for path in (data_path, fit_path):
        verify(path, digest(path))
    plan = summary["plan"]
    metadata = summary["patterns"]
    assert len(plan["patterns"]) == len(set(plan["patterns"])) == 6
    assert set(metadata) == set(plan["patterns"])
    assert Counter(m["occupancy_class"] for m in metadata.values()) == {1: 2, 2: 2, 3: 2}
    betas = [float(x) for x in plan["betas"]]
    assert betas == [i/4 for i in range(9)]
    rows = read_csv(out / "primary_coefficients.csv")
    assert len(rows) == len({(r["pattern"], float(r["beta"])) for r in rows}) == 54
    assert {(r["pattern"], float(r["beta"])) for r in rows} == {
        (name, beta) for name in plan["patterns"] for beta in betas}
    for row in rows:
        meta = metadata[row["pattern"]]
        eq = meta["equilibrium"]
        assert int(row["occupancy_class"]) == meta["occupancy_class"]
        assert int(row["eq_Lmin"]) == eq["Lmin"] and int(row["eq_Lmax"]) == eq["Lmax"]
        assert row["eq_model"] == "log_inv1"
        assert row["primary_model"] == "log_inv1_inv2" and row["primary_window"] == "4-7"
        assert int(row["primary_L"]) == 180 and int(row["comparison_L"]) == 144
        assert row["scale_screen_both"].lower() == "true"
        assert all(math.isfinite(number(row, key)) for key in ("c_eq", "c_neq"))
    expected_figures = {"corner_coefficients_by_class"}
    for name in plan["patterns"]:
        expected_figures.update((f"{name}_corner_coefficient_beta", f"{name}_coefficient_diagnostics"))
        expected_figures.update(f"{name}_scaling_beta_{b:g}".replace(".", "p") for b in (0, .5, 1, 1.5, 2))
    assert set(summary["figures"]) >= expected_figures, "Final analysis did not render all required figures"
    return rows, summary, data, fit


def number(row, key):
    return float(row[key]) if row.get(key) else math.nan


def fmt(x):
    return f"{x:.7g}" if math.isfinite(x) else "—"


def pct(x):
    return f"{100*x:.2f}%" if math.isfinite(x) else "—"


def passed(row):
    return row["local_diagnostic_pass"].lower() == "true"


def picture(folder, stem, caption):
    for ext in ("png", "pdf"):
        path = ROOT / folder / f"{stem}.{ext}"
        assert path.is_file(), path
        USED.add(path)
    return f"![{caption}]({folder}/{stem}.png)\n\n[PNG]({folder}/{stem}.png) · [PDF]({folder}/{stem}.pdf)\n"


def table(rows):
    lines = ["| β | 平衡 c | 非平衡 c | 相对差 | 尺寸漂移 | 移窗漂移 | 阶数漂移 | 时间侧局部检查 |",
             "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        vals = [fmt(number(row, "beta")), fmt(number(row, "c_eq")), fmt(number(row, "c_neq"))]
        vals += [pct(number(row, key)) for key in ("relative_gap", "size_drift", "window_drift", "model_drift")]
        vals += ["通过" if passed(row) else "未稳定"]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def details(title, body):
    return f"<details>\n<summary>{title}</summary>\n\n{body}\n</details>\n"


def gallery(folder, prefix, label):
    return "\n".join(picture(folder, prefix + f"scaling_beta_{b:g}".replace(".", "p"),
                              f"{label}，β={b:g}：左 ln L，右 ln θ") for b in (0, .5, 1, 1.5, 2))


def uniform_section(rows):
    folder = "figures_uniformA_dense"
    pass_betas = "、".join(fmt(number(r, "beta")) for r in rows if passed(r))
    gap = max(abs(number(r, "relative_gap")) for r in rows)
    coefficient_note = r"""
**系数比较图怎么读：** 左图给出两端系数本身；中、右图把左图不易看清的小差异显示出来。

| 子图 | 表示什么 | 相等时的基准 |
|---|---|---|
| 左：系数 | 平衡与非平衡各自拟合的对数系数随 β 变化；阴影为模型/窗口敏感性范围，不是置信区间 | 两条曲线重合 |
| 中：比值 | 非平衡系数除以平衡系数；例如 1.02 表示大约高 2% | 蓝线 1 |
| 右：相对偏差 | 两端差值除以平衡系数绝对值；正值表示非平衡更大，负值表示更小 | 蓝线 0% |

$$
R=\frac{c_{\rm neq}}{c_{\rm eq}},\qquad
\delta=\frac{c_{\rm neq}-c_{\rm eq}}{|c_{\rm eq}|}\times100\%.
$$

本批平衡系数为正，所以中、右图是同一偏差的两种表示：右图等于中图减 1 后换成百分数。较大 β 下两端系数都很小，左图看似重合，右图仍显示约 1.96%–1.98% 的相对差。实心/空心非平衡方块分别表示局部检查通过/未稳定；这些有限窗口差异不等于已确认的渐近角系数差异。
"""
    time_note = r"""
**时间侧检查图怎么读：** 三图均只比较非平衡拟合，检查结果是否依赖尺寸、时间窗口或修正模型，完全不使用平衡系数。

| 子图 | 固定什么、改变什么 | 主要检查 |
|---|---|---|
| 左：尺寸漂移 | 固定 θ=4–7 的相同实测点和四参数模型，比较 L=144 与 180 的系数 | 有限尺寸影响 |
| 中：窗口敏感性 | 固定 L=180 和四参数模型，改变预定且满足筛选的可用时间窗口，取相对主窗口的最大偏差 | 取哪些时间点会不会改变系数；不只是单次 4–7→5–8 比较 |
| 右：模型敏感性 | 固定 L=180、θ=4–7 和相同实测点，比较含一、二、三阶逆时间修正的三、四、五参数模型，取最大偏差 | 早时修正的截断阶数是否受控 |

中图 β=1 的小突起来自额外纳入 5–9 窗口（约 0.816%）；只比较各 β 共有窗口时约为 0.677%，与邻点连续，不能把该突起解释为物理突变。

三图都以主非平衡系数作分母，画绝对相对变化；中、右图再对相应检查集合取最大值：

$$
\Delta=\frac{|c_{\rm variant}-c_*|}{|c_*|}\times100\%,\qquad
c_*=c_{\rm neq}(L=180,\ \theta=4\text{–}7,\ \text{四参数模型}).
$$

左图最大约 0.72%，中图约 0.17%–0.82%，右图在较大 β 时接近 7%。因此当前较强测量区的主要未通过项是**修正模型敏感性**，即使尺寸和移窗变化很小，也不能认定系数已经收敛。本次对应门槛为 2%、5%、5%，且完整局部检查还包括同点 D 差、变号和已有第三尺寸检查。这里方块仅标数据点，不用实心/空心区分通过状态；三图均不是统计误差条。
"""
    return "\n".join([
        f"21 个 β（0、0.1、…、2），698 个参数点（626 新算、72 复用）。两端主估计相对差最大 **{pct(gap)}**；时间侧局部检查通过 β={pass_betas}。较强测量处模型漂移约 7%，所以接近并不等于已证明渐近相等。",
        picture(folder, "corner_coefficient_beta", "uniform A：平衡与非平衡系数随 β 变化"),
        details("展开：21 个 β 的系数表与诊断图", table(rows) + "\n" +
                picture(folder, "coefficients_beta", "uniform A：系数比值、相对差和敏感性范围") + coefficient_note + "\n" +
                picture(folder, "time_diagnostics_beta", "uniform A：独立于平衡目标的时间侧检查") + time_note),
        details("展开：β=0、0.5、1、1.5、2 的平衡 / 非平衡 scaling 与拟合", gallery(folder, "", "uniform A")),
        "将 L=180 的窗口从 4–7 移到 5–8，代表性正 β 点的阶数差由约 6.71%–6.99% 降为 4.12%–4.30%。但只有 β=0、1 具有完整第三尺寸对照，不能据此替换全部 β 的主估计。[逐尺寸与移窗数据](analysis_uniformA_dense/time_individual_fits.csv)。",
        "[完整主系数表](analysis_uniformA_dense/primary_coefficients.csv) · [平衡拟合](analysis_uniformA_dense/equilibrium_fits.csv) · [全部时间敏感性](analysis_uniformA_dense/time_sensitivity.csv)"
    ])


def patterns_section(rows):
    folder = "figures_beta_patterns"
    parts = ["三个构型各 9 个 β（0、0.25、…、2），共 909 个参数点（653 新算、256 复用）、27 组系数对照。各计算尺寸均检查测量前基态可达性；占据图与编码见 [构型指南](SCALING_GUIDE.md#patterns)。",
             picture(folder, "corner_coefficients_comparison", "三个构型的平衡与非平衡系数随 β 变化"),
             "`one_B_2`、`diagonal_B_3` 的平衡主估计随 β 下降；`random_half_3` 先升后降。后者的平衡网格峰值在 β=1.25，非平衡网格峰值在 β=1.5，但所有非平衡点均未稳定，不能认定渐近峰位不同。"]
    for name in ("one_B_2", "diagonal_B_3", "random_half_3"):
        block = [r for r in rows if r["pattern"] == name]
        passes = "、".join(fmt(number(r, "beta")) for r in block if passed(r)) or "无"
        sensitivity = max(max(abs(number(r, k)-number(r, "c_eq")) for k in ("eq_range_min", "eq_range_max")) / abs(number(r, "c_eq")) for r in block)
        body = picture(folder, f"{name}_corner_coefficient_beta", f"{name}：两端系数随 β 变化")
        body += f"\n时间侧局部检查通过的 β：**{passes}**。平衡拟合改变尺寸下限和修正阶数后，最大相对变化为 **{pct(sensitivity)}**。\n\n"
        body += table(block) + "\n" + picture(folder, f"{name}_coefficient_diagnostics", f"{name}：比值、差异与模型敏感性")
        body += "\n" + gallery(folder, name + "_", name)
        parts.append(details(f"展开 {name}：全部系数、诊断与 5 组 scaling 图", body))
    parts.append("[27 组主系数](analysis_beta_patterns/primary_coefficients.csv) · [平衡拟合](analysis_beta_patterns/equilibrium_fits.csv) · [时间敏感性](analysis_beta_patterns/time_sensitivity.csv) · [原始区域熵拟合](analysis_beta_patterns/raw_entropy_fits.csv)")
    return "\n\n".join(parts)


def occupancy_section(rows, summary, data, fit):
    folder, out = "figures_occupancy_classes", ROOT / "analysis_occupancy_classes"
    names, metadata = list(summary["plan"]["patterns"]), dict(summary["patterns"])
    followup, _, _, follow_rows, _, _, _, _ = audited_irregular_followup()
    mixed_name = "near02_irregular3"
    mixed_meta = followup["plan"]["pattern_metadata"][mixed_name]
    bits = mixed_meta["occupied_bits"]
    cell_n = [int(bits[i]) + int(bits[i + 1]) for i in range(0, len(bits), 2)]
    metadata[mixed_name] = {**mixed_meta, "empty_cells": cell_n.count(0),
                            "single_cells": cell_n.count(1), "double_cells": cell_n.count(2)}
    names.append(mixed_name)
    rows = list(rows) + [r for r in follow_rows if r["pattern"] == mixed_name]
    assert len(rows) == 63 and metadata[mixed_name]["occupancy_class"] == 3
    eqfits = read_csv(out / "equilibrium_fits.csv")
    new_rows = len(read_csv(ROOT / "data" / "occupancy_classes_observations.csv"))
    groups = {name: sorted((r for r in rows if r["pattern"] == name), key=lambda r: number(r, "beta"))
              for name in names}
    classes = {
        1: "每个原胞恰有一个粒子",
        2: "只含空原胞和双占原胞",
        3: "单占、空、双占原胞混合",
    }
    lines = ["| 类别 | 构型 | 超胞 | 空 / 单占 / 双占原胞数 | 平衡主窗口 | 尺寸下限检查 |",
             "|---|---|---:|---:|---|---|"]
    for name in names:
        meta, eq = metadata[name], metadata[name]["equilibrium"]
        counts = f"{meta['empty_cells']} / {meta['single_cells']} / {meta['double_cells']}"
        lines.append(f"| {meta['occupancy_class']}：{classes[meta['occupancy_class']]} | [{name}](#occupancy-{name.replace('_', '-')}) | "
                     f"{meta['m']}×{meta['m']} | {counts} | L={eq['Lmin']}–{eq['Lmax']} | "
                     f"{'、'.join(map(str, eq['lower_cuts']))} |")
    parts = [
        "本节统一按原胞占据分类：第 1 类两个可达例，第 2 类四个 4×4 可达例并补充全部六种 2×2 受阻排列，"
        "第 3 类三个可达例（包括新增的 `near02_irregular3`）。"
        "首批七个可达例每个计算 9 个 β（0、0.25、…、2），共 63 组两端主拟合参数和 35 张左右拟合图。"
        "**本次另增两个可达 4×4 构型和两种 2×2 交替构型，每个计算 5 个 β，共 590 个新观测点、20 张左右拟合图。** "
        "现在共九个可达构型；两种交替构型的熵也实际计算，但其严格长时极限是激发态，不能计入基态角系数。",
        f"原六构型计划的 **{summary['required_keys']:,} 个参数键全部齐全**，"
        f"其中该批新增观测 CSV 共 **{new_rows:,} 行**；"
        "`one_B_2`、`diagonal_B_3`、`random_half_3` 复用上一批已有观测，保留逐行来源；"
        "新增 3×3 混合构型另有 252 个完整参数点，采用独立的冻结计划与核验记录。"
        "每构型展示 β=0、0.5、1、1.5、2；首批九个 β 的系数均收入曲线和完整数据表。"
        "[第二类新增四构型及交替结果](#occupancy-second-class-extra) 单列本次五个 β 的全量结果。",
        "### 首批七个可达构型的示意图与索引",
        "[第一类](#occupancy-class-1) · [第二类](#occupancy-class-2) · "
        "[第二类新增四构型](#occupancy-second-class-extra) · "
        "[第二类 2×2 受阻证明](#occupancy-pure02-m2-blocked) · [第三类](#occupancy-class-3)",
        picture("figures_pure02_m2", "classified_ground_patterns", "三类七个可达构型：新增 near02_irregular3 位于第三类"),
        "每个灰框表示一个原胞，蓝圆为 A、橙方为 B；实心表示占据，空心表示空位。"
        "每行对应一类，图中 n0、n1、n2 为一个超胞内空、单占、双占原胞的数量。"
        "这是原胞坐标示意图，整个系统按超胞周期重复；不是蜂窝晶格的实际键角图。"
        "[精确构型定义](occupancy_classes_patterns.jl) · [基态可达性与判据](SCALING_GUIDE.md#patterns)。"
        "下表点击构型名可跳到它的系数曲线和全部五组拟合图。受阻的 2×2 图案另在第二类展开，"
        "它们不在这张可达图中。",
        "\n".join(lines),
        details("拟合模型、尺寸与时间窗口", r"""这里分类依据是每个原胞两个格点的总占据：

$$
n_{\rm cell}=n_A+n_B.
$$

两个 4×4 构型的平衡网格为 L=24,36,…,144，图和表均标注实际的 24–144 窗口；该网格保证尺寸与超胞、切口相位相容。其余构型使用各自的 18–144 网格。平衡主模型为对数加常数与逆尺寸修正，并检查不同下限及逆平方修正：

$$
D_{\rm eq}(L)=c_{\rm eq}\ln L+b+\frac dL,
\qquad
D_{\rm eq}^{(2)}(L)=c_{\rm eq}\ln L+b+\frac dL+\frac e{L^2}.
$$

非平衡仍固定使用 L=180、θ=4,4.5,…,7 的四参数主模型，并以 L=144 的同点网格检查尺寸：

$$
D_{\rm neq}(\theta)=c_{\rm neq}\ln\theta+b+\frac d\theta+\frac e{\theta^2},
\qquad
\theta\geq4,\quad\frac{\theta}{L}\leq0.05.
$$

比较含一、二、三阶逆时间修正的三、四、五参数模型及预定子窗口。全部 β 另有 L=180 的 5–8 移窗；这不等于 L=144、180 都具备满足预筛的完整 5–8 同点尺寸对照。旧数据中存在的额外第三尺寸和窗口保留为诊断，因此每个构型、每个 β 的检查覆盖不完全相同。

这些是操作尺度筛选和有限窗口估计；主窗口只跨越 1.75 倍，不能据此证明真正渐近区已经达到。所有窗口均不按与平衡系数的接近程度选择，也不假设两方向条带熵相等。图中阴影是**模型和窗口敏感性范围，不是统计置信区间**；未稳定的正、负估计均保留。
"""),
        "### 首批七个可达构型的系数随 β 变化：总览",
        picture("figures_pure02_m2", "classified_ground_coefficients", "三类七个可达构型：平衡与非平衡 D 的有限窗口参数"),
        "总览每行对应一类，**各 panel 采用独立纵轴范围**，比较大小时需读刻度。"
        "蓝色粗实线与大空心圆表示平衡，橙色虚线与小方块表示非平衡；重合处仍可区分。"
        "实心/空心非平衡方块表示时间侧局部检查通过/未稳定，蓝色空心圆仅区分系列。"
        "总览省略敏感性阴影，完整范围见各构型单图和诊断图。"
        "时间主估计幅度超过平衡的十倍时，另附平衡曲线放大面板；原面板仍保留完整正负范围。",
    ]
    status_lines = ["| 构型 | 时间侧局部通过的 β | 其中仅有两尺寸检查的 β | β=2 平衡敏感性范围 |",
                    "|---|---|---|---|"]
    for name in names:
        block = groups[name]
        pass_rows = [r for r in block if passed(r)]
        passes = "、".join(fmt(number(r, "beta")) for r in pass_rows) or "无"
        two_sizes = "、".join(fmt(number(r, "beta")) for r in pass_rows
                             if r.get("third_size_missing", "").lower() == "true") or "无"
        at_two = next(r for r in block if number(r, "beta") == 2)
        bounds = f"[{fmt(number(at_two, 'eq_range_min'))}, {fmt(number(at_two, 'eq_range_max'))}]"
        status_lines.append(f"| `{name}` | {passes} | {two_sizes} | {bounds} |")
    parts += [
        "\n".join(status_lines),
        "局部通过只说明当前时间侧检查没有发现超门槛变化，不认证平衡侧已收敛，更不证明两端渐近角系数相等。"
        "**未稳定的有限窗口差异不能当成反例。** 本表和全部图报告的是加权 D 的拟合系数；"
        "原始矩形熵的对数项还可能含非角贡献，不能未经检查就把二者等同或简单除以角数。",
    ]
    zero_values = [number(r, "c_eq") for r in rows if number(r, "beta") == 0]
    parts.append(f"β=0 时七个平衡主估计位于 **{fmt(min(zero_values))}–{fmt(max(zero_values))}**。"
                 "本轮不将这点有限尺寸差异解释为不同普适值：周期系统的零模占据由各自初态选定，"
                 "当前有限 L 拟合也未消除全部尺寸修正。")

    parts.append("**下面按类别逐一展开全部拟合图。** 每个构型先给出两端系数随 β 的变化，"
                 "再依次给出 β=0、0.5、1、1.5、2 的数据和拟合。"
                 "每张左右图的**左侧是平衡 D 对 ln L，右侧是非平衡 D 对 ln θ**；"
                 "右侧同时显示 L=144、180 的数据，橙线是 L=180、θ=4–7 的主拟合。"
                 "拟合含逆尺度修正，所以在对数横轴上不一定是直线；"
                 "`unresolved` 表示时间侧检查未通过。")
    pattern_notes = {
        "one_B_2": "两端主系数随 β 增大而下降；曲线接近还需结合时间侧检查判断。",
        "diagonal_B_3": "两端主系数随 β 增大而下降；只有 β=0.5 通过当前时间侧局部检查。",
        "pure02_block4": "平衡主值在小 β 近似平台后下降，非平衡主值先有浅峰再下降；九个 β 的时间侧检查均未通过。",
        "pure02_irregular4": "**当前窗口无法提取稳定的非平衡对数系数。** 很大的正、负主估计来自对数项与逆时间修正项的补偿，不能解释为渐近角系数；β=1.5、1.75、2 的平衡拟合也有明显敏感性。",
        "random_half_3": "两端主值先升后降，但九个 β 的时间侧检查均未通过，不能从峰位和幅度差异确认渐近不一致。",
        "general_matched_a": "平衡主值随 β 下降，非平衡主值在小 β 有浅峰后下降；它与 random_half_3 的各类原胞数和 A/B 总数相同，但空间排列不同。",
        "near02_irregular3": "**归入第三类混合构型。** 半填充的 3×3 超胞有 9 个粒子，纯空胞／双占胞只能给出偶数粒子，因而不能把本例归入第二类。"
        "本例有 4 空胞、4 双占胞和 1 个 A 单占胞。预定候选筛选仅检查基态可达性，全部 13 个实验尺寸通过。"
        "平衡主值随 β 先升后降，在本次网格 β=1 处最大；有限尺寸稳定性比 irregular4 强测量区好，"
        "但九个 β 的时间侧检查均未通过，不能把两端差异当成已确认的渐近差异。",
    }
    for class_id, class_label in classes.items():
        parts.append(f'<a id="occupancy-class-{class_id}"></a>\n\n### 第 {class_id} 类：{class_label}')
        for name in names:
            meta = metadata[name]
            if meta["occupancy_class"] != class_id:
                continue
            block = groups[name]
            entry_folder = "figures_irregular_followup" if name == mixed_name else folder
            coefficient_stem = f"{name}_coefficients_beta" if name == mixed_name else f"{name}_corner_coefficient_beta"
            passes = "、".join(fmt(number(r, "beta")) for r in block if passed(r)) or "无"
            parts.extend([
                f'<a id="occupancy-{name.replace("_", "-")}"></a>\n\n#### {name}',
                f"超胞 {meta['m']}×{meta['m']}；空 / 单占 / 双占原胞数为 "
                f"{meta['empty_cells']} / {meta['single_cells']} / {meta['double_cells']}。"
                f"时间侧局部通过的 β：**{passes}**。 " + pattern_notes[name],
            ])
            if name == mixed_name:
                parts.append(picture("figures_pure02_m2", "near02_irregular3_pattern", "第三类 near02_irregular3：4空胞、1单占胞、4双占胞"))
                at_two = next(r for r in block if number(r, "beta") == 2)
                parts.append(f"β=2 平衡主值为 {fmt(number(at_two, 'c_eq'))}，尺寸下限／模型敏感性范围为 "
                             f"[{fmt(number(at_two, 'eq_range_min'))}, {fmt(number(at_two, 'eq_range_max'))}]。"
                             "[构型定义](irregular_m3_patterns.jl) · [逐尺寸可达性](data/irregular_m3_reachability.csv) · "
                             "[独立数据与拟合核验](analysis_irregular_followup/validation.json)。")
            parts.append(picture(entry_folder, coefficient_stem, f"{name}：平衡与非平衡系数随 β 的变化"))
            if name != mixed_name:
                diagnostics = f"{folder}/{name}_coefficient_diagnostics"
                for ext in ("png", "pdf"):
                    path = ROOT / f"{diagnostics}.{ext}"
                    assert path.is_file(), path
                    USED.add(path)
                parts.append(f"[敏感性、比值及相对差诊断 PNG]({diagnostics}.png) · [PDF]({diagnostics}.pdf)")
            for beta, tag in (("0", "0"), ("0.5", "0p5"), ("1", "1"), ("1.5", "1p5"), ("2", "2")):
                parts.append(f"##### {name}，β={beta}：左平衡，右非平衡\n\n" +
                             picture(entry_folder, f"{name}_scaling_beta_{tag}",
                                     f"{name}，β={beta}：左侧平衡 D 对 ln L；右侧非平衡 D 对 ln θ"))
            parts.append("[返回三类构型索引](#results-occupancy-classes)")
        if class_id == 2:
            parts.append(second_class_extra_section())
            parts.append(pure02_m2_section())

    parts.append("### 补充对照、完整系数表与核验")

    # These comparisons are generated only after all final gates above pass.
    # Values and sensitivity intervals come from that frozen numerical snapshot.
    pair = ("random_half_3", "general_matched_a")
    if all(name in metadata for name in pair):
        statistics = []
        for name in pair:
            meta = metadata[name]
            bits = meta["occupied_bits"]
            statistics.append((meta["empty_cells"], meta["single_cells"], meta["double_cells"],
                               bits[::2].count("1"), bits[1::2].count("1")))
        assert statistics[0] == statistics[1]
        n0, n1, n2, na, nb = statistics[0]
        comparison = ["| β | 构型 | 平衡主 c | 平衡尺寸 / 模型范围 |",
                      "|---:|---|---:|---|"]
        separated = True
        for beta in (1, 2):
            selected = []
            for name in pair:
                row = next(r for r in groups[name] if number(r, "beta") == beta)
                selected.append(row)
                comparison.append(f"| {beta:g} | `{name}` | {fmt(number(row, 'c_eq'))} | "
                                  f"[{fmt(number(row, 'eq_range_min'))}, {fmt(number(row, 'eq_range_max'))}] |")
            separated &= (max(number(r, "eq_range_min") for r in selected) >
                          min(number(r, "eq_range_max") for r in selected))
        statement = ("这两个具体图案的平衡有限窗口系数不同，且上述敏感性范围不重合，"
                     "说明这些占据计数不能唯一决定本批平衡拟合结果，空间排列仍需保留。"
                     if separated else
                     "这张表比较相同占据计数的两个具体图案，不凭主值差异自动断言不同渐近系数。")
        parts.append(details("展开：相同占据计数的两个 mixed 构型",
            f"两者同属第 3 类，空、单占、双占原胞数均为 ({n0},{n1},{n2})，"
            f"A/B 粒子数也均为 ({na},{nb})；各自测量自身的初始空位。\n\n" +
            "\n".join(comparison) + "\n\n" + statement +
            "这项比较只涉及平衡端；非平衡是否稳定仍按各自时间诊断判断。"
            "每类少数例子也不足以建立对整类构型的普遍定理。"))

    irregular = "pure02_irregular4"
    if irregular in groups:
        order_labels = {"log_inv1": "含 1/L", "log_inv1_inv2": "含 1/L、1/L²"}
        controls = [r for r in eqfits if r["pattern"] == irregular and number(r, "beta") == 2]
        control_lines = ["| 平衡尺寸下限 | 修正模型 | c | RMSE | 修正变化 / 对数变化 |",
                         "|---:|---|---:|---:|---:|"]
        for row in sorted(controls, key=lambda r: (int(r["requested_Lmin"]), r["model"])):
            control_lines.append(f"| {row['requested_Lmin']} | {order_labels[row['model']]} | "
                                 f"{fmt(number(row, 'c'))} | {fmt(number(row, 'rmse'))} | "
                                 f"{fmt(number(row, 'total_correction_to_log_span'))} |")
        strong = [r for r in groups[irregular] if number(r, "beta") in (1.5, 1.75, 2)]
        ranges = "；".join(f"β={fmt(number(r, 'beta'))}："
                          f"[{fmt(number(r, 'eq_range_min'))}, {fmt(number(r, 'eq_range_max'))}]"
                          for r in strong)
        at_two = next(r for r in strong if number(r, "beta") == 2)
        crosses_zero = number(at_two, "eq_range_min") < 0 < number(at_two, "eq_range_max")
        warning = ("β=2 的估计随下限和修正模型改变而跨零，平衡端仍未稳定；"
                   "不能把图中的大正值或负值解释为已提取出的渐近角系数。"
                   if crosses_zero else
                   "需结合下限、修正模型和残差判断平衡端是否稳定，不能只读主曲线。")
        parts.append(details("展开：pure02_irregular4 强测量区的平衡敏感性",
                             ranges + "。\n\n" + "\n".join(control_lines) + "\n\n" + warning))

    all_tables = "\n\n".join(f"**{name}**\n\n{table(groups[name])}" for name in names)
    parts.append(details("展开：63 组主拟合参数与时间侧诊断全表", all_tables))
    parts.extend([
        "[原六构型 54 组主系数 CSV](analysis_occupancy_classes/primary_coefficients.csv) · "
        "[新增 3×3 的来源表（筛选 near02_irregular3 的 9 行）](analysis_irregular_followup/primary_coefficients.csv) · "
        "[平衡拟合](analysis_occupancy_classes/equilibrium_fits.csv) · "
        "[时间敏感性](analysis_occupancy_classes/time_sensitivity.csv) · "
        "[原始区域熵拟合](analysis_occupancy_classes/raw_entropy_fits.csv) · "
        "[覆盖情况](analysis_occupancy_classes/coverage.csv)",
        f"原六构型的数据与来源核验为 final/PASS；独立 80 位 QR 复算 **{fit['fit_count_including_small_pairs']:,} 次拟合**"
        "（包括尺寸对双方与原始区域熵），核对 54 个主估计的真实尺寸锚点。"
        f"对数系数最大数值差为 **{fmt(fit['max_c_or_raw_logcoefficient_error'])}**。"
        "这项核验检查拟合数值与锚点，不另行宣称每个局部质量判据均由第二套程序重实现。",
        "[最终计划](analysis_occupancy_classes/plan.json) · "
        "[分析摘要](analysis_occupancy_classes/summary.json) · "
        "[数据与来源核验](analysis_occupancy_classes/data_validation.json) · "
        "[80 位拟合核验](analysis_occupancy_classes/independent_fit_validation.json) · "
        "[新增观测 CSV](data/occupancy_classes_observations.csv)",
        "CSV 沿用旧字段名：`premeasurement_ground`、`premeasurement_excess` 和 "
        "`premeasurement_residual` 记录该构型独立计算的**无限虚时、测量前极限**是否为基态，"
        "不表示每个有限 θ 的实际状态已经到达基态。实际熵始终按该行 θ 演化后测量；"
        "`energy`、`residual` 等字段描述该行的测量后态。",
        """复现本批最终分析及本节（不重跑熵计算）：

```powershell
python src/Measure/ai/validate_occupancy_classes.py --stage final
python src/Measure/ai/analyze_occupancy_classes.py --stage final
python src/Measure/ai/analyze_occupancy_classes.py --validate-fits
python src/Measure/ai/write_results_report.py --sections OCCUPANCY_CLASSES
```
""",
    ])
    return "\n\n".join(parts)


def second_class_extra_section():
    """New class-two data, with blocked limits kept distinct from equilibrium."""
    out, folder = ROOT / "analysis_second_class_extra", "figures_second_class_extra"
    summary, receipt, audit = (read_json(out / filename) for filename in
                               ("summary.json", "receipt.json", "validation.json"))
    assert summary["stage"] == receipt["stage"] == audit["stage"] == "final"
    assert receipt["status"] == "COMPLETE" and audit["status"] == "PASS"
    assert summary["rows"] == summary["required_keys"] == 590
    assert summary["missing_required_keys"] == 0 and summary["primary_complete"]
    assert audit["precision_digits"] >= 80 and audit["primary_rows"] == 20
    for record in (receipt, audit):
        for relative, expected in record["files"].items():
            verify(ROOT / relative, expected)
    plan = read_json(out / "plan.json")
    verify(out / "plan.json", summary["plan_sha256"])
    verify(ROOT / "analyze_second_class_extra.py", summary["script_sha256"])
    physical = read_toml(ROOT / "data/second_class_extra_validation.toml")
    reach = read_json(ROOT / "data/second_class_extra_reachability.json")
    boundary = read_json(out / "alternating_boundary_layer.json")
    assert boundary["status"] == "PASS" and boundary["rows"] == 26
    for filename, expected in boundary["files"].items():
        verify(ROOT / filename, expected)
    assert physical["status"] == reach["status"] == "PASS"
    verify(ROOT / "data/second_class_extra_validation.csv", physical["csv_sha256"])
    for filename, expected in physical["code_sha256"].items():
        verify(ROOT / filename, expected)
    for filename, expected in reach["files"].items():
        verify(ROOT / filename, expected)
    rows = read_csv(out / "primary_parameters.csv")
    time_fits = read_csv(out / "time_individual_fits.csv")
    observations = read_csv(ROOT / "data/second_class_extra_observations.csv")
    assert len(observations) == 590
    assert len(rows) == 20
    names, meta = plan["patterns"], plan["pattern_metadata"]
    expected_figures = {"pattern_catalogue", "parameters_beta"}
    for name in names:
        expected_figures.update((f"{name}_pattern", f"{name}_coefficients_beta"))
        for beta in plan["betas"]:
            tag = f"{beta:g}".replace(".", "p")
            expected_figures.update((f"{name}_scaling_beta_{tag}", f"{name}_raw_slopes_beta_{tag}"))
    assert len(expected_figures) == 50 and set(summary["figures"]) >= expected_figures
    groups = {name: sorted((r for r in rows if r["pattern"] == name), key=lambda r: number(r, "beta"))
              for name in names}
    labels = {
        "pure02_staircase4": "阶梯形：沿 y 的双占胞行长依次为 4、3、1、0，每行从 x=1 填起。",
        "pure02_notched_stripe4": "缺口条带：先占满 x=1、2 两列，再把 (1,1) 的双占胞移到 (3,2)。",
        "pure02_alternating_x": "条纹交替：x 为奇数的原胞双占，x 为偶数的原胞全空；沿 y 不变。",
        "pure02_alternating_checker": "棋盘交替：x+y 为偶数的原胞双占，奇数的原胞全空。",
    }
    parts = [
        '<a id="occupancy-second-class-extra"></a>\n\n#### 第二类追加：两种新可达构型与两种交替构型',
        "**新增四种构型均为严格的第二类：每个原胞只有 n=0 或 n=2。** "
        "全部计算 β=0、0.5、1、1.5、2；两种 4×4 各有 135 个观测点，两种 2×2 各有 160 个，共 590 个。"
        "两种新 4×4 构型先按几何和投影秩选择，没有按熵系数是否相等筛选。",
        picture(folder, "pattern_catalogue", "新增四种第二类构型：阶梯、缺口条带、条纹交替、棋盘交替"),
        "**条纹交替、棋盘交替均不能投影到基态，但这次仍完整计算了它们的熵。** "
        "其左端数据使用保持初态谱支撑的严格 θ→∞ 可达激发态，右端使用有限 θ 演化态，随后都测量初态自身空位。"
        "所以这两种的左图叫“可达极限”，不叫基态平衡；参数随 β 的曲线是描述性拟合，不能称为基态角系数。"
        "[受阻的解析原因](SCALING_GUIDE.md#m2-obstruction-proof) 不因多演化一段时间而消失。",
        details("本批固定网格、模型和判据", r"""可达 4×4 的极限网格为 L=24,36,…,144，检查下限 24、36、48；受阻 2×2 的极限网格为 L=18,24,…,90,108,126,144，检查下限 18、36、54。两者均拟合

$$
D_{\rm limit}(L)=c_{\rm limit}^{\rm fit}\ln L+b+\frac dL,
$$

并加入逆平方项检查模型敏感性。只有可达构型的 limit 才对应测量前已准备基态的平衡端。

所有四构型均有 L=144、θ=4,4.5,…,7 和 L=180、θ=4,4.5,…,8 的实测数据。时间主拟合固定为 L=180、θ=4–7：

$$
D(\theta)=c_{\rm time}^{\rm fit}\ln\theta+b+\frac d\theta+\frac e{\theta^2}.
$$

预先固定的比较窗口为 4–7、4.5–7、4–6.5、5–7、5–8，在点数足够的窗口比较一至三阶逆时间修正。主窗 4–7 及其它有共同网格的窗口以 L=144 同点数据检查尺寸；5–8 仅用于 L=180 的移窗敏感性，不能冒称有两尺寸全窗口对照。尺寸、模型、窗口漂移门槛分别为 2%、5%、5%，尺寸间 D 差与主对数变化之比门槛为 1%；缺失必要比较或系数跨零均不通过。

这些数值检查并不证明已进入渐近区。满足 θ≥4、θ/L≤0.05 只是统一的操作筛选，还需要初态相关微观过渡已结束、有限尺寸效应足够小。只跨 1.75 倍的主窗口很容易让多个修正项互相补偿；不能用“曲线拟合得好”替代原始局部斜率的平台检查。

图中范围是预定模型和窗口的敏感性范围，不是统计置信区间。受阻构型即使数值检查通过，也不获得基态角系数的物理解释。"""),
        picture(folder, "parameters_beta", "新增第二类四构型：极限与有限时间拟合参数随β的变化"),
        "蓝色粗实线与大空心圆表示极限端，橙色虚线与小方块表示有限时间端，重合时也可区分。"
        "每行左栏比较两端，右栏单独放大极限端。"
        "各面板采用独立纵轴；很大的正负数值要结合敏感性范围和原始曲线阅读，不能自动当成新的普适角系数。",
    ]
    raw_table = ["| β=0，构型 | D(36) | D(72) | D(144) | 36→72 的 ΔD/ln2 | 72→144 的 ΔD/ln2 |",
                 "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        vals = {int(r["L"]): number(r, "D") for r in observations
                if r["pattern"] == name and math.isinf(number(r, "theta")) and number(r, "beta") == 0}
        left = (vals[72] - vals[36]) / math.log(2)
        right = (vals[144] - vals[72]) / math.log(2)
        raw_table.append(f"| `{name}` | {fmt(vals[36])} | {fmt(vals[72])} | {fmt(vals[144])} | "
                         f"{fmt(left)} | {fmt(right)} |")
    parts.extend([
        "**先看未经拟合的 β=0 数据：两种交替态的可达极限没有显示常数对数斜率。** "
        "从 36→72 再到 72→144，两个区间的 ln L 跨度相同；若 D 已由 c ln L+b 主导，"
        "两段斜率应接近同一 c。下表中交替态的斜率却随尺寸约翻倍，当前范围更接近线性增长。"
        "这与两个新增可达构型在 β=0 的近似对数趋势明显不同；不应把它们的大对数拟合参数命名为基态角系数。",
        "\n".join(raw_table),
        "这里比较的是先固定有限 L 取 θ→∞ 的不同状态族。交替态保留受阻动量的激发成分；"
        "其结果不是“已经投影到基态但失去角 scaling”的反例，也不能直接推断先取大 L、有限 θ 的时间曲线。",
        details("为什么交替态仅满足 θ/L≪1 仍然不够（β=0 局部诊断）", r"""独立四轨道计算在指定动量截面上发现：受阻点附近的小基态重叠与虚时指数因子竞争，产生宽度约为

$$
w_\theta=C e^{-2\theta},\qquad C=3\ \text{（条纹）},\quad C=2\ \text{（棋盘）}
$$

的局部动量过渡层。分辨它还需要

$$
\frac{2\pi/L}{w_\theta}\ll1.
$$

条纹在 θ=4 时宽度约为 0.001006，而 L=144 的动量间隔约为 0.04363，L=180 也只有约 0.03491。精确受阻点保留正能粒子，最近的非零网格点却已接近填满负能态。因此小 θ/L 并没有保证这个过渡层被有限网格分辨。

这为本批交替态的有限尺寸影响提供局部机制，**不直接推导完整 D 或角系数，也不自动适用于正 β**。完整推导、26 点独立核验及误差见 [指南](SCALING_GUIDE.md#alternating-momentum-layer) 与 [数值记录](analysis_second_class_extra/alternating_boundary_layer.json)。"""),
    ])
    for name in names:
        block, item = groups[name], meta[name]
        reachable = item["ground_reachable"]
        endpoint = "平衡" if reachable else "可达激发态极限"
        passes = "、".join(fmt(number(r, "beta")) for r in block
                             if r["numeric_diagnostic_pass"].lower() == "true") or "无"
        parts.extend([
            f'<a id="occupancy-{name.replace("_", "-")}"></a>\n\n##### {name}',
            f"{item['m']}×{item['m']}；{labels[name]} " +
            ("全部计划尺寸通过基态可达性检查。" if reachable else "全部计划尺寸受阻，超胞 Γ 块负能重叠秩为 3/4。") +
            f"时间侧数值局部检查通过的 β：**{passes}**。" +
            ("局部通过也不认证渐近角系数。" if reachable else "该状态的基态角系数始终不适用。"),
            picture(folder, f"{name}_pattern", f"{name} 的原胞占据示意图"),
            picture(folder, f"{name}_coefficients_beta", f"{name}：{endpoint}和有限时间拟合参数随β变化"),
        ])
        if reachable:
            trends = {
                "pure02_staircase4": "平衡主值在小 β 略升、随后下降，并在采样点 β=2 回升；时间侧明显依赖修正模型。",
                "pure02_notched_stripe4": "平衡主值在本次五个采样点中以 β=1 较高，随后下降。时间原始曲线较平滑、两尺寸也很接近，但修正阶数仍会显著改变系数。",
            }
            values = []
            for model in ("log_inv1", "log_inv1_inv2", "log_inv1_inv2_inv3"):
                item_fit = next(r for r in time_fits if r["pattern"] == name and number(r, "beta") == 0
                                and int(r["L"]) == 180 and number(r, "theta_min") == 4
                                and number(r, "theta_max") == 7 and r["model"] == model)
                values.append(fmt(number(item_fit, "c")))
            parts.append(trends[name] + "例如 β=0、L=180、θ=4–7 时，包含一、二、三阶逆时间修正的拟合分别给出 "
                         + "、".join(values) + "。这说明当前窗口不足以给出稳健时间角系数；"
                         "不能把某一个主值接近平衡当成提取成功。")
        lines = [f"| β | {endpoint}拟合参数 | 极限尺寸 / 模型范围 | 时间拟合参数 | 时间模型 / 窗口范围 | 时间数值检查 |",
                 "|---:|---:|---|---:|---|---|"]
        for row in block:
            lines.append(f"| {fmt(number(row, 'beta'))} | {fmt(number(row, 'c_limit_fit'))} | "
                         f"[{fmt(number(row, 'limit_range_min'))}, {fmt(number(row, 'limit_range_max'))}] | "
                         f"{fmt(number(row, 'c_time_fit'))} | "
                         f"[{fmt(number(row, 'time_range_min'))}, {fmt(number(row, 'time_range_max'))}] | "
                         f"{'局部通过' if row['numeric_diagnostic_pass'].lower() == 'true' else '未稳定'} |")
        parts.append("\n".join(lines))
        for beta, tag in (("0", "0"), ("0.5", "0p5"), ("1", "1"), ("1.5", "1p5"), ("2", "2")):
            parts.extend([
                f"###### {name}，β={beta}：左{endpoint}，右有限时间",
                picture(folder, f"{name}_scaling_beta_{tag}",
                        f"{name}，β={beta}：左侧D对lnL，右侧D对lnθ；{endpoint}"),
                f"[原始局部斜率 PNG]({folder}/{name}_raw_slopes_beta_{tag}.png) · "
                f"[PDF]({folder}/{name}_raw_slopes_beta_{tag}.pdf)",
            ])
        parts.append("原始局部斜率由相邻点的 ΔD/Δln x 直接计算，不使用修正模型。"
                     "随尺度持续漂移意味着当前数据尚未显示稳定的对数斜率；它不是对无限尺度行为的否定证明。")
    parts.extend([
        "##### 本批核验与复现",
        f"全部 **590/590** 个冻结参数键完整，逐行来源、代码快照及拟合表通过独立核验。"
        f"独立 {audit['precision_digits']} 位 QR 复算 {audit['fit_count']} 次拟合，"
        f"对数系数最大数值差为 {fmt(audit['max_fit_error'])}；这项数值核对不认证物理渐近区。"
        f"另用完整实空间关联矩阵核对 **{physical['states']}** 个小系统状态，三个区域熵最大差 "
        f"{fmt(physical['max_entropy_error'])}，D 最大差 {fmt(physical['max_D_error'])}。"
        "受阻构型的无限时间参考采用独立的能量块／零空间构造，保持严格谱支撑；"
        "没有把普通 Float64 长时 QR 混入的基态作为参考。",
        "[新构型定义](second_class_extra_patterns.jl) · [逐尺寸投影秩](data/second_class_extra_reachability.csv) · "
        "[原始熵数据](data/second_class_extra_observations.csv) · [20 组参数及诊断](analysis_second_class_extra/primary_parameters.csv) · "
        "[极限拟合](analysis_second_class_extra/limit_fits.csv) · [原始局部斜率](analysis_second_class_extra/local_log_slopes.csv) · "
        "[冻结计划](analysis_second_class_extra/plan.json) · [独立核验](analysis_second_class_extra/validation.json)",
        "```powershell\n# 已有数据的分析、核验和文档刷新\n"
        "python src/Measure/ai/analyze_second_class_extra.py --stage final\n"
        "python src/Measure/ai/validate_second_class_extra.py\n"
        "python src/Measure/ai/write_results_report.py --sections OCCUPANCY_CLASSES\n"
        "# 复跑熵数据（跳过已有完整行）\n"
        "& ./src/Measure/ai/run_second_class_extra.ps1\n```",
        "[返回三类构型索引](#results-occupancy-classes)",
    ])
    return "\n\n".join(parts)


def pure02_m2_section():
    """Present all requested m=2 class-two states as blocked controls."""
    audit = read_json(ROOT / "data/pure02_m2_summary.json")
    plots = read_json(ROOT / "figures_pure02_m2/receipt.json")
    assert audit["status"] == plots["status"] == "PASS"
    for receipt in (audit, plots):
        for relative, expected in receipt["files"].items():
            verify(ROOT / relative, expected)
    gamma = read_csv(ROOT / "data/pure02_m2_gamma.csv")
    reach = read_csv(ROOT / "data/pure02_m2_reachability.csv")
    projection = read_csv(ROOT / "data/pure02_m2_projection.csv")
    assert len(gamma) == 6 and len(reach) == 6 * len(audit["sizes"])
    assert all(int(r["gamma_rank"]) == 3 and int(r["gamma_zero_modes"]) == 0 for r in gamma)
    assert all(r["reachable"].lower() == "false" and number(r, "accessible_excess") > 0 for r in reach)
    assert audit["reachable_patterns"] == []
    largest = max(audit["sizes"])
    table_lines = [f"| 构型 | 占据 bits | Γ 负能重叠秩 | L=6 严格极限 E−E₀ | L={largest} 严格极限 E−E₀ |",
                   "|---|---|---:|---:|---:|"]
    for pattern in audit["patterns"]:
        name = pattern["name"]
        low = next(r for r in reach if r["pattern"] == name and int(r["L"]) == 6)
        high = next(r for r in reach if r["pattern"] == name and int(r["L"]) == largest)
        table_lines.append(f"| `{name}` | `{pattern['bits']}` | 3 / 4 | "
                           f"{fmt(number(low, 'accessible_excess'))} | {fmt(number(high, 'accessible_excess'))} |")
    qr_lines = ["| L=6，θ=128 | 保持初态谱支撑的 E−E₀ | 直接 Float64 QR 的 E−E₀ | 超出初态所能占据的负能粒子数 |",
                "|---|---:|---:|---:|"]
    for name in ("pure02_m2_cells_12", "pure02_m2_cells_14"):
        subset = [r for r in projection if r["pattern"] == name and number(r, "theta") == 128]
        direct = next(r for r in subset if r["method"] == "direct_qr")
        strict = next(r for r in subset if r["method"] == "strict_spectral_support")
        qr_lines.append(f"| `{name}` | {fmt(number(strict, 'excess_energy'))} | "
                        f"{fmt(number(direct, 'excess_energy'))} | {fmt(number(direct, 'negative_support_violation'))} |")
    return "\n\n".join([
        '<a id="occupancy-pure02-m2-blocked"></a>\n\n#### 第二类 2×2：全部六种排列的计算结果',
        "**2×2 可以构造半填充的第二类图案，但六种排列全部无法通过本协议投影到基态。** "
        "四个粒子必须放在两个双占胞中，另外两胞全空，因此本次穷举全部六种，而不是只挑一个。"
        f"对 L={','.join(map(str, audit['sizes']))} 的 **{len(reach)} 个构型／尺寸组合**重新计算，全部受阻。"
        "这里报告的是实际可达性和投影结果；平衡角系数为“不适用”，不是零。",
        picture("figures_pure02_m2", "pattern_catalogue", "第二类2×2的全部六种半填充排列：Γ负能重叠秩均为3/4"),
        "bits 仍按 A(1,1),B(1,1),A(2,1),B(2,1),… 排列，x 先变；"
        "cells_ij 表示按 x 先变编号的第 i、j 个原胞双占。条纹的实际平移周期可能小于画出的 2×2。",
        "\n".join(table_lines),
        r"""所有相容的周期系统都包含超胞 Γ 点。这里四个负能级为 −3、−1、−1、−1，没有零模，而每个图案都有

$$
\operatorname{rank}(V_{<0}^{\dagger}P_{\Gamma})=3<4=\dim\mathcal H_{<0}.
$$

因此缺少一个必须填充的负能方向；其它动量处的零模不能补救。这个结论对所有相容偶数 L 成立，依据是 [Γ 块解析证明](SCALING_GUIDE.md#m2-obstruction-proof)，不是把有限尺寸枚举外推为定理。较大尺寸上还有其它受阻动量；一般非对称动量可以满秩，并非每个动量都受阻。""",
        picture("figures_pure02_m2", "energy_excess_vs_L", "第二类2×2：严格可达极限的总能量差与每格点能量差"),
        "左图是总激发能，右图除以 2L² 得到每格点能量差。即使能量密度随 L 增大而减小，也不代表有限尺寸的态已经是基态，"
        "也不会恢复严格为零的多体基态重叠。这张图是投影受阻诊断，不是纠缠熵的 scaling 图。",
        "##### 为什么直接长时间 QR 可能显示“到了基态”",
        "另在 L=6 直接用实空间 exp(−0.25H) 与 QR 逐步演化到 θ=128，并与保持初态精确谱支撑的参考比较。"
        "两者早时一致，随后直接 Float64 演化把舍入误差带入原本禁止的负能方向，并指数放大。"
        "下面的负能粒子数超过初态谱支撑允许上限，直接表明数值轨迹已经离开原来的严格可达子空间。",
        "\n".join(qr_lines),
        picture("figures_pure02_m2", "projection_energy", "L6真实空间投影：保持谱支撑的参考与直接Float64 QR演化"),
        "保持谱支撑的无限时间总能量与另一套逐 Bloch 块计算相符，最大差约 1.42×10⁻¹⁴；"
        "早时 θ≤4 的完整关联矩阵之差，其 Frobenius 范数除以 L 后最大约 7.07×10⁻¹³。"
        "**更长虚时或 QR 收敛本身不能作为基态可达性的证据。**",
        "**因此这六种 2×2 图案不能提供原协议的基态平衡角系数。** "
        "本次已为条纹交替和棋盘交替两种代表实际计算有限 θ 熵与严格可达极限熵，"
        "见 [新增四构型结果](#occupancy-second-class-extra)；它们的拟合参数不能标成基态角系数。"
        "直接独立准备某个基态再用相同空位集合测量，会改变此前约定的制备协议。"
        "第二类可达结果由本节四个 4×4 构型提供；3×3 混合例位于下一节第三类。",
        "[本次检查代码](check_pure02_m2.py) · [Γ 枚举](data/pure02_m2_gamma.csv) · "
        "[逐尺寸结果](data/pure02_m2_reachability.csv) · [投影曲线原始数据](data/pure02_m2_projection.csv) · "
        "[全部受阻动量](data/pure02_m2_blocked_momenta.csv) · [结果与哈希](data/pure02_m2_summary.json)。",
        "```powershell\npython src/Measure/ai/check_pure02_m2.py\n"
        "python src/Measure/ai/plot_pure02_m2.py\n"
        "python src/Measure/ai/write_results_report.py --sections OCCUPANCY_CLASSES\n```",
        "[返回三类构型索引](#results-occupancy-classes)",
    ])


def audited_irregular_followup():
    """Read one frozen follow-up for class-three presentation and diagnostics."""
    out, folder = ROOT / "analysis_irregular_followup", "figures_irregular_followup"
    summary = read_json(out / "summary.json")
    audit = read_json(out / "validation.json")
    receipt = read_json(out / "receipt.json")
    modes = read_json(out / "mode_diagnostics.json")
    ground_check = read_toml(ROOT / "data/irregular_ground_validation.toml")
    new_check = read_toml(ROOT / "data/irregular_m3_validation.toml")
    for label, check in (("irregular_ground", ground_check), ("irregular_m3", new_check)):
        assert check["status"] == "PASS" and check["max_entropy_error"] < 1e-9
        verify(ROOT / f"data/{label}_validation.csv", check["csv_sha256"])
        for name, expected in check["code_sha256"].items():
            verify(ROOT / name, expected)
    assert summary["stage"] == receipt["stage"] == audit["stage"] == "final"
    assert audit["status"] == modes["status"] == "PASS" and receipt["status"] == "COMPLETE"
    assert summary["missing_required_keys"] == 0 and summary["primary_complete"]
    verify(out / "summary.json", audit["summary_sha256"])
    verify(ROOT / "analyze_irregular_followup.py", summary["script_sha256"])
    verify(out / "plan.json", summary["plan_sha256"])
    for item in receipt["outputs"]:
        path = Path(item["path"])
        verify(path if path.is_absolute() else ROOT / path, item["sha256"])
    for item in summary["sources"]:
        verify(Path(item["path"]), item["sha256"])
    for item in summary["dependencies"]:
        verify(Path(item["path"]), item["sha256"])
    verify(ROOT / "diagnose_irregular_modes.py", modes["script_sha256"])
    verify(out / "mode_diagnostics.csv", modes["csv_sha256"])
    coefficients = read_csv(out / "primary_coefficients.csv")
    equilibrium_fits = read_csv(out / "equilibrium_fits.csv")
    slopes = read_csv(out / "local_log_slopes.csv")
    heldout = read_csv(out / "heldout_extrapolation.csv")
    mode_rows = read_csv(out / "mode_diagnostics.csv")
    return summary, modes, ground_check, coefficients, slopes, heldout, mode_rows, equilibrium_fits


def irregular_followup_section():
    """Keep mechanism and extrapolation checks separate from class entries."""
    folder = "figures_irregular_followup"
    summary, modes, ground_check, coefficients, slopes, heldout, mode_rows, equilibrium_fits = audited_irregular_followup()
    names = summary["plan"]["patterns"]
    parts = [
        "**复核结论：目前未发现原始波函数演化、测量归一化或熵计算的实现错误；"
        "但 `pure02_irregular4` 强测量区的平衡数据，以及该构型在当前 θ 窗口内的全部非平衡数据，"
        "不能用原先主拟合的 c 认定为已提取出的角贡献系数。** "
        "此前把这些未稳定参数与角系数放在同一张图中容易造成误解；这里将它们明确作为拟合失效的诊断保留。"
        "β=0 的平衡数据仍接近对数行为，不能概括为这个构型在所有 β 下都没有对数项。",
        f"本次新增 3×3 构型的 252 个参数点，并补算两个 4×4 构型在 L=180、216、β=0、1、2 的 12 个平衡点。"
        f"连同复用数据，共 **{summary['required_keys']} 个要求的参数点齐全**。"
        "原 24–144 的拟合保持冻结，用新增尺寸检验其预测；包含新尺寸的重拟合另表列出。",
        "构型图和全部 β 拟合已按类别集中："
        "[第二类 block4](#occupancy-pure02-block4)、[第二类 irregular4](#occupancy-pure02-irregular4)、"
        "[第三类 near02_irregular3](#occupancy-near02-irregular3)。"
        "新 3×3 的平衡有限尺寸表现较规则，但它是混合构型，不能当作纯 0/2 类内的严格尺寸对照。"
        "本节只保留原始斜率、扩大尺寸外推与机制复核。",
        "### 先看原始数据的斜率，而不是强行给出一个对数系数",
        "问题主要出在当前窗口对拟合模型的支持程度：用“对数＋常数＋逆尺度修正”拟合一段弯曲曲线，"
        "即使残差不大，也不代表对数项已经主导。修正项占主导或彼此补偿时，c 只是对模型非常敏感的外推参数。"
        "高精度重算能复现这个参数，不能使它自动成为可靠物理系数。",
        "例如 irregular4 的 β=2，保留同一 log＋1/L 模型和 Lmax=144，"
        "仅改变尺寸下限便得到：" + "；".join(
            f"Lmin={cut}：c={fmt(number(next(r for r in equilibrium_fits if r['pattern'] == 'pure02_irregular4' and number(r, 'beta') == 2 and r['model'] == 'log_inv1' and number(r, 'requested_Lmin') == cut), 'c'))}"
            for cut in (24, 36, 48)) + "。这种变号不支持把其中某个值当成已经确定的角系数。",
        r"""若已经进入主导的对数区，相邻数据点的局部斜率应逐渐接近平台：

$$
c_{\rm loc}(L_i,L_{i+1})=
\frac{D(L_{i+1})-D(L_i)}{\ln L_{i+1}-\ln L_i}.
$$

下面的斜率直接由原始数据作差，未拟合任何修正项。有限尺寸下允许有漂移；持续的大幅漂移表示当前窗口不足以给出稳定的对数系数，并不证明无限尺寸一定没有对数项。""",
    ]
    local_table = ["| β | 构型 | 原范围末段 | 该段局部斜率 | 新增末段 | 该段局部斜率 |",
                   "|---:|---|---|---:|---|---:|"]
    for beta in (0, 1, 2):
        for name in names:
            block = [r for r in slopes if r["pattern"] == name and r["domain"] == "equilibrium"
                     and number(r, "beta") == beta]
            base = max((r for r in block if number(r, "x_right") <= 144), key=lambda r: number(r, "x_right"))
            extra = [r for r in block if number(r, "x_right") > 144]
            tail = max(extra, key=lambda r: number(r, "x_right")) if extra else None
            interval = lambda r: f"{number(r, 'x_left'):g}–{number(r, 'x_right'):g}"
            local_table.append(f"| {beta} | `{name}` | {interval(base)} | {fmt(number(base, 'local_log_slope'))} | "
                               f"{interval(tail) if tail else '未补算'} | {fmt(number(tail, 'local_log_slope')) if tail else '—'} |")
    parts.append("\n".join(local_table))
    for beta in (0, 1, 2):
        parts.append(picture(folder, f"equilibrium_raw_and_slopes_beta_{beta}",
                             f"β={beta}：三构型的原始 D 对 ln L，以及不作模型拟合的相邻点对数斜率"))
    prediction_table = ["| 构型 | β | 新尺寸 L | 实算 D | 原拟合预测 D | 实算−预测 |",
                        "|---|---:|---:|---:|---:|---:|"]
    for name in names[:2]:
        for row in sorted((r for r in heldout if r["pattern"] == name and number(r, "beta") == 2
                           and r["model"] == "log_inv1" and number(r, "train_Lmin") == 24),
                          key=lambda r: number(r, "heldout_L")):
            prediction_table.append(f"| `{name}` | 2 | {number(row, 'heldout_L'):g} | "
                                    f"{fmt(number(row, 'D_observed'))} | {fmt(number(row, 'D_prediction'))} | "
                                    f"{fmt(number(row, 'residual'))} |")
    parts.extend(["**β=2 的真正外推检验：新增点不参与原拟合。**", "\n".join(prediction_table),
                  picture(folder, "pure02_irregular4_extrapolation_beta_2", "irregular4，β=2：冻结拟合外推误差及扩大尺寸后的系数敏感性"),
                  "图中重新改变尺寸下限和修正阶数只是敏感性检查。"
                  "不能挑选其中最接近另一个构型或最接近平衡端的值作为答案。"
                  "[全部外推检验](analysis_irregular_followup/heldout_extrapolation.csv) · "
                  "[扩大尺寸后的拟合表](analysis_irregular_followup/extended_equilibrium_fits.csv)。",
                  "### 异常有何机制线索，是否说明新物理",
                  "独立重建 Bloch Hamiltonian，并用另一种子空间演化公式复算后，"
                  f"与原程序的 30 个 Hamiltonian／关联矩阵比较，最大元素误差为 **{fmt(modes['FE_comparison']['max_absolute_error'])}**。"
                  f"两个旧 4×4 构型另在 L=12 用完全实空间逐步指数演化与 QR 投影到 θ=128，"
                  f"核对 θ=64→128 收敛和初态选取的零模；测量后区域熵最大差为 **{fmt(ground_check['max_entropy_error'])}**，"
                  f"D 最大差为 **{fmt(ground_check['max_D_error'])}**。"
                  "新构型另外通过 L=9 的完整实空间指数演化与 QR 检查，区域熵误差约 10⁻¹³；"
                  "无限虚时参考还比较了 θ=64 与 128，包含初态选取的零模。"
                  "这些检查支持原始数值的正确性，但不自动认证 scaling。"])
    mode_table = ["| 构型 | 该模式正负能隙 | 基态重叠最小奇异值 s | θ=4 投影距离 | θ=7 投影距离 | β=2 最弱 F 主方向占据 |",
                  "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        pick = lambda t, b: next(r for r in mode_rows if r["pattern"] == name and number(r, "theta") == t and number(r, "beta") == b)
        t4, t7, ground = pick(4, 0), pick(7, 0), pick(math.inf, 2)
        mode_table.append(f"| `{name}` | {fmt(number(t4, 'positive_negative_energy_gap'))} | {fmt(number(t4, 'ground_overlap_sigma_min'))} | "
                          f"{fmt(number(t4, 'premeasurement_ground_projector_frobenius'))} | "
                          f"{fmt(number(t7, 'premeasurement_ground_projector_frobenius'))} | "
                          f"{fmt(number(ground, 'measured_min_F_principal_occupation'))} |")
    parts.extend(["下表比较各自超胞坐标中的**有能隙、非 Dirac 动量 (π,0)**。两个 4×4 构型的谱与能隙相同，适合隔离初态重叠的影响；3×3 的折叠、能隙和矩阵维度不同，其速度差不能只归因于 s。投影距离是未测量态与基态关联矩阵之差的 Frobenius 范数；s 是该动量的重叠奇异值，不是全动量最小值或多体重叠。",
                  "\n".join(mode_table),
                  r"""irregular4 的小重叠让该模式在 θ=4–7 仍跨越明显的投影过渡区。在有能隙且负能重叠 A 可逆的动量块中，令正交初态轨道为 P、正负能基为 V，演化后的占据空间可写成 V[I;X]，其中

$$
P^\dagger P=I,\quad A=V_-^\dagger P,\quad B=V_+^\dagger P,\quad s_{\min}=\sigma_{\min}(A),
$$

$$
X_\theta=e^{-\theta E_+}BA^{-1}e^{\theta E_-},
$$

$$
\|X_\theta\|_2\leq e^{-\Delta\theta}\sqrt{s_{\min}^{-2}-1},
\qquad \Delta=\min E_+-\max E_->0.
$$

两个 4×4 构型的能隙相同，但该界的前因子分别约为 8.90 和 193.63，因此仅有“非零重叠”不足以保证同一个 θ 窗口都已跨过微观过渡区。这个界是保守的单动量收敛界，不是完整熵拟合下限的判据。

对正交基态轨道 Q，令 F 为初始占据位置，s 为 Q 在 F 上的一个奇异值，则测量后相应 F 主方向的占据恰为

$$
p_F(\beta)=\frac{s^2}{s^2+e^{-4\beta}(1-s^2)}.
$$

因此相同 β 不代表不同构型已经处于同一个强测量阶段。irregular4 在这个模式的 β=2 主占据仍很小，另外两个构型已接近 1。**这是构型相关交叉尺度的具体证据，不是新相或新角普适系数的证据。** 单动量的诊断也不能证明它是整个 D(L) 曲率的唯一来源。

对任意有限 β≥0，测量矩阵 M 的最小奇异值是 e^{-2β}，且

$$
e^{-4\beta}I\preceq Q^\dagger M^2Q\preceq I,
\qquad
C_\beta=M Q(Q^\dagger M^2Q)^{-1}Q^\dagger M.
$$

在 0≤β≤2 时中间矩阵的条件数上界约为 2981，不支持“测量归一化已因条件数而失效”的解释；这并不保证熵差的大数相减或拟合外推也稳定。这个有限、可逆滤波不会在原关联投影光滑的动量处凭空产生奇异面；它仍可能产生尖锐但光滑的结构与很长的交叉区。这里没有据此证明所有构型的渐近角系数相同。""",
                  picture(folder, "mode_crossover", "固定有能隙动量的投影过渡与测量后主方向占据：不是熵的 scaling 图")])
    parts.append("[本批主系数 CSV](analysis_irregular_followup/primary_coefficients.csv) · "
                 "[相邻点斜率](analysis_irregular_followup/local_log_slopes.csv) · "
                 "[独立数据／拟合核验](analysis_irregular_followup/validation.json) · "
                 "[单动量独立诊断](analysis_irregular_followup/mode_diagnostics.json) · "
                 "[旧4×4基态／零模实空间核验](data/irregular_ground_validation.toml) · "
                 "[新构型实空间核验](data/irregular_m3_validation.toml)。\n\n"
                 "下列命令从仓库根目录复现。前两条计算熵（已有参数键会跳过）；"
                 "后三条生成图表、独立核验并更新本节。扩大尺寸的 12 个点单独保存，"
                 "不会覆盖旧数据。\n\n"
                 "```powershell\n"
                 "powershell -NoProfile -File src/Measure/ai/run_irregular_followup.ps1\n"
                 "julia --startup-file=no src/Measure/ai/occupancy_classes_experiments.jl "
                 "--backend=openblas --threads=8 --patterns=pure02_block4,pure02_irregular4 "
                 "--Ls=180,216 --thetas=Inf --betas=0,1,2 "
                 "--output=src/Measure/ai/data/irregular_followup_extended_equilibrium.csv\n"
                 "python src/Measure/ai/analyze_irregular_followup.py --stage final\n"
                 "python src/Measure/ai/validate_irregular_followup.py\n"
                 "python src/Measure/ai/write_results_report.py --sections OCCUPANCY_CLASSES IRREGULAR_FOLLOWUP\n"
                 "```"
                 "\n\n独立实现检查代码：[旧 4×4 实空间投影](validate_irregular_ground.jl)、"
                 "[新 3×3 实空间演化](validate_irregular_m3.jl)、"
                 "[单动量子空间诊断](diagnose_irregular_modes.py)。")
    return "\n\n".join(parts)


def calibration_section():
    out = ROOT / "analysis_beta_patterns"
    review = read_json(out / "calibration_review.json")
    assert review["status"] == "COMPLETE_TIME_ONLY_REVIEW"
    assert review["actual_keys"] == review["required_keys"] == 198
    verify(Path(review["source"]), review["source_sha256"])
    verify(ROOT / "review_beta_patterns_calibration.py", review["script_sha256"])
    for name, sha in review["table_sha256"].items():
        verify(out / name, sha)
    lines = ["| 构型 | β | 通过的局部窗口 | 通过的相邻窗口比较 |", "|---|---:|---|---|"]
    for r in review["summary"]:
        lines.append(f"| {r['pattern']} | {r['beta']:g} | {'、'.join(r['local_windows']) or '无'} | {'；'.join(r['adjacent_diagnostic_pairs']) or '无'} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+",
                        choices=("UNIFORM", "PATTERNS", "CALIBRATION", "OCCUPANCY_CLASSES", "IRREGULAR_FOLLOWUP", "BETA0_STUDY"),
                        default=None,
                        help="Refresh only these generated blocks; all selected blocks must have final audits")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate final inputs and prepare text without changing RESULTS or its receipt")
    args = parser.parse_args()
    if args.sections is None:
        args.sections = ["UNIFORM", "PATTERNS", "CALIBRATION", "OCCUPANCY_CLASSES"]
        if "<!-- GENERATED_IRREGULAR_FOLLOWUP_START -->" in (ROOT / "RESULTS.md").read_text(encoding="utf-8"):
            args.sections.append("IRREGULAR_FOLLOWUP")
        if "<!-- GENERATED_BETA0_STUDY_START -->" in (ROOT / "RESULTS.md").read_text(encoding="utf-8"):
            args.sections.append("BETA0_STUDY")
    assert len(args.sections) == len(set(args.sections)), "Duplicate section selection"
    USED.clear()
    VERIFIED.clear()
    from beta0_report_section import build as build_beta0_section
    builders = {
        "UNIFORM": lambda: uniform_section(audited_rows("uniformA_dense", 21)),
        "PATTERNS": lambda: patterns_section(audited_rows("beta_patterns", 27)),
        "CALIBRATION": calibration_section,
        "OCCUPANCY_CLASSES": lambda: occupancy_section(*audited_occupancy_rows()),
        "IRREGULAR_FOLLOWUP": irregular_followup_section,
        "BETA0_STUDY": lambda: build_beta0_section(sys.modules[__name__]),
    }
    sections = {name: builders[name]() for name in args.sections}
    path = ROOT / "RESULTS.md"
    original = path.read_text(encoding="utf-8")
    body = original
    for key, content in sections.items():
        start, end = f"<!-- GENERATED_{key}_START -->", f"<!-- GENERATED_{key}_END -->"
        assert body.count(start) == body.count(end) == 1, key
        body = re.sub(re.escape(start) + r".*?" + re.escape(end),
                      lambda _: start + "\n\n" + content + "\n\n" + end, body, flags=re.S)
    for source, expected in VERIFIED.items():
        assert digest(source) == expected, f"Source changed while preparing report: {source}"
    if args.check_only:
        print(f"Validated final inputs and prepared {', '.join(args.sections)}; no document or receipt changed.")
        return
    assert path.read_text(encoding="utf-8") == original, "RESULTS changed while preparing report"
    path.write_text(body, encoding="utf-8")
    receipt = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Documentation only; source data, numerical fits and figures are not recomputed.",
               "report": "RESULTS.md", "report_sha256": digest(path),
               "generator_sha256": digest(Path(__file__)),
               "updated_sections": args.sections,
               "sources": {(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)): digest(p)
                           for p in sorted(USED)}}
    (ROOT / "data/documentation_report_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    count = sum({"UNIFORM": 21, "PATTERNS": 27, "CALIBRATION": 0, "OCCUPANCY_CLASSES": 83, "IRREGULAR_FOLLOWUP": 27, "BETA0_STUDY": 16}[name]
                for name in args.sections)
    print(f"Updated {path.name}: {', '.join(args.sections)}; audited {count} section estimate records (overlapping sources included); no separate Markdown reports created.")


if __name__ == "__main__":
    main()
