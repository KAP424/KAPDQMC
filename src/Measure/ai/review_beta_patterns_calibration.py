"""Independent time-only calibration review; no equilibrium target is loaded.

Run only after the 198 requested calibration observations are complete. The
default input is the production raw CSV, not a fitted/merged analysis table.
The synthetic --self-check never opens the production CSV.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys

AI = Path(__file__).resolve().parent
OUT = AI / "analysis_beta_patterns"
sys.path.insert(0, str(OUT / "audit_dependencies"))
import mpmath as mp

PATTERNS = {
    "one_B_2": "10101001",
    "diagonal_B_3": "011010100110101001",
    "random_half_3": "100110001011010110",
}
BETAS = (0, 1, 2)
SIZES = (180, 216)
TIMES = tuple(Decimal(i) / 2 for i in range(8, 19))
WINDOWS = ((4, 7), (5, 8), (6, 9))
MODELS = ("log_inv1", "log_inv1_inv2", "log_inv1_inv2_inv3")
PRIMARY_MODEL = "log_inv1_inv2"
FLOOR = 1e-10
GATES = {"coefficient_size_drift": .02, "model_drift": .05,
         "adjacent_window_drift": .05, "D_difference_to_log_span": .01}
PRECISION = 80


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol():
    return {
        "scope": "Time-only diagnostic, not a revised main estimate or an asymptotic-scaling test.",
        "patterns": PATTERNS, "betas": BETAS, "sizes": SIZES,
        "theta": list(map(float, TIMES)), "windows": WINDOWS,
        "models": MODELS, "reference_model": PRIMARY_MODEL,
        "precision_digits": PRECISION, "minimum_n": "parameter_count + 2",
        "gates": GATES, "coefficient_floor": FLOOR,
        "size_and_model_denominator": "abs(c216 of the four-parameter model in that window)",
        "per_model_size_denominator": "abs(c216 of that model in that window)",
        "window_denominator": "abs(c at the same size/model in the earlier adjacent window); both sizes retained, L216 used for local pair diagnostic",
        "adjacent_window_rule": "Original adjacent windows only: 4-7 to 5-8, and 5-8 to 6-9.",
        "local_rule": "Four-parameter size and raw-D gates, all-model agreement at L216, and all six fitted c values nonzero and same sign.",
        "pair_rule": "Both window-local diagnostics and the four-parameter adjacent-window gate; all model-specific window shifts are also retained.",
        "selection": "No window is selected, no equilibrium coefficient is read, and all estimates remain visible.",
        "limitations": ["The rules are a post hoc operational review, not a blind preregistration.",
                        "L180/216 provide only two-size evidence in this calibration.",
                        "Small residuals and passing local diagnostics do not prove asymptotic logarithmic scaling.",
                        "A negative but stable c is a D logarithmic coefficient, not a positive corner contribution.",
                        "Five-parameter fits have seven observations and only two residual degrees of freedom.",
                        "Model/window ranges are deterministic sensitivity ranges, not statistical confidence intervals.",
                        "Ground-state reachability does not imply the time-scaling diagnostics pass."],
    }


def write_csv(path, rows):
    names = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def ratio(value, denominator):
    return None if abs(denominator) < FLOOR else abs(value) / abs(denominator)


def within(value, gate):
    return value is not None and math.isfinite(value) and value <= gate


def same_sign(values):
    return all(abs(x) >= FLOOR for x in values) and (all(x > 0 for x in values) or all(x < 0 for x in values))


def fit(grid, values, model):
    order = MODELS.index(model) + 1
    p = order + 2
    if len(grid) < p + 2:
        raise ValueError("Fewer than parameter_count + 2 actual points")
    with mp.workdps(PRECISION):
        xs = [mp.mpf(str(x)) for x in grid]
        ys = mp.matrix([mp.mpf(str(y)) for y in values])
        design = mp.matrix([[mp.log(x), 1] + [x**(-j) for j in range(1, order+1)] for x in xs])
        coef, _ = mp.qr_solve(design, ys)
        residual = ys - design * coef
        singular = mp.svd(design, compute_uv=False)
        c = coef[0]
        span = abs(c) * mp.log(xs[-1] / xs[0])
        correction = sum(abs(coef[j+1]) * (xs[0]**(-j) - xs[-1]**(-j)) for j in range(1, order+1))
        names = ("c", "b", "inv1", "inv2", "inv3")[:p]
        result = {name: float(coef[j]) for j, name in enumerate(names)}
        result.update({f"{name}_decimal80": mp.nstr(coef[j], PRECISION) for j, name in enumerate(names)})
        result.update(n=len(xs), parameter_count=p, residual_dof=len(xs)-p,
                      rmse=float(mp.sqrt(mp.fsum(x*x for x in residual)/len(xs))),
                      max_abs_residual=float(max(abs(x) for x in residual)),
                      condition=float(singular[0] / singular[len(singular)-1]),
                      log_span=float(span), correction_span=float(correction),
                      correction_to_log_span=None if abs(c) < FLOOR else float(correction/span),
                      grid=";".join(str(x) for x in grid))
        return result


def load_calibration(raw):
    """Only finite calibration rows are retained; no equilibrium fields consumed."""
    expected = set(itertools.product(PATTERNS, BETAS, SIZES, TIMES))
    records = {}
    for line, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))), 2):
        name = row.get("pattern")
        if name not in PATTERNS:
            continue
        theta = Decimal(row["theta"])
        if not theta.is_finite():
            continue
        key = name, Decimal(row["beta"]), int(row["L"]), theta
        if key not in expected:
            continue
        if key in records:
            raise ValueError(f"Duplicate calibration key {key}")
        bits = PATTERNS[name]
        m = math.isqrt(len(bits)//2)
        family = "special" if all(bits[i:i+2] in ("10", "01") for i in range(0, len(bits), 2)) else "general"
        for field, value in (("occupied_bits", bits), ("m", str(m)), ("family", family)):
            if row.get(field) != value:
                raise ValueError(f"Pattern metadata mismatch at line {line}: {field}")
        if key[2] % (3*m):
            raise ValueError(f"Incompatible size at line {line}")
        for field in ("phase_x", "phase_y"):
            if row.get(field) and Decimal(row[field]) != 0:
                raise ValueError(f"Translated source at line {line}")
        for field in ("x0", "y0"):
            if row.get(field) and Decimal(row[field]) != 1:
                raise ValueError(f"Different region origin at line {line}")
        entropy = [Decimal(row[field]) for field in ("S_A", "S_C", "S_B", "D")]
        if not all(x.is_finite() for x in entropy):
            raise ValueError(f"Nonfinite entropy at line {line}")
        if abs((2*entropy[0]+entropy[1])/3-entropy[2]-entropy[3]) > Decimal("1e-9"):
            raise ValueError(f"D identity mismatch at line {line}")
        records[key] = {"D": row["D"], "source_row": line, "run_id": row.get("run_id", "")}
    missing = sorted(expected-records.keys())
    return records, missing


def analyze(records):
    fits, pairs, local, shifts, summaries = [], [], [], [], []
    index = {}
    for pattern, beta, window, model in itertools.product(PATTERNS, BETAS, WINDOWS, MODELS):
        label = f"{window[0]}-{window[1]}"
        grid = [x for x in TIMES if window[0] <= x <= window[1]]
        # Both sizes use this exact seven-point measured grid. Missing data abort
        # before analyze(), so there is no hidden fitting of different grids.
        values = {size: [records[(pattern, beta, size, x)]["D"] for x in grid] for size in SIZES}
        for size in SIZES:
            item = dict(pattern=pattern, beta=beta, L=size, window=label, model=model,
                        **fit(grid, values[size], model))
            index[pattern, beta, label, model, size] = item
            fits.append(item)
        small, large = [index[pattern, beta, label, model, size] for size in SIZES]
        delta = max(abs(float(x)-float(y)) for x, y in zip(values[SIZES[0]], values[SIZES[1]]))
        pairs.append(dict(pattern=pattern, beta=beta, window=label, model=model,
                          L_small=SIZES[0], L_large=SIZES[1], grid=large["grid"],
                          c_small=small["c"], c_large=large["c"],
                          size_drift=ratio(small["c"]-large["c"], large["c"]),
                          max_unshifted_D_difference=delta,
                          D_difference_to_log_span=None if abs(large["c"]) < FLOOR else delta/large["log_span"],
                          signs_agree=same_sign([small["c"], large["c"]])))
    for pattern, beta, window in itertools.product(PATTERNS, BETAS, WINDOWS):
        label = f"{window[0]}-{window[1]}"
        primary = index[pattern, beta, label, PRIMARY_MODEL, SIZES[1]]
        primary_small = index[pattern, beta, label, PRIMARY_MODEL, SIZES[0]]
        model_values = [index[pattern, beta, label, model, SIZES[1]]["c"] for model in MODELS]
        small_values = [index[pattern, beta, label, model, SIZES[0]]["c"] for model in MODELS]
        pair = next(x for x in pairs if (x["pattern"], x["beta"], x["window"], x["model"]) == (pattern, beta, label, PRIMARY_MODEL))
        model_drift = ratio(max(abs(c-primary["c"]) for c in model_values), primary["c"])
        small_model_drift = ratio(max(abs(c-primary_small["c"]) for c in small_values), primary_small["c"])
        signs = same_sign(model_values+small_values)
        reasons = []
        if not within(pair["size_drift"], GATES["coefficient_size_drift"]):
            reasons.append("size_drift")
        if not within(model_drift, GATES["model_drift"]):
            reasons.append("model_drift")
        if not within(pair["D_difference_to_log_span"], GATES["D_difference_to_log_span"]):
            reasons.append("unshifted_D_difference")
        if not signs:
            reasons.append("zero_or_mixed_sign")
        local.append(dict(pattern=pattern, beta=beta, window=label,
                          primary_model=PRIMARY_MODEL, c180=primary_small["c"], c216=primary["c"],
                          size_drift=pair["size_drift"], model_drift=model_drift,
                          model_drift_L180=small_model_drift,
                          max_unshifted_D_difference=pair["max_unshifted_D_difference"],
                          D_difference_to_log_span=pair["D_difference_to_log_span"],
                          all_model_signs_agree=signs, window_local_diagnostic=not reasons,
                          failure_reasons=";".join(reasons),
                          five_parameter_residual_dof=index[pattern,beta,label,MODELS[-1],SIZES[1]]["residual_dof"]))
    local_index = {(x["pattern"],x["beta"],x["window"]):x for x in local}
    for pattern, beta, wi, model, size in itertools.product(PATTERNS, BETAS, range(len(WINDOWS)-1), MODELS, SIZES):
        first = "-".join(map(str,WINDOWS[wi]))
        second = "-".join(map(str,WINDOWS[wi+1]))
        a = index[pattern,beta,first,model,size]["c"]
        b = index[pattern,beta,second,model,size]["c"]
        drift = ratio(b-a,a)
        shifts.append(dict(pattern=pattern,beta=beta,L=size,model=model,
                           earlier_window=first,later_window=second,c_earlier=a,c_later=b,
                           window_drift=drift,signs_agree=same_sign([a,b]),
                           drift_within_gate=within(drift,GATES["adjacent_window_drift"]),
                           adjacent_local_diagnostic=(model==PRIMARY_MODEL and size==SIZES[1] and
                               local_index[pattern,beta,first]["window_local_diagnostic"] and
                               local_index[pattern,beta,second]["window_local_diagnostic"] and
                               same_sign([a,b]) and within(drift,GATES["adjacent_window_drift"]))))
    for pattern,beta in itertools.product(PATTERNS,BETAS):
        p_local=[x for x in local if (x["pattern"],x["beta"])==(pattern,beta)]
        p_shifts=[x for x in shifts if (x["pattern"],x["beta"],x["model"],x["L"])==(pattern,beta,PRIMARY_MODEL,SIZES[1])]
        valid_pairs=[x["earlier_window"]+" -> "+x["later_window"] for x in p_shifts if x["adjacent_local_diagnostic"]]
        status=("adjacent_finite_window_diagnostic_agreement" if valid_pairs else
                "no_adjacent_window_stability_established")
        if not any(x["window_local_diagnostic"] for x in p_local):
            status="all_three_windows_unresolved"
        summaries.append(dict(pattern=pattern,beta=beta,status=status,
                              local_windows=[x["window"] for x in p_local if x["window_local_diagnostic"]],
                              adjacent_diagnostic_pairs=valid_pairs,
                              main_4_7_estimate_preserved=local_index[pattern,beta,"4-7"]["c180"],
                              main_4_7_L216_diagnostic=local_index[pattern,beta,"4-7"]["c216"]))
    return fits,pairs,local,shifts,summaries


def percent(x):
    return "undefined" if x is None else f"{100*x:.3f}%"


def markdown(result):
    lines=["# 三构型有限时间校准复核", "",
           "仅使用有限时间数据，不读取或比较平衡系数。4–7 仍是统一主窗口；本复核不选择更接近平衡的窗口，也不替换主报告估计。", "",
           "每个窗口在 L180/216 上都采用同一组 7 个实际时间点，以 80 位精度独立 QR 拟合。三种模型依次为：", "",
           "$$", "D(\\theta)=c\\log\\theta+b+\\frac{d_1}{\\theta}", "$$", "",
           "$$", "D(\\theta)=c\\log\\theta+b+\\frac{d_1}{\\theta}+\\frac{d_2}{\\theta^2}", "$$", "",
           "$$", "D(\\theta)=c\\log\\theta+b+\\frac{d_1}{\\theta}+\\frac{d_2}{\\theta^2}+\\frac{d_3}{\\theta^3}", "$$", "",
           "各窗以 L216 四参数系数的绝对值为尺寸与模型诊断的分母；相邻窗漂移以较早窗口、相同模型和尺寸的系数绝对值为分母，两个尺寸分别保留，局部相邻窗结论使用 L216。低于 1e−10 时相对诊断未定义。", "",
           "沿用操作性阈值：尺寸漂移 2%、模型和邻窗漂移 5%、未平移 D 尺寸差/拟合 log 跨度 1%。跨模型同号也必须满足。它们只是局部诊断，不能证明渐近 scaling；五参数拟合只有两个残差自由度。没有统计误差条。", "",
           "## 各构型、各 beta 结论", "",
           "| 构型 | beta | 局部诊断符合的窗口 | 相邻窗口诊断 | 状态 |",
           "|---|---:|---|---|---|"]
    labels={"all_three_windows_unresolved":"三个窗口均未稳定",
            "no_adjacent_window_stability_established":"尚无相邻窗口稳定证据",
            "adjacent_finite_window_diagnostic_agreement":"相邻有限窗诊断一致；非渐近证明"}
    for row in result["summary"]:
        lines.append(f"| {row['pattern']} | {row['beta']} | {', '.join(row['local_windows']) or '无'} | {'; '.join(row['adjacent_diagnostic_pairs']) or '无'} | {labels[row['status']]} |")
    lines += ["", "## 三个窗口的四参数读数与失败原因", "",
              "| 构型 | beta | 窗口 | c180 | c216 | 尺寸漂移 | 模型漂移 | D尺寸差/log跨度 | 未通过项 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for row in result["window_diagnostics"]:
        lines.append(f"| {row['pattern']} | {row['beta']} | {row['window']} | {row['c180']:.9g} | {row['c216']:.9g} | {percent(row['size_drift'])} | {percent(row['model_drift'])} | {percent(row['D_difference_to_log_span'])} | {row['failure_reasons'] or '局部符合'} |")
    lines += ["", "所有 162 个单尺寸拟合（3 构型 × 3 beta × 3 窗口 × 3 模型 × 2 尺寸）保存在 calibration_review_fits.csv；81 个尺寸比较和两个尺寸的 108 个相邻窗/模型比较分别保存，失败结果没有删去。", "",
              "本次只能说明所计算尺度上的诊断结果。未通过不等于不存在渐近 scaling；基态可达同样不能推出这些时间窗口已进入 scaling。", "",
              f"源文件 SHA256：`{result['source_sha256']}`。脚本、协议、拟合依赖及产物哈希见 calibration_review.json。", ""]
    return "\n".join(lines)


def self_check():
    """Only synthetic data; never reads source CSV or existing analysis outputs."""
    records={}
    for pattern,beta,size,t in itertools.product(PATTERNS,BETAS,SIZES,TIMES):
        with mp.workdps(PRECISION):
            x=mp.mpf(str(t)); c=mp.mpf("0.17")+mp.mpf(beta)/100
            value=c*mp.log(x)+mp.mpf("0.25")+mp.mpf("0.03")/x
            records[pattern,beta,size,t]={"D":mp.nstr(value,PRECISION)}
    fits,pairs,local,shifts,summaries=analyze(records)
    error=max(abs(x["c"]-(.17+x["beta"]/100)) for x in fits)
    assert error < 1e-13
    assert len(fits)==162 and len(pairs)==81 and len(local)==27 and len(shifts)==108
    assert all(x["window_local_diagnostic"] for x in local)
    assert all(x["status"]=="adjacent_finite_window_diagnostic_agreement" for x in summaries)
    assert ratio(1,0) is None and not same_sign([.1,-.1])
    changed=dict(records)
    for key,value in records.items():
        if key[2]==216:
            changed[key]={"D":str(Decimal(value["D"])+Decimal("0.1"))}
    changed_local=analyze(changed)[2]
    assert all("unshifted_D_difference" in x["failure_reasons"] for x in changed_local)
    return dict(status="PASS", synthetic_only=True, real_source_opened=False,
                core_fits=len(fits), size_pairs=len(pairs), window_pairs=len(shifts),
                max_known_c_error=error,
                checks=["known coefficients across all patterns/models/windows",
                        "correct counts", "zero denominator", "sign reversal",
                        "unshifted D offset rejects false size collapse"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=AI/"data/beta_patterns_observations.csv")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--write-protocol",action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.write_protocol:
        target=args.output_dir/"calibration_review_protocol.json"
        target.write_text(json.dumps(protocol(),indent=2)+"\n",encoding="utf-8")
        print(str(target))
        return
    if args.self_check:
        receipt=self_check()
        receipt["script_sha256"]=sha(Path(__file__).read_bytes())
        (args.output_dir/"calibration_review_self_check.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(receipt))
        return
    source=args.source.resolve()
    raw=source.read_bytes()
    if not raw.endswith(b"\n"):
        raise SystemExit("Source has an incomplete trailing line; wait for completed calibration.")
    records,missing=load_calibration(raw)
    if missing:
        raise SystemExit(f"Calibration incomplete: {len(missing)}/198 required keys missing; no output replaced.")
    fits,pairs,local,shifts,summaries=analyze(records)
    after=source.read_bytes()
    if not after.startswith(raw):
        raise SystemExit("Captured source prefix changed during review; no output replaced.")
    saved_protocol=args.output_dir/"calibration_review_protocol.json"
    if saved_protocol.exists() and json.loads(saved_protocol.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol())):
        raise SystemExit("Saved calibration protocol differs from current script; review the change explicitly.")
    outputs={"calibration_review_fits.csv":fits,"calibration_review_size_pairs.csv":pairs,
             "calibration_review_windows.csv":local,"calibration_review_window_pairs.csv":shifts}
    for name,rows in outputs.items():
        write_csv(args.output_dir/name,rows)
    snapshot=args.output_dir/"calibration_review_source_snapshot.csv"
    snapshot.write_bytes(raw)
    package=Path(mp.__file__).resolve().parent
    mp_files=[(str(p.relative_to(package)).replace("\\","/"),sha(p.read_bytes()))
              for p in sorted(package.rglob("*.py"))]
    result=dict(status="COMPLETE_TIME_ONLY_REVIEW",generated_utc=datetime.now(timezone.utc).isoformat(),
                source=str(source),source_sha256=sha(raw),source_size_bytes=len(raw),required_keys=198,
                source_snapshot=str(snapshot),source_snapshot_sha256=sha(raw),
                source_appended_during_review_bytes=len(after)-len(raw),source_end_sha256=sha(after),
                actual_keys=len(records),missing_keys=[],script_sha256=sha(Path(__file__).read_bytes()),
                arithmetic={"library":"mpmath","version":mp.__version__,"precision_digits":PRECISION,
                            "solver":"independent rectangular QR",
                            "source_tree_sha256":sha(json.dumps(mp_files,separators=(",",":")).encode())},
                protocol=protocol(),protocol_sha256=sha(json.dumps(protocol(),sort_keys=True).encode()),
                protocol_hash_kind="canonical JSON of protocol object, not on-disk file bytes",
                protocol_file_sha256=sha(saved_protocol.read_bytes()) if saved_protocol.exists() else None,
                raw_D_rows=[dict(pattern=k[0],beta=float(k[1]),L=k[2],theta=float(k[3]),**v) for k,v in sorted(records.items())],
                fits=fits,size_pairs=pairs,window_diagnostics=local,window_pairs=shifts,summary=summaries,
                table_sha256={name:sha((args.output_dir/name).read_bytes()) for name in outputs})
    result["narrative_markdown"]=markdown(result)
    result["report_section"]="RESULTS.md#calibration"
    (args.output_dir/"calibration_review.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"keys":len(records),"fits":len(fits),"summary":summaries},indent=2))


if __name__=="__main__":
    main()
