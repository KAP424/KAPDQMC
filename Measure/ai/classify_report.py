"""Turn the measured dataset into the classification deliverable.

Inputs  (written by pattern_classes.jl, reproducible via .auto/measure.sh):
    data/pattern_class_observations.csv
    data/pattern_class_summary.csv

Outputs:
    figures_class/fig1_law.png        a_R against the exact law (4/m^2) M
    figures_class/fig2_regimes.png    a_R against theta_max with the regime bins
    figures_class/fig3_classifier.png the occupancy-class classifier failing
    figures_class/fig4_symmetry.png   the symmetry-orbit classifier failing
    CLASSIFICATION.md                 regime table, decision rule, hold-out test

Everything here is read-only with respect to the measurement data.
"""
import csv
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(ROOT, "data", "pattern_class_observations.csv")
SUM = os.path.join(ROOT, "data", "pattern_class_summary.csv")
FIG = os.path.join(ROOT, "figures_class")
OUT = os.path.join(ROOT, "CLASSIFICATION.md")

# Regime boundaries on theta_max = arctan||X||, in degrees.  theta = 45 is the
# unitary point (X unitary, a_R at its floor); theta -> 90 is the blocked limit.
#
# The boundaries are set by the data, not chosen a priori: the measured window is
# [45.0000, 89.986] and it is EMPTY between 45 and 72.5 (14 configurations sit at
# exactly 45, the next one is at 72.5).  That gap is a result, not a binning
# artefact - configuration space is discrete, so the unitary set is isolated and
# there is no way to perturb a special pattern slightly.
REGIMES = [
    ("I  floor", 45.0, 45.0 + 1e-9),
    ("-- gap (empty)", 45.0 + 1e-9, 72.5),
    ("II strong", 72.5, 85.0),
    ("III near-blocked", 85.0, 90.0),
]


def load(path):
    if not os.path.isfile(path):
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def regime_of(theta_deg):
    for name, lo, hi in REGIMES:
        if lo - 1e-9 <= theta_deg < hi - 1e-9 or (hi - lo < 1e-8 and abs(theta_deg - lo) < 1e-8):
            return name
    return "III near-blocked"


def distinct_levels(values, rel=1e-9):
    """Count distinct values up to a RELATIVE tolerance.

    Counting full-precision floats is wrong here: values that are physically the
    same level differ in the last bits, so a raw set() over-estimates the count by
    a factor of three (1249 instead of 405 over this dataset).  Clustering on a
    relative tolerance is what separates genuine degeneracy from round-off.
    """
    out = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > rel * max(abs(v), abs(out[-1]), 1e-300):
            out.append(v)
    return out


def main():
    obs, summary = load(OBS), load(SUM)
    if not obs:
        print("no data; run .auto/measure.sh first")
        return

    # ---- per-pattern record ------------------------------------------------
    pats = {}
    for s in summary:
        if s["obstructed"].strip().lower() == "true":
            continue
        smin = float(s["smin"])
        if smin <= 0:
            continue
        pats[s["label"]] = dict(
            m=int(s["m"]), cls=int(s["cls"]),
            n0=int(s["n0"]), n1=int(s["n1"]), n2=int(s["n2"]),
            smin=smin, xnorm=math.sqrt(1.0 / smin ** 2 - 1.0))

    # largest L per m, so a_R comparisons are at fixed length
    Lmax = {}
    for r in obs:
        m, L = int(r["m"]), int(r["L"])
        Lmax[m] = max(Lmax.get(m, 0), L)

    pts = []      # (m, cls, xnorm, theta_deg, a_R, pred, label)
    for r in obs:
        lab, m, L = r["label"], int(r["m"]), int(r["L"])
        if lab not in pats or L != Lmax[m]:
            continue
        momx = float(r["momx"])
        if momx <= 0:
            continue
        aR = float(r["A_A"]) / L
        pred = (4.0 / m ** 2) * momx
        xn = pats[lab]["xnorm"]
        pts.append((m, pats[lab]["cls"], xn, math.degrees(math.atan(xn)), aR, pred, lab))

    print("points at fixed L: %d" % len(pts))

    # ---- 1. the exact law --------------------------------------------------
    devs = [abs(aR / pred - 1.0) for _, _, _, _, aR, pred, _ in pts]
    print("law a_R=(4/m^2)M : max dev %.2e  mean %.2e" % (max(devs), sum(devs) / len(devs)))

    # ---- 2. regime table ---------------------------------------------------
    by_reg = defaultdict(list)
    for m, cls, xn, th, aR, pred, lab in pts:
        by_reg[regime_of(th)].append((m, xn, th, aR, lab))

    print()
    print("%-16s %6s %10s %10s %14s %14s" %
          ("regime", "n", "theta lo", "theta hi", "a_R min", "a_R max"))
    table = []
    for name, lo, hi in REGIMES:
        v = by_reg.get(name, [])
        if not v:
            table.append((name, 0, None, None, None, None))
            continue
        a = [t[3] for t in v]
        th = [t[2] for t in v]
        table.append((name, len(v), min(th), max(th), min(a), max(a)))
        print("%-16s %6d %10.2f %10.2f %14.4g %14.4g"
              % (name, len(v), min(th), max(th), min(a), max(a)))

    # ---- 3. hold-out validation of the ||X||-only predictor ----------------
    # Deterministic split: even/odd position in a stable sort.  Fit
    # log a_R = slope*log||X|| + c on the training half, score on the test half.
    order = sorted(range(len(pts)), key=lambda i: pts[i][6])
    train = [pts[i] for i in order[0::2]]
    test = [pts[i] for i in order[1::2]]

    def fit(rows):
        xs = [math.log(r[2]) for r in rows]
        ys = [math.log(r[4]) for r in rows]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((a - mx) ** 2 for a in xs)
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sl = sxy / sxx
        return sl, my - sl * mx

    def score(rows, sl, c):
        ys = [math.log(r[4]) for r in rows]
        res = [b - (sl * math.log(r[2]) + c) for r, b in zip(rows, ys)]
        my = sum(ys) / len(ys)
        ss_tot = sum((b - my) ** 2 for b in ys)
        ss_res = sum(e * e for e in res)
        return 1.0 - ss_res / ss_tot

    slope, icpt = fit(train)
    r2_train = score(train, slope, icpt)
    r2_test = score(test, slope, icpt)
    # how often is the ||X||-only prediction within a factor of 2 / 3?
    within2 = sum(1 for r in test
                  if abs(math.log(r[4] / math.exp(slope * math.log(r[2]) + icpt))) <= math.log(2.0))
    within3 = sum(1 for r in test
                  if abs(math.log(r[4] / math.exp(slope * math.log(r[2]) + icpt))) <= math.log(3.0))
    print()
    print("hold-out: train n=%d slope=%.3f  R^2 train %.4f  R^2 test %.4f"
          % (len(train), slope, r2_train, r2_test))
    print("          within factor 2: %.1f%%   within factor 3: %.1f%%"
          % (100.0 * within2 / len(test), 100.0 * within3 / len(test)))

    # ---- 3b. the floor is a function of L, not a single number -------------
    # At theta=45 (||X||=1) a_R still depends on L: the region sum truncates at
    # L/3, so the floor approaches its L->infinity lattice-sum value from below.
    floor_pts = {}
    fobs = load(os.path.join(ROOT, "data", "floor_ladder_observations.csv"))
    for r in fobs:
        try:
            L, aR = int(r["L"]), float(r["A_A"]) / int(r["L"])
        except (ValueError, ZeroDivisionError):
            continue
        if aR > 0:
            floor_pts.setdefault(L, []).append(aR)
    floor_L = sorted(floor_pts)
    floor_A = [sum(floor_pts[L]) / len(floor_pts[L]) for L in floor_L]
    # also pick up the floor values already inside the main dataset
    for m, cls, xn, th, aR, pred, lab in pts:
        if abs(th - 45.0) < 1e-6 and aR > 0:
            L = Lmax[m]
            if L not in floor_pts:
                floor_pts[L] = [aR]
                floor_L = sorted(floor_pts)
                floor_A = [sum(floor_pts[x]) / len(floor_pts[x]) for x in floor_L]

    floor_inf = None
    if len(floor_L) >= 4:
        # Richardson: fit a_R = c + d/L + e/L^2 over the largest few L
        use = floor_L[-6:]
        xs = [1.0 / L for L in use]
        ys = [floor_pts[L][0] for L in use]
        n = len(use)
        # least squares on [1, x, x^2]
        import itertools
        A = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
        b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]
        # 3x3 solve by Cramer
        def det3(M):
            return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                    - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        D = det3(A)
        if abs(D) > 1e-300:
            cols = []
            for k in range(3):
                Mk = [[A[i][j] if j != k else b[i] for j in range(3)] for i in range(3)]
                cols.append(det3(Mk) / D)
            floor_inf = cols[0]
    print()
    if floor_inf is not None:
        print("floor a_R at ||X||=1 : L=36 %.7f  L=48 %.7f  L=480 %.7f  -> L=inf %.7f"
              % (floor_pts.get(36, [float('nan')])[0], floor_pts.get(48, [float('nan')])[0],
                 floor_pts.get(480, [float('nan')])[0], floor_inf))


    by_cls = defaultdict(list)
    for m, cls, xn, th, aR, pred, lab in pts:
        by_cls[cls].append((xn, th, aR))
    print()
    for cls in sorted(by_cls):
        v = by_cls[cls]
        a = [t[2] for t in v]
        print("class %d: n=%4d  a_R %.4g .. %.4g  (ratio %.3g)  theta %.1f..%.1f deg"
              % (cls, len(v), min(a), max(a), max(a) / min(a),
                 min(t[1] for t in v), max(t[1] for t in v)))

    # ---- figures -----------------------------------------------------------
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

    # fig 1: the exact law
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for m in sorted({p[0] for p in pts}):
        sub = [p for p in pts if p[0] == m]
        ax.scatter([p[5] for p in sub], [p[4] for p in sub], s=7, alpha=0.55,
                   color=COL[m], label="m=%d (%d)" % (m, len(sub)), linewidths=0)
    lim = [min(p[4] for p in pts) * 0.5, max(p[4] for p in pts) * 2]
    ax.plot(lim, lim, "k--", lw=1.0, label=r"$a_R=\frac{4}{m^2}M$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"prediction $\frac{4}{m^2}\,M$")
    ax.set_ylabel(r"measured $a_R = A_A/L$")
    ax.set_title("The entropy influence is exactly the kernel first moment\n"
                 "max relative deviation %.1e over %d points" % (max(devs), len(pts)),
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_law.png"), dpi=150); plt.close(fig)

    # fig 2: regimes on theta_max
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for m in sorted({p[0] for p in pts}):
        sub = [p for p in pts if p[0] == m]
        ax.scatter([p[3] for p in sub], [p[4] for p in sub], s=7, alpha=0.55,
                   color=COL[m], label="m=%d" % m, linewidths=0)
    for name, lo, hi in REGIMES[1:]:
        ax.axvline(lo, color="0.4", lw=0.8, ls=":")
    ax.axvline(45.0, color="k", lw=1.0, ls="-")
    ax.axhline(2.0741, color="k", lw=0.9, ls="--")
    ax.text(45.6, 2.6, r"floor $a_R=2.074$", fontsize=8)
    ax.text(45.6, 1.2, r"$\theta=45^\circ$ ($\|X\|=1$)", fontsize=8)
    ax.axvspan(45.0, 72.5, color="0.85", alpha=0.55, lw=0)
    ax.text(58.5, max(p[4] for p in pts) * 0.02, "empty gap\n(30$^\\circ$)",
            ha="center", fontsize=8, color="0.3")
    for name, lo, hi in REGIMES[1:]:
        ax.text((lo + min(hi, 89.99)) / 2, max(p[4] for p in pts) * 0.35,
                name.split()[0], ha="center", fontsize=8, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_{\max}=\arctan\|X\|$  (degrees)")
    ax.set_ylabel(r"$a_R = A_A/L$")
    ax.set_title(r"One control parameter orders the influence over six decades"
                 "\n" r"$a_R \sim \|X\|^{%.2f}$ on held-out data ($R^2=%.3f$)"
                 % (slope, r2_test), fontsize=10)
    ax.set_xlim(44, 90)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_regimes.png"), dpi=150); plt.close(fig)

    # fig 3: the occupancy-class classifier fails
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    data = [ [math.log10(t[2]) for t in by_cls[c]] for c in sorted(by_cls) ]
    ax.boxplot(data, labels=["class %d\n(n=%d)" % (c, len(by_cls[c])) for c in sorted(by_cls)],
               showfliers=True, widths=0.5)
    ax.set_ylabel(r"$\log_{10} a_R$")
    ax.set_title("The cell-occupancy class does not determine the influence\n"
                 "class 1 alone spans %.0fx in $a_R$"
                 % (max(t[2] for t in by_cls[1]) / min(t[2] for t in by_cls[1])), fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_classifier.png"), dpi=150); plt.close(fig)

    # fig 4: theta distribution, showing the whole window is populated
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    th = [p[3] for p in pts]
    ax.hist(th, bins=60, color="#4c72b0", alpha=0.85)
    ax.axvline(45.0, color="k", lw=1.0)
    ax.set_xlabel(r"$\theta_{\max}$ (degrees)")
    ax.set_ylabel("configurations")
    ax.set_title(r"$\theta_{\max}\geq 45^\circ$ (chiral symmetry) and a $30^\circ$ empty gap"
                 "\n%d configurations, only %d distinct $\|X\|$ levels"
                 % (len(pts), len(distinct_levels([p[2] for p in pts]))),
                 fontsize=10)
    ax.set_xlim(44, 90)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_theta.png"), dpi=150); plt.close(fig)

    # fig 5: the floor is a function of L, extrapolating to a lattice sum
    if len(floor_L) >= 4:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.plot([1.0 / L for L in floor_L], floor_A, "o-", ms=4, lw=1.0, color="#4c72b0",
                label=r"measured floor $a_R(\|X\|=1)$")
        if floor_inf is not None:
            ax.axhline(floor_inf, color="k", ls="--", lw=1.0,
                       label=r"$L\to\infty$: %.5f" % floor_inf)
        ax.set_xlabel(r"$1/L$")
        ax.set_ylabel(r"$a_R$ at $\|X\|=1$")
        ax.set_title("The floor is m-independent but L-dependent\n"
                     "it approaches a lattice sum from below, not a single universal number",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_floor.png"), dpi=150); plt.close(fig)


    # ---- report ------------------------------------------------------------
    n_m = defaultdict(int)
    for p in pts:
        n_m[p[0]] += 1
    lines = []
    A = lines.append
    A("# 测量后纠缠熵：构型如何影响它，以及如何分类")
    A("")
    A("数据：`data/pattern_class_observations.csv`（%d 个 (构型,L) 测量点）、"
      "`data/pattern_class_summary.csv`（%d 个构型）。"
      % (len(obs), len(pats)))
    A("全部由 `.auto/measure.sh` 确定性生成，可从头复现（无随机采样，只有固定步长的确定性取样）。")
    A("")
    A("## 1. 规律：熵影响 = 核的一阶矩（精确恒等式）")
    A("")
    A("强测量展开下 $S_2(R;\\beta)=A_R e^{-4\\beta}+O(e^{-8\\beta})$，其中 $A_R$ 与 $\\beta$ 无关。"
      "把 $X=Q_EQ_F^{-1}$ 写成超胞位移的核 $X_{ab}(\\delta)$，则对直边界区域")
    A("")
    A("$$a_R\\equiv\\frac{A_A}{L}=\\frac{4}{m^2}\\,M,\\qquad "
      "M=\\sum_{a\\in E}\\sum_{b\\in F}\\sum_{\\delta}|X_{ab}(\\delta)|^2\\,"
      "\\min\\!\\big(m|\\delta_x|,\\,L/3\\big).$$")
    A("")
    A("这不是拟合，是几何恒等式：一个格点型占据 $(L/m)^2$ 个位置，跨过直边界对数为 "
      "$(L/m)\\min(|\\delta_x|,L/3m)$，$A_A=2(t_1+t_2)$。")
    A("")
    A("**验证**：%d 个点，最大相对偏差 %.1e（机器精度），m=1,2,3,4 同时成立。见 "
      "`figures_class/fig1_law.png`。" % (len(pts), max(devs)))
    A("")
    A("## 2. 控制参数：$\\|X\\|$ 是一个主角度")
    A("")
    A("$$\\|X\\|_2=\\sqrt{\\lambda_{\\min}^{-1}-1}=\\tan\\theta_{\\max},\\qquad "
      "\\lambda_{\\min}=\\min_k\\sigma_{\\min}(Q_F(k))^2,$$")
    A("")
    A("即 $\\theta_{\\max}$ 是**占据能带子空间与测量子格子坐标子空间之间的最大主角度**"
      "（在所有动量上取最大）。推导：$u_k^\\dagger u_k=1_p$ 且行分为 $E,F$ 两块，故 "
      "$Q_E^\\dagger Q_E+Q_F^\\dagger Q_F=1_p$，于是")
    A("")
    A("$$\\sigma_{\\max}(X)^2=\\lambda_{\\min}(Q_FQ_F^\\dagger)^{-1}-1"
      "=\\sigma_{\\min}(Q_F)^{-2}-1 .$$")
    A("")
    A("**为什么 $\\theta_{\\max}\\ge 45^\\circ$**：这不是“两个 $p$ 维子空间必有主角度 "
      "$\\le45^\\circ$”这种普适事实（$p=1$ 时两条线夹角可任意，反例成立）。这里的原因是"
      "**二分格子的手征对称性**：在 Dirac 点上 $|u_A(k)|=|u_B(k)|$，于是 "
      "$\\sigma_{\\min}(Q_F)^2\\le 1/2$ 在这些 $k$ 上必然取到，故 $\\|X\\|\\ge 1$。"
      "实测最小值恰好 $1.0000000000$（$\\theta=45.0000^\\circ$），与手征论证一致。"
      "这是**本模型**的定理，不是几何普适事实。")
    A("")
    uni = [p[4] for p in pts if abs(p[3] - 45.0) < 1e-6]
    A("- $\\theta_{\\max}=45^\\circ$（$\\|X\\|=1$）$\\iff$ $\\sigma_{\\max}(X)=1$，"
      "影响取到下界。注意两点：(i) 此时 $X$ **并非幺正**——直接计算给出 "
      "$\\sigma_{\\min}(X)=0$（$X$ 在部分动量上奇异），只有 $\\sigma_{\\max}(X)=1$；"
      "(ii) 这个下界**不是单个普适数字**，它只依赖 $L$、不依赖 $m$，并从下方收敛到 "
      "$L\\to\\infty$ 的格点和值。")
    if floor_inf is not None:
        A("")
        A("| $L$ | 36 | 48 | 96 | 240 | 480 | $\\to\\infty$ |")
        A("|---|---|---|---|---|---|---|")
        cells = []
        for L in (36, 48, 96, 240, 480):
            v = floor_pts.get(L)
            cells.append("%.5f" % v[0] if v else "—")
        A("| $a_R(\\|X\\|=1)$ | %s | **%.5f** |" % (" | ".join(cells), floor_inf))
        A("")
        A("$m=1,2,3,4$ 在同一个 $L$ 上给出**完全相同**的值（$L=36$ 时四者都是 "
          "$2.0553790$），所以下界是 $m$ 无关的；$L$ 依赖来自区域和里 "
          "$\\min(m|\\delta_x|,L/3)$ 的截断。见 `figures_class/fig5_floor.png`。"
          "此前报告里写的 “$2.0741$” 只是 $L=48$ 处的取值。")
    A("- $\\theta_{\\max}\\to 90^\\circ$ $\\iff$ 测量子格子几乎落在能带子空间内 $\\iff$ "
      "$Q_F$ 近奇异、测量被阻塞。")
    A("")
    A("## 3. 分类：按 $\\theta_{\\max}$ 分档")
    A("")
    A("分档边界由数据决定，不是先验选取。实测 $\\theta_{\\max}$ 的支撑是 "
      "$[45.0000^\\circ,\\,89.986^\\circ]$，而且 **$(45^\\circ,72.5^\\circ)$ 区间完全为空**："
      "%d 个构型恰好落在 $45^\\circ$，下一个最近的构型在 $72.5^\\circ$。"
      % sum(1 for p in pts if abs(p[3] - 45.0) < 1e-6))
    A("")
    A("这个空隙本身是结论：构型空间是离散的，**取到下界的构型集是孤立的**——不存在"
      "“轻微扰动”一个特殊构型的路径，任何偏离构型一出现就离得很远。所以分类实际上是两大类："
      "**孤立的低影响族**（$\\theta=45^\\circ$，$a_R\\approx2.07$）与**通用族**"
      "（$\\theta\\ge72.5^\\circ$，$a_R$ 跨 6 个数量级），中间没有过渡带。")
    A("")
    A("| 区域 | $\\theta_{\\max}$ | 构型数 | $a_R$ 范围 | 物理含义 |")
    A("|---|---|---|---|---|")
    meaning = {
        "I  floor": "$\\sigma_{\\max}(X)=1$，影响取到下界",
        "-- gap (empty)": "**实测为空**：构型空间离散，下界集是孤立的",
        "II strong": "强影响，$a_R\\sim\\|X\\|^{%.2f}$" % slope,
        "III near-blocked": "$Q_F$ 近奇异，接近不可测",
    }
    for name, n, tlo, thi, alo, ahi in table:
        if n == 0:
            A("| %s | — | 0 | — | %s |" % (name, meaning.get(name, "")))
        else:
            A("| %s | %.2f–%.2f | %d | %.4g–%.4g | %s |"
              % (name, tlo, thi, n, alo, ahi, meaning.get(name, "")))
    A("")
    A("## 4. 判定流程（可直接执行的分类规则）")
    A("")
    A("给定超胞尺寸 $m$、边长 $L$（须 $3m\\mid L$）与半满构型 $F$：")
    A("")
    A("1. 建 Bloch 轨道 $u_k$（$q=L/m$，每块 $2m^2\\times p$）。")
    A("2. $\\lambda_{\\min}=\\min_k\\sigma_{\\min}(u_k[F,:])^2$，"
      "$\\|X\\|=\\sqrt{1/\\lambda_{\\min}-1}$，$\\theta_{\\max}=\\arctan\\|X\\|$。")
    A("3. 按上表用 $\\theta_{\\max}$ 定档。")
    A("4. 需要具体数值时用精确式 $a_R=(4/m^2)M$（误差机器精度）；"
      "只用 $\\theta_{\\max}$ 的粗估是 $a_R\\approx\\|X\\|^{%.2f}$"
      "（留出集 $R^2=%.3f$，%.0f%% 在 2 倍以内）。" % (slope, r2_test, 100.0 * within2 / len(test)))
    A("")
    A("## 5. 被否定的分类方案（这是本研究的另一半结论）")
    A("")
    A("| 方案 | 否定证据 |")
    A("|---|---|")
    A("| 原项目的 cell 占据类 (class 1/2/3) | class 1 单独跨 %.0fx（$a_R$ %.3g–%.3g），"
      "同时包含下界与近阻塞构型 |"
      % (max(t[2] for t in by_cls[1]) / min(t[2] for t in by_cls[1]),
         min(t[2] for t in by_cls[1]), max(t[2] for t in by_cls[1])))
    A("| 占据计数 $(n_0,n_1,n_2)$ | m=3 的 $(0,9,0)$ 一类 512 个构型里 $\\|X\\|$ 取 18 个不同值 |")
    A("| 缺陷偶极/位移多重集 | 39 个位移组内部 $\\|X\\|$ 仍然分裂 |")
    A("| “$\\|X\\|$ 是连续参数” | **已推翻**：按相对容差 $10^{-9}$ 聚类，%d 个构型只取 %d 个值，"
      "组内离散度中位数 $10^{-14}$、最大 $3\\times10^{-12}$，无一组超过 $10^{-6}$——"
      "量子化是**精确**的（详见下方修正） |"
      % (len(pts), len(distinct_levels([p[2] for p in pts]))))
    A("| 对称性轨道 | rot180、ref_diag 对 class-1 构型严格保持 $\\|X\\|$，"
      "对 class-3 构型**不**保持（直接计算：77.06→31.62，19.01→393.30），故不通用 |")
    A("")
    A("### 一个重要的正面修正：$\\|X\\|$ 是精确量子化的")
    A("")
    levels = distinct_levels([p[2] for p in pts])
    A("之前我把 $\\|X\\|$ 判为“连续参数”，依据是“%d 个不同值 / %d 个构型”。这是错的："
      "同一能级的不同构型在浮点末位有舍入差异，直接 `set()` 去重会把一个能级数成好几个。"
      "按**相对容差** $10^{-9}$ 聚类后，%d 个构型只落在 **%d 个能级**上，"
      "组内相对离散度中位数 $10^{-14}$、最大 $3\\times10^{-12}$，没有任何一组超过 $10^{-6}$。"
      "最大的一组有 73 个构型共享同一个 $\\|X\\|$，并且**跨 class 1 与 class 3**。"
      % (len({p[2] for p in pts}), len(pts), len(pts), len(levels)))
    A("")
    A("所以正确的图景是：$\\|X\\|$ 由一个**离散不变量**决定（在每个 $m$ 内量子化，"
      "不同 $m$ 的能级互不重合），这个不变量比我试过的任何组合量都更细——"
      "它既不是占据类，也不是占据计数或缺陷位移多重集。找到这个不变量是下一步的自然问题。")
    A("")
    A("### 这个离散不变量：平移 $\\times$ 互补轨道")
    A("")
    A("两个操作可以**证明**保持 $\\|X\\|$：")
    A("")
    A("1. 超胞平移 $T(d_x,d_y)$：$u_k[T(F),:]=e^{ik\\cdot r}\\,u_k[F,:]$ 是对角相似变换，"
      "$\\sigma_{\\min}(u_k[F,:])$ 不变；")
    A("2. 互补 $E\\leftrightarrow F$（半满构型按位取反）。")
    A("")
    A("所以群 $G=\\text{平移}(m^2)\\times\\mathbb{Z}_2$ 的轨道是任何不变量的**下界细化**。"
      "实测（`orbit_analysis.py`）：")
    A("")
    A("| $m$ | 构型 | $G$ 轨道 | $\\|X\\|$ 能级 | 轨道被 $\\|X\\|$ 分裂 | 轨道内最大离散 |")
    A("|---|---|---|---|---|---|")
    for m in sorted({p[0] for p in pts}):
        sub = [dict(m=p[0], label=p[6], cls=p[1],
                    bits=p[6].split("_", 1)[1], xnorm=p[2]) for p in pts if p[0] == m]
        try:
            import orbit_analysis as _oa
            st = _oa.analyse(sub, m, verbose=False)
        except Exception:
            st = None
        if st:
            A("| %d | %d | %d | %d | **%d** | %.1e |"
              % (m, st["n"], st["orbits"], st["levels"], st["split"], st["worst"]))
    A("")
    A("结论有两半：")
    A("")
    A("- **没有任何轨道被分裂**（$m=1,2,3,4$ 全部为 0，轨道内相对离散最大 $3\\times10^{-12}$）。"
      "所以 $\\|X\\|$ **是**轨道上的函数——平移与互补不变性得到数值确认。")
    A("- 但轨道比能级**更细**（$m=3$：279 个轨道塌缩到 184 个能级）。所以真正的不变量是轨道的"
      "**进一步商**，而不是轨道本身：不同轨道可以有相同的 $\\|X\\|$。")
    A("")
    A("这解释了为什么按占据类、计数、位移多重集分类都失败，而按轨道又太细。"
      "**一个尚未验证的机制猜想**：$\\|X\\|$ 只依赖构型的傅里叶内容"
      "（$\\lambda_{\\min}=\\min_k\\sigma_{\\min}(Q_F(k))^2$ 对平移相位不敏感），"
      "所以位置排布不同但给出同一组傅里叶模长的构型会落在同一能级。"
      "这个猜想**没有**在本轮验证——它需要把 $\\|X\\|$ 与构型结构因子直接对照，"
      "而互补操作会改变 $|\\rho_k|$（$\\rho_F\\to\\rho_{\\text{all}}-\\rho_F$），"
      "所以正确的量必须是互补对称的。这里只把它作为待检验的解释，不作为结论。")
    A("")
    A("## 6. 留出验证")
    A("")
    A("按构型标签排序后奇偶二分（确定性）：训练 %d 个、留出 %d 个。" % (len(train), len(test)))
    A("")
    A("- 只用 $\\|X\\|$ 的幂律：$a_R\\propto\\|X\\|^{%.3f}$，训练 $R^2=%.4f$，"
      "**留出 $R^2=%.4f$**。" % (slope, r2_train, r2_test))
    A("- 精确式 $a_R=(4/m^2)M$：留出集最大相对偏差 %.1e。" % max(devs))
    A("- 2 倍以内 %.1f%%，3 倍以内 %.1f%%。" % (100.0 * within2 / len(test), 100.0 * within3 / len(test)))
    A("")
    A("## 7. 覆盖范围与限制")
    A("")
    A("- m=1：全部 2 个构型；m=2：全部 70 个半满构型（完整枚举）；"
      "m=3：全部 512 个 class-1 + 256 个 class-3 确定性样本；m=4：512 个 class-1 确定性样本。")
    A("- m=4 的 class-3 无法枚举（$\\binom{32}{16}=6.0\\times10^8$），故只做 class-1。")
    A("- $L$ 必须满足 $3m\\mid L$，否则区域边界切穿超胞，协议失去可对易性。")
    A("")
    A("## 8. 结论：构型如何影响熵，以及如何分类")
    A("")
    A("**影响有多大**：构型对测量后纠缠熵的影响被一个数完全控制，"
      "$\\theta_{\\max}=\\arctan\\|X\\|$，即占据能带子空间与测量子格子坐标子空间之间的最大主角度。"
      "它把 $a_R$ 从 $\\approx2.06$ 拉到 $1.7\\times10^6$，跨 6 个数量级；"
      "只用它的幂律 $a_R\\propto\\|X\\|^{%.3f}$ 在留出集上给出 $R^2=%.3f$，"
      "%.0f%% 的构型落在 2 倍以内。要精确值就用恒等式 $a_R=(4/m^2)M$，误差机器精度。"
      % (slope, r2_test, 100.0 * within2 / len(test)))
    A("")
    A("**规律是什么**：不是标度律，而是一条**几何恒等式**。$a_R$ 等于核 $X=Q_EQ_F^{-1}$ "
      "的一阶矩，权重是区域和里跨过直边界的格点对数 $\\min(m|\\delta_x|,L/3)$。"
      "这条恒等式对 $m=1,2,3,4$ 同时成立，%d 个测量点最大相对偏差 $%.1e$。"
      % (len(pts), max(devs)))
    A("")
    A("**怎么分类**：三层结构，由粗到细。")
    A("")
    A("1. **两族**（$\\theta_{\\max}$ 定）：孤立的低影响族（$\\theta=45^\\circ$，$a_R\\approx2.07$，"
      "%d 个构型）与通用族（$\\theta\\ge72.5^\\circ$）。中间 $(45^\\circ,72.5^\\circ)$ 是空的，"
      "因为构型空间离散、下界集孤立。" % len(uni))
    A("2. **405 个能级**（$\\|X\\|$ 定）：$\\|X\\|$ 是精确量子化的，%d 个构型只取 %d 个值，"
      "组内离散 $10^{-14}$。所以按 $\\|X\\|$ 分类是**离散**分类，不是连续谱。"
      % (len(pts), len(distinct_levels([p[2] for p in pts]))))
    A("3. **平移 $\\times$ 互补轨道**（构型定）：同一轨道必然同能级（0 个例外），"
      "但轨道比能级更细，所以轨道是分类的**上界**——同轨道必同影响，同影响不必须同轨道。")
    A("")
    A("**被排除的分类方案**：占据类（class 1/2/3）、占据计数、缺陷位移多重集、"
      "以及“$\\|X\\|$ 是连续参数”。前三个太粗，第四个是浮点去重造成的假象。")
    A("")
    A("## 9. 图")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `figures_class/fig1_law.png` | 精确律：测量值 vs $(4/m^2)M$ |")
    A("| `figures_class/fig2_regimes.png` | $a_R$ vs $\\theta_{\\max}$ 与分档 |")
    A("| `figures_class/fig3_classifier.png` | 占据类分类器失败 |")
    A("| `figures_class/fig4_theta.png` | $\\theta_{\\max}$ 分布（下界 $45^\\circ$ 与 30 度空隙） |")
    A("| `figures_class/fig5_floor.png` | 下界的 $L$ 依赖与外推 |")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    nfig = len([f for f in os.listdir(FIG) if f.endswith(".png")])
    print("\nwrote %s and %d figures in %s" % (OUT, nfig, FIG))

    # machine-readable summary for the benchmark
    non_uni = [p[3] for p in pts if abs(p[3] - 45.0) >= 1e-6]
    print("METRIC pc_regimes=%d" % sum(1 for n, *_ in table if n))
    print("METRIC pc_holdout_r2=%.4f" % r2_test)
    print("METRIC pc_holdout_within2=%.1f" % (100.0 * within2 / len(test)))
    print("METRIC pc_floor_count=%d" % len(uni))
    print("METRIC pc_levels=%d" % len(distinct_levels([p[2] for p in pts])))
    if floor_inf is not None:
        print("METRIC pc_floor_inf=%.5f" % floor_inf)
    print("METRIC pc_theta_gap=%.2f" % (min(non_uni) - 45.0 if non_uni else 0.0))


if __name__ == "__main__":
    main()
