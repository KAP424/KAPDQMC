# Benchmark for the configuration-classification study.
#
# NOTE ON THE FILE NAME: the host has no bash on PATH and the autoresearch harness
# always launches ".auto/measure.sh" through bash, so this file is PowerShell.
#
# Primary metric: pc_patterns (higher is better) - how many configurations the
# classification covers.
# Guardrails:
#   pc_law_maxdev   the exact law a_R = (4/m^2)*M must hold for every (pattern,L)
#   pc_aR_unitary   a_R at ||X||=1 must be the universal constant ~2.07
#   pc_theta_*      theta_max = arctan||X|| must stay inside [45, 90) degrees
#   pc_obstructed   patterns whose Q_F is singular, counted not dropped
#
# Two sweeps, because the protocol needs L to be a multiple of 3m and no single
# ladder satisfies m=1,2 and m=3 at once.  Both write into the same cumulative
# CSVs (rows are merged on (label,L)), so the benchmark is self-contained: running
# it alone reproduces the whole dataset.

$ErrorActionPreference = 'Stop'
if ($PSScriptRoot) { Set-Location (Split-Path -Parent $PSScriptRoot) }
if (-not (Test-Path 'pattern_classes.jl')) {
    Set-Location 'D:\JuliaDQMC\code\KAPDQMC\src\Measure\ai'
}

$t0 = Get-Date

# --- sweep 1: m=1 and m=2, complete enumeration of every half-filled pattern ---
$env:PC_M = '1,2'
$env:PC_LADDER = '12,24,36,48'
$env:PC_CLASSES = '1,2,3'
$env:PC_MAX_PATTERNS = '0'
$env:PC_MAX_CLASS1 = '0'
$env:PC_SMIN_TOL = '1e-8'
$env:PC_PROGRESS = '0'
julia --startup-file=no pattern_classes.jl 2>&1 | Out-Host

# --- sweep 2: m=3, complete class 1 (2^9 = 512) plus a deterministic class-3
# sample.  The full m=3 enumeration is C(18,9)=48620, so classes 2/3 are sampled
# by a fixed stride (never randomly).
$env:PC_M = '3'
$env:PC_LADDER = '18,27,36'
$env:PC_CLASSES = '1,2,3'
$env:PC_MAX_PATTERNS = '256'
$env:PC_MAX_CLASS1 = '0'
julia --startup-file=no pattern_classes.jl 2>&1 | Out-Host

# --- sweep 3: m=4, class 1 only.  The full half-filled enumeration at m=4 is
# C(32,16) = 6.0e8, which is infeasible, and class 1 is 2^16 = 65536, so both are
# reduced by a fixed stride.  This sweep exists to test whether the law and the
# theta_max window are m-universal beyond m=3.
$env:PC_M = '4'
$env:PC_LADDER = '12,24,36'
$env:PC_CLASSES = '1'
$env:PC_MAX_PATTERNS = '0'
$env:PC_MAX_CLASS1 = '512'
julia --startup-file=no pattern_classes.jl 2>&1 | Out-Host

$secs = ((Get-Date) - $t0).TotalSeconds

python .auto/lawcheck.py 2>&1 | Out-Host

# --- sweep 4: the floor family alone, scanned to very large L.  Written to its
# own files (PC_OUT_PREFIX) so it cannot disturb the fixed-L comparisons in the
# main dataset.  m=1 keeps this cheap: q=L and the momentum blocks are 2x1.
$env:PC_M = '1'
$env:PC_LADDER = '48,96,144,192,240,300,360,480'
$env:PC_CLASSES = '1'
$env:PC_MAX_PATTERNS = '0'
$env:PC_MAX_CLASS1 = '0'
$env:PC_OUT_PREFIX = 'floor_ladder'
julia --startup-file=no pattern_classes.jl 2>&1 | Out-Host
$env:PC_OUT_PREFIX = 'pattern_class'

# Build the classification deliverable (regime table, decision rule, hold-out
# validation) and the four figures from the dataset just measured.
python classify_report.py 2>&1 | Out-Host

# Test whether ||X|| is a function of the translation x complement orbit
# (both operations are proved to preserve it, so the orbit is a lower bound on
# the invariant that fixes the level).
python orbit_analysis.py 2>&1 | Out-Host

Write-Host ''
Write-Host ("METRIC pc_seconds={0:N3}" -f $secs)
