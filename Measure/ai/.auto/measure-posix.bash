#!/usr/bin/env bash
# POSIX version of the stable benchmark (kept for portability/documentation).
# This Windows host has no bash on PATH, so .auto/measure.sh is the PowerShell
# twin that the harness actually executes via: pwsh -NoProfile -File .auto/measure.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export SS_PATH=dense
export SS_PATTERNS=uniform_A,one_B_2
export SS_LADDER=12,18,24,30,36,42,48
export SS_XCHECK_LMAX=48
export SS_BETA=4.0
export SS_DENSE_ORACLE_LMAX=0
julia --startup-file=no strong_scaling.jl
