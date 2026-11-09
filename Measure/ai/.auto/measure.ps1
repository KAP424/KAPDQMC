# Baseline segment (run 1): dense oracle route, L <= 48.
# Same command sequence as measure.sh; this host has no bash, so pwsh drives it.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:SS_PATH            = 'dense'
$env:SS_PATTERNS        = 'uniform_A,one_B_2'
$env:SS_LADDER          = '12,18,24,30,36,42,48'
$env:SS_XCHECK_LMAX     = '48'
$env:SS_BETA            = '4.0'
$env:SS_DENSE_ORACLE_LMAX = '0'
julia --startup-file=no strong_scaling.jl
