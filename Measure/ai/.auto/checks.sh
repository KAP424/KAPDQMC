#!/usr/bin/env pwsh
# Correctness gate for the post-measurement entropy classification benchmark.
# Re-derives every headline number from the CSVs and checks the report agrees.
# Fails the run if the law, the level count, the floor extrapolation or the
# theta window regress.
#
# Invoked either as a script file or piped into `powershell -Command -`, so the
# root is resolved without relying on $MyInvocation (empty when piped).
$ErrorActionPreference = 'Stop'
$root = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Push-Location $root
try {
    python .auto/verify_report.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "checks failed with exit $LASTEXITCODE"
        exit 1
    }
} finally {
    Pop-Location
}
exit 0
