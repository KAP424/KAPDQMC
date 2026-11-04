param([ValidateSet('openblas','mkl')][string]$Backend = 'openblas')
$ErrorActionPreference = 'Stop'
$Workspace = 'D:\JuliaDQMC\code\KAPDQMC'
$AiDirectory = Join-Path $Workspace 'src\Measure\ai'
$PlanPath = Join-Path $AiDirectory 'analysis_beta_patterns\plan.json'
$OutputPath = Join-Path $AiDirectory 'data\beta_patterns_observations.csv'
$StatusPath = Join-Path $AiDirectory 'data\beta_patterns_queue_status.json'
$RunnerPath = Join-Path $AiDirectory 'beta_patterns_experiments.jl'
$Plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding utf8 | ConvertFrom-Json
$Stages = @(
    @{ Name = 'calibration'; Grid = $Plan.required_finite_grids[2] },
    @{ Name = 'equilibrium_one_B_2'; Grid = $Plan.required_equilibrium_grids[0] },
    @{ Name = 'equilibrium_m3'; Grid = $Plan.required_equilibrium_grids[1] },
    @{ Name = 'finite_primary'; Grid = $Plan.required_finite_grids[0] },
    @{ Name = 'finite_shift'; Grid = $Plan.required_finite_grids[1] }
)
$Status = [ordered]@{
    status = 'running'
    started_utc = [DateTime]::UtcNow.ToString('o')
    plan_path = $PlanPath
    plan_sha256 = (Get-FileHash -LiteralPath $PlanPath -Algorithm SHA256).Hash.ToLowerInvariant()
    runner_path = $RunnerPath
    blas_backend = $Backend
    output = $OutputPath
    stages = @()
}
function Write-QueueStatus {
    $Status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
function Join-GridValues($Values) {
    [string]::Join(',', @($Values | ForEach-Object {
        if ($_ -is [System.IFormattable]) { $_.ToString($null, [Globalization.CultureInfo]::InvariantCulture) }
        else { [string]$_ }
    }))
}
Set-Location -LiteralPath $Workspace
Write-QueueStatus
try {
    foreach ($Stage in $Stages) {
        $Grid = $Stage.Grid
        $LogPath = Join-Path $AiDirectory ('data\beta_patterns_' + $Stage.Name + '_run.log')
        $StageStatus = [ordered]@{
            name = $Stage.Name
            status = 'running'
            started_utc = [DateTime]::UtcNow.ToString('o')
            log = $LogPath
        }
        $Status.stages += $StageStatus
        Write-QueueStatus
        $Arguments = @(
            '--startup-file=no'
        )
        if ($Backend -eq 'mkl') { $Arguments += '--project=C:/Users/admin/.julia/environments/v1.11' }
        $Arguments += @(
            $RunnerPath, '--mode=run', '--threads=8', ('--backend=' + $Backend),
            '--method=real_cholesky', ('--output=' + $OutputPath),
            ('--patterns=' + (Join-GridValues $Grid.patterns)),
            ('--Ls=' + (Join-GridValues $Grid.L)),
            ('--thetas=' + (Join-GridValues $Grid.theta)),
            ('--betas=' + (Join-GridValues $Grid.beta))
        )
        $StageStatus.arguments = $Arguments
        Write-QueueStatus
        Write-Output ('Starting ' + $Stage.Name + ': ' + [DateTime]::UtcNow.ToString('o'))
        & julia @Arguments *> $LogPath
        if ($LASTEXITCODE -ne 0) { throw ('Julia failed in stage ' + $Stage.Name + ' with exit code ' + $LASTEXITCODE) }
        $StageStatus.status = 'complete'
        $StageStatus.finished_utc = [DateTime]::UtcNow.ToString('o')
        Write-QueueStatus
        Write-Output ('Completed ' + $Stage.Name + ': ' + [DateTime]::UtcNow.ToString('o'))
    }
    $Status.status = 'complete'
} catch {
    $Status.status = 'failed'
    $Status.error = $_.ToString()
    if ($null -ne $StageStatus) { $StageStatus.status = 'failed' }
    throw
} finally {
    $Status.finished_utc = [DateTime]::UtcNow.ToString('o')
    Write-QueueStatus
}
