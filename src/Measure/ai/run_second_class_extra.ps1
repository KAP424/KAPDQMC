param(
    [ValidateSet('openblas','mkl')][string]$Backend = 'openblas',
    [long]$ProcessorAffinity = 4095,
    [int]$WaitForPid = 0
)
$ErrorActionPreference = 'Stop'
if ($ProcessorAffinity -gt 0) {
    # Opt-in, machine-specific mask; child workers inherit this queue's affinity.
    [Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]$ProcessorAffinity
}
if ($WaitForPid -gt 0) { Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue }
$AiRoot = $PSScriptRoot
$DataRoot = Join-Path $AiRoot 'data'
$Runner = Join-Path $AiRoot 'second_class_extra_experiments.jl'
$StatusPath = Join-Path $DataRoot 'second_class_extra_queue_status.json'
$Stages = @(
    @{ name='limit_m4'; patterns='pure02_staircase4,pure02_notched_stripe4'; sizes='24:12:144'; theta='Inf' },
    @{ name='limit_alternating'; patterns='pure02_alternating_x,pure02_alternating_checker'; sizes='18,24,30,36,42,48,54,60,66,72,78,84,90,108,126,144'; theta='Inf' },
    @{ name='finite_L144'; patterns='pure02_staircase4,pure02_notched_stripe4,pure02_alternating_x,pure02_alternating_checker'; sizes='144'; theta='4:0.5:7' },
    @{ name='finite_L180'; patterns='pure02_staircase4,pure02_notched_stripe4,pure02_alternating_x,pure02_alternating_checker'; sizes='180'; theta='4:0.5:8' }
)
$Status = [ordered]@{
    status='running'
    started_utc=[DateTime]::UtcNow.ToString('o')
    backend=$Backend
    processor_affinity=[Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity.ToInt64()
    stages=@()
}
function Save-Status {
    $Status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
Save-Status
try {
    foreach ($Stage in $Stages) {
        $LogPath = Join-Path $DataRoot ('second_class_extra_' + $Stage.name + '_run.log')
        $Item = [ordered]@{
            name=$Stage.name
            status='running'
            started_utc=[DateTime]::UtcNow.ToString('o')
            log=$LogPath
        }
        $Status.stages += $Item
        Save-Status
        $Arguments = @('--startup-file=no')
        if ($Backend -eq 'mkl') { $Arguments += '--project=C:/Users/admin/.julia/environments/v1.11' }
        $Arguments += @($Runner, ('--backend=' + $Backend), '--threads=8',
            ('--patterns=' + $Stage.patterns), ('--Ls=' + $Stage.sizes),
            ('--thetas=' + $Stage.theta), '--betas=0:0.5:2')
        $Item.arguments=$Arguments
        Save-Status
        Write-Output ('Starting ' + $Stage.name)
        & julia @Arguments *> $LogPath
        if ($LASTEXITCODE -ne 0) { throw ('Julia failed: ' + $Stage.name + ' exit=' + $LASTEXITCODE) }
        $Item.status='complete'
        $Item.finished_utc=[DateTime]::UtcNow.ToString('o')
        Save-Status
        Write-Output ('Completed ' + $Stage.name)
    }
    $Status.status='complete'
} catch {
    $Status.status='failed'
    $Status.error=$_.ToString()
    throw
} finally {
    $Status.finished_utc=[DateTime]::UtcNow.ToString('o')
    Save-Status
}

