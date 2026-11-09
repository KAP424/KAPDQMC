# Auditable consolidation driver.
#
#   Get-Content .auto/consolidate.ps1 -Raw | powershell -NoProfile -Command - -Batch A
#
# Batch A: regenerable caches, vendored third-party code, byte-identical
#          duplicates, the single-file-per-directory provenance tree, and
#          interrupted-run leftovers.
# Batch B: run logs, per-run toml/json manifests, source_snapshots, and the one
#          analysis directory no document references.
#
# Hard guards (the script aborts instead of deleting):
#   * a candidate must not be a markdown link target, nor contain one
#   * every candidate must be present in the archive before the delete happens
#   * every candidate is recorded in .auto/removed_manifest.csv with its sha256
param([string]$Batch = $env:BATCH)
if (-not $Batch) { $Batch = 'A' }
$DryRun = ($env:DRYRUN -eq '1')

$ErrorActionPreference = 'Stop'
if ($PSScriptRoot) { Set-Location (Split-Path -Parent $PSScriptRoot) }
if (-not (Test-Path strong_scaling.jl)) { Set-Location (Split-Path -Parent (Resolve-Path .auto/consolidate.ps1)) }
$root = (Get-Location).Path
$zipPath = Join-Path $root 'data/archive_20261103.zip'
$manifestPath = Join-Path $root '.auto/removed_manifest.csv'

# ---------------------------------------------------------------- doc targets
$linkRe = [regex]'!?\[[^\]]*\]\(([^)]+)\)'
$protected = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($doc in @('README.md', 'RESULTS.md', 'SCALING_GUIDE.md', 'neq.md')) {
    if (-not (Test-Path $doc)) { continue }
    $text = Get-Content $doc -Raw
    foreach ($m in $linkRe.Matches($text)) {
        $t = $m.Groups[1].Value.Trim().Trim('<', '>').Split('#')[0].Split('?')[0]
        if (-not $t -or $t -match '://' -or $t -match '^mailto:') { continue }
        $full = [System.IO.Path]::GetFullPath((Join-Path $root ($t -replace '/', '\')))
        [void]$protected.Add($full)
    }
}
Write-Host ("doc link targets that exist: {0}" -f (@($protected | Where-Object { Test-Path $_ }).Count))

# --------------------------------------------------------------- code corpus
# Concatenated production code, used to protect any file that a script names
# explicitly (so collapsing a duplicate never dangles a reference).
$script:codeText = ((Get-ChildItem -Recurse -File -Include *.py, *.jl, *.ps1, *.bash |
        Where-Object { $_.FullName -notlike '*\.auto\*' } |
        ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n")
$script:docText = ((Get-ChildItem -File -Filter *.md | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n")

# ------------------------------------------------------------------ candidates
$files = Get-ChildItem -Recurse -File
$skip = @('.auto\', 'data\archive_20261103.zip')
$candidates = New-Object System.Collections.Generic.List[object]
$reasons = @{}

function Add-Cand([string]$full, [string]$reason) {
    if ($skip | Where-Object { $full -like ('*' + $_) }) { return }
    if (-not $reasons.ContainsKey($full)) { $candidates.Add($full); $reasons[$full] = $reason }
}

if ($Batch -eq 'A') {
    # (a) caches
    foreach ($d in (Get-ChildItem -Recurse -Directory -Filter '__pycache__')) {
        foreach ($f in (Get-ChildItem $d.FullName -Recurse -File)) { Add-Cand $f.FullName 'pycache' }
    }
    foreach ($f in ($files | Where-Object { $_.Extension -eq '.pyc' })) { Add-Cand $f.FullName 'pycache' }

    # (b) vendored third-party dependency snapshot
    $vend = Join-Path $root 'analysis_beta_patterns\audit_dependencies'
    if (Test-Path $vend) {
        foreach ($f in (Get-ChildItem $vend -Recurse -File)) { Add-Cand $f.FullName 'vendored' }
    }

    # (c) byte-identical duplicates: keep the canonical (hash-free, shortest, shallowest) name
    $hashToken = [regex]'[0-9a-f]{12,}'
    $pool = $files | Where-Object {
        $_.Extension -ne '.pyc' -and
        $_.FullName -notlike '*\__pycache__\*' -and
        $_.FullName -notlike '*\audit_dependencies\*' -and
        $_.FullName -notlike '*\.auto\*' -and
        $_.FullName -notlike '*\data\provenance\*' -and
        $_.Length -gt 0
    }
    foreach ($g in ($pool | Group-Object Length | Where-Object { $_.Count -gt 1 })) {
        $byHash = @{}
        foreach ($f in $g.Group) {
            $h = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
            if (-not $byHash.ContainsKey($h)) { $byHash[$h] = New-Object System.Collections.Generic.List[object] }
            $byHash[$h].Add($f)
        }
        foreach ($h in $byHash.Keys) {
            $arr = $byHash[$h]
            if ($arr.Count -lt 2) { continue }
            $sorted = $arr | Sort-Object `
                @{Expression = { if ($hashToken.IsMatch($_.Name)) { 1 } else { 0 } } }, `
                @{Expression = { $_.Name.Length } }, `
                @{Expression = { $_.FullName.Split('\').Count } }, `
                @{Expression = { $_.FullName } }
            $keep = $sorted[0]
            for ($i = 1; $i -lt $sorted.Count; $i++) {
                # never drop a duplicate whose basename production code mentions by
                # name: the surviving twin has a different name, so the reference
                # would dangle even though the bytes are identical.
                if ($script:codeText.Contains($sorted[$i].Name)) { continue }
                Add-Cand $sorted[$i].FullName ('dup-of:' + $keep.Name)
            }
        }
    }

    # (d) provenance tree: 37 directories each holding exactly one file
    $prov = Join-Path $root 'data\provenance'
    if (Test-Path $prov) { foreach ($f in (Get-ChildItem $prov -Recurse -File)) { Add-Cand $f.FullName 'provenance' } }

    # (e) interrupted-run leftovers
    foreach ($f in ($files | Where-Object { $_.Name -like '*.partial-*' })) { Add-Cand $f.FullName 'partial' }
}

if ($Batch -eq 'B') {
    foreach ($f in (Get-ChildItem (Join-Path $root 'data') -File | Where-Object {
                $_.Extension -in '.log', '.toml', '.json', '.partial-20261018T105830701' -and
                $_.Name -notlike 'archive_*' -and
                -not $script:codeText.Contains($_.Name)
            })) { Add-Cand $f.FullName 'run-metadata' }
    foreach ($d in (Get-ChildItem -Recurse -Directory -Filter 'source_snapshots')) {
        foreach ($f in (Get-ChildItem $d.FullName -Recurse -File)) { Add-Cand $f.FullName 'source-snapshot' }
    }
    $unref = Join-Path $root 'analysis_random_half_bloch'
    if (Test-Path $unref) { foreach ($f in (Get-ChildItem $unref -Recurse -File)) { Add-Cand $f.FullName 'unreferenced-dir' } }
}

if ($Batch -eq 'C') {
    # Unreferenced figure artifacts: neither a link target nor even mentioned by
    # basename anywhere in the docs, so nothing in the documentation points at
    # them.  A file that is merely mentioned in prose is kept.
    foreach ($f in (Get-ChildItem -Recurse -File -Include *.png, *.pdf | Where-Object {
                $_.DirectoryName -match 'analysis_' -or $_.DirectoryName -match '\\(figures_|figs_)'
            })) {
        if ($protected.Contains($f.FullName)) { continue }
        if ($script:docText.Contains($f.Name)) { continue }
        Add-Cand $f.FullName 'orphan-figure'
    }
}

# ------------------------------------------------------- guard: docs and code
# Protected files are dropped from the batch, not fatal: a directory tree whose
# members are individually protected simply keeps those members (empty-dir
# cleanup only removes directories that are genuinely empty afterwards).
$protectedHits = New-Object System.Collections.Generic.List[string]
$kept = New-Object System.Collections.Generic.List[object]
foreach ($c in $candidates) {
    $isP = $protected.Contains($c)
    if (-not $isP) {
        $lprefix = ($c + '\').ToLowerInvariant()
        foreach ($p in $protected) { if ($p.ToLowerInvariant().StartsWith($lprefix)) { $isP = $true; break } }
    }
    if ($isP) { $protectedHits.Add($c) } else { $kept.Add($c) }
}
if ($protectedHits.Count -gt 0) {
    Write-Host ("protected (referenced by documentation, kept): {0} file(s)" -f $protectedHits.Count)
    $protectedHits | Select-Object -First 8 | ForEach-Object { Write-Host ("   " + $_.Substring($root.Length + 1)) }
}
$candidates = $kept

$totalBytes = 0
foreach ($c in $candidates) { $totalBytes += (Get-Item $c).Length }Write-Host ("batch {0}: {1} candidate files, {2:N2} MB, {3} protected doc targets" -f `
        $Batch, $candidates.Count, ($totalBytes / 1MB), $protected.Count)
$candidates | Group-Object { $reasons[$_] } | Sort-Object Count -Descending |
    ForEach-Object { Write-Host ("   {0,-16} {1,5} files  {2,8:N2} MB" -f $_.Name, $_.Count, (($_.Group | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum / 1MB)) }

if ($candidates.Count -eq 0) { Write-Host "nothing to do"; exit 0 }

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN - no file will be archived or deleted. Sample:"
    $n = if ($env:DRYRUN_ALL -eq '1') { $candidates.Count } else { 25 }
    $candidates | Select-Object -First $n | ForEach-Object {
        Write-Host ("   [{0}] {1}" -f $reasons[$_], $_.Substring($root.Length + 1))
    }
    exit 0
}

# ------------------------------------------------------------------- archive
Add-Type -AssemblyName System.IO.Compression | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $zipPath) | Out-Null
$fs = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::OpenOrCreate)
$zip = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Update)
$rows = New-Object System.Collections.Generic.List[string]
if (-not (Test-Path $manifestPath)) { $rows.Add('path,bytes,sha256,batch') }
foreach ($c in $candidates) {
    $rel = $c.Substring($root.Length + 1)
    $entryName = $rel -replace '\\', '/'
    $entry = $zip.GetEntry($entryName)
    if ($entry) { $entry.Delete() }
    $entry = $zip.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
    $es = $entry.Open()
    $src = [System.IO.File]::OpenRead($c)
    $src.CopyTo($es)
    $src.Dispose(); $es.Dispose()
    $len = (Get-Item $c).Length
    $sha = (Get-FileHash $c -Algorithm SHA256).Hash
    $rows.Add(('"{0}",{1},{2},{3}' -f $rel, $len, $sha, $Batch))
}
$zip.Dispose(); $fs.Dispose()
$rows | Add-Content -Path $manifestPath -Encoding UTF8

# --------------------------------------------------------------------- verify
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$names = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($e in $zip.Entries) { [void]$names.Add($e.FullName) }
$missing = 0
foreach ($c in $candidates) {
    $entryName = ($c.Substring($root.Length + 1)) -replace '\\', '/'
    if (-not $names.Contains($entryName)) { $missing++; Write-Host ("  MISSING FROM ARCHIVE: " + $entryName) -ForegroundColor Red }
}
$entryCount = $zip.Entries.Count
$zip.Dispose()
if ($missing -gt 0) { Write-Host "ABORT: archive incomplete, nothing deleted" -ForegroundColor Red; exit 3 }
Write-Host ("archive ok: {0} entries, all {1} candidates present" -f $entryCount, $candidates.Count)

# ---------------------------------------------------------------------- delete
$failed = 0
foreach ($c in $candidates) {
    try { Remove-Item -LiteralPath $c -Force -ErrorAction Stop }
    catch { $failed++; Write-Host ("  could not delete " + $c + " : " + $_.Exception.Message) -ForegroundColor Yellow }
}
# drop directories left empty
for ($pass = 0; $pass -lt 6; $pass++) {
    $empties = Get-ChildItem -Recurse -Directory | Where-Object {
        (Get-ChildItem $_.FullName -Force | Measure-Object).Count -eq 0 -and $_.Name -ne '.auto'
    }
    if (-not $empties) { break }
    foreach ($d in $empties) { Remove-Item -LiteralPath $d.FullName -Force -Recurse -ErrorAction SilentlyContinue }
}
$left = (Get-ChildItem -Recurse -File).Count
$dirsLeft = (Get-ChildItem -Recurse -Directory).Count
Write-Host ("deleted {0} files ({1} failed); tree now {2} files / {3} dirs" -f `
        ($candidates.Count - $failed), $failed, $left, $dirsLeft)
