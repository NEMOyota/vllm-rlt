#requires -Version 5.1
# Read existing Chrome traces and probe results. All sizes printed are bytes.
param([string]$Root = 'D:\image-deploy\results\paged-prefill-probe')
$ErrorActionPreference = 'Stop'

function Read-JsonFile([string]$Path) {
    $raw = [IO.File]::ReadAllText($Path)
    try { $obj = ConvertFrom-Json -InputObject $raw }
    catch {
        # Windows PowerShell cannot represent case-distinct JSON keys on PSCustomObject.
        if ($PSVersionTable.PSEdition -ne 'Desktop') { throw }
        Add-Type -AssemblyName System.Web.Extensions
        $parser = New-Object System.Web.Script.Serialization.JavaScriptSerializer
        $parser.MaxJsonLength = [int]::MaxValue
        $parser.RecursionLimit = 256
        $obj = $parser.DeserializeObject($raw)
    }
    if ($raw.TrimStart().StartsWith('[') -and $obj -isnot [array]) { $obj = @($obj) }
    return ,$obj
}

function Get-TransferBytes($Meta) {
    # Prefer explicitly byte-labelled fields over a generic size. Never use duration,
    # bandwidth, tensor shape or allocated memory as a transfer byte count.
    foreach ($key in @('bytes', 'num bytes', 'byte count', 'size(bytes)', 'size (bytes)',
                        'size(B)', 'size (B)', 'data size(bytes)', 'data size(byte)',
                        'data size(B)', 'data size', 'size', 'count(bytes)')) {
        $value = $Meta.$key
        if ($null -eq $value) { continue }
        $s = ([string]$value).Trim()
        if ($s -match '^(\d+|\d{1,3}(?:,\d{3})+)(?:\.([0-9]+))?\s*(B|bytes?|KiB|MiB|GiB|KB|MB|GB)?$') {
            $number = ($Matches[1] -replace ',', '') + '.' + $Matches[2]
            $units = @{B=1; byte=1; bytes=1; KiB=1024; MiB=1048576; GiB=1073741824;
                       KB=1000; MB=1000000; GB=1000000000}
            $factor = 1
            if ($Matches[3]) { $factor = $units[$Matches[3]] }
            return [int64]([double]::Parse($number, [CultureInfo]::InvariantCulture) * $factor)
        }
    }
}

function Show-Case([string]$CardDir, [string]$Case) {
    $rdir = Join-Path $CardDir "$Case-16-rlt"
    $r = Read-JsonFile (Join-Path $rdir 'result.json')
    $c = Read-JsonFile (Join-Path (Join-Path $CardDir "$Case-16-compact") 'result.json')
    foreach ($rec in @($r, $c)) {
        if ($rec.case -ne $Case -or $rec.page -ne 16 -or $rec.dtype -ne 'BF16' -or
            $null -eq $rec.extra_peak_allocated_bytes -or !$rec.one_layer_k_bytes -or
            !$rec.one_full_k_pool_bytes -or !$rec.actual_seq_lengths -or !$rec.head_dim -or !$rec.heads) {
            throw "Incomplete or mismatched result.json ($Case)"
        }
    }
    if ($r.layout -ne 'rlt' -or $c.layout -ne 'compact') { throw 'Layout mismatch' }
    $matched = [bool]($r.logical_sha256 -and $r.logical_sha256 -eq $c.logical_sha256)
    $outBytes = [int64]$r.actual_seq_lengths[-1] * $r.heads * $r.head_dim * 2
    $peak = [int64]$r.extra_peak_allocated_bytes
    $layer = [int64]$r.one_layer_k_bytes
    $pool = [int64]$r.one_full_k_pool_bytes
    Write-Output ("  result=$($r.status)/$($c.status) input_match=$matched " +
                  "K_alias=$($r.k_alias) V_alias=$($r.v_alias) replay_match=$($r.profile.matches_unprofiled_output)")
    Write-Output ("  bytes: peak_rlt=$peak peak_compact=$($c.extra_peak_allocated_bytes) " +
                  "difference=$($peak - $c.extra_peak_allocated_bytes) output=$outBytes " +
                  "peak_minus_output=$($peak - $outBytes) one_layer_K=$layer full_K=$pool")

    $trace = Read-JsonFile (Join-Path $rdir 'trace.json')
    if ($trace -is [array]) { $events = $trace } else { $events = $trace.traceEvents }
    if (!$events) { throw 'No Chrome trace events' }
    $devPids = @{}
    foreach ($e in $events) {
        if ($e.ph -ceq 'M' -and $e.name -eq 'process_name' -and
            ([string]$e.args.name -match 'Ascend Hardware|NPU Hardware|^Device\s*\d+')) {
            $devPids[[string]$e.pid] = $true
        }
    }
    $candidates = [Collections.Generic.List[object]]::new()
    $views = 0; $deviceEvents = 0; $fiaDevice = 0; $fiaHost = 0
    $deviceNames = @{}
    foreach ($e in $events) {
        if ($e.ph -cnotin @('X', 'B')) { continue }
        $name = [string]$e.name
        $cat = [string]$e.cat
        $dev = $devPids.ContainsKey([string]$e.pid) -or
               $cat -match '^(kernel|memcpy|memset|Ascend Hardware|AI_CORE|AI_VECTOR_CORE)$'
        if ($dev) {
            $deviceEvents++
            $deviceNames[$name]++
        }
        if (($name -replace '[_\s-]', '') -match 'fusedinferattention|flashattention|fainfer') {
            if ($dev) { $fiaDevice++ } else { $fiaHost++ }
        }
        if ($name -match '^aten::(_unsafe_view|view|view_as|as_strided)(\.|$)') {
            $views++
            continue
        }
        $taskType = [string]$e.args.'Task Type'
        $opType = [string]$e.args.'OP Type'
        if ("$name $cat $taskType $opType" -notmatch 'memcpy|trans[_-]?data|tensor[_-]?copy|contiguous|clone|aten::(_to_copy|copy_|cat|stack|reshape)\b|strided.?copy|inplace.?copy|format.?cast|identitycopy|view_copy|as_strided_copy|\bcopy\b') { continue }
        $bytes = Get-TransferBytes $e.args
        if ($null -eq $bytes) {
            $bytes = Get-TransferBytes $e.args.details
        }
        $dims = $e.args.'Input Dims'
        if ($null -eq $dims) { $dims = $e.args.'Input Shapes' }
        $shape = ''
        if ($null -ne $dims) {
            $shape = ConvertTo-Json -InputObject $dims -Depth 5 -Compress
            if ($shape.Length -gt 100) { $shape = $shape.Substring(0, 100) + '...' }
        }
        $candidates.Add([pscustomobject]@{Name=$name; Device=$dev; Bytes=$bytes; Shape=$shape})
    }

    $deviceCopies = @($candidates | Where-Object { $_.Device })
    $sized = @($deviceCopies | Where-Object { $null -ne $_.Bytes })
    $unknown = $candidates.Count - @($candidates | Where-Object { $null -ne $_.Bytes }).Count
    $unknownDevice = $deviceCopies.Count - $sized.Count
    $max = [int64](($sized | Measure-Object Bytes -Maximum).Maximum)
    $sum = [int64](($sized | Measure-Object Bytes -Sum).Sum)
    $largestAny = [int64](($candidates | Measure-Object Bytes -Maximum).Maximum)
    Write-Output ("  trace: total=$($events.Count) device=$deviceEvents fia_host=$fiaHost fia_device=$fiaDevice " +
                  "views=$views copy_candidates=$($candidates.Count) device_candidates=$($deviceCopies.Count) " +
                  "unknown_bytes=$unknown device_unknown=$unknownDevice device_max_B=$max device_sum_B=$sum")
    $topOps = @($deviceNames.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 4 |
                ForEach-Object { "$($_.Key) x$($_.Value)" })
    if ($topOps.Count) { Write-Output ("  device_ops: " + ($topOps -join '; ')) }
    # Show the largest sized group and unknown-size groups even if small events dominate.
    $groups = @($candidates | Group-Object Name,Device | ForEach-Object {
        $known = @($_.Group | Where-Object { $null -ne $_.Bytes })
        [pscustomobject]@{Name=$_.Group[0].Name; Dev=$_.Group[0].Device; N=$_.Count;
            Shape=$_.Group[0].Shape; Unknown=$_.Count - $known.Count; Max=[int64](($known | Measure-Object Bytes -Maximum).Maximum)}
    })
    foreach ($g in @($groups | Sort-Object Max,Unknown,N -Descending | Select-Object -First 6)) {
        Write-Output "  candidate: $($g.Name) device=$($g.Dev) count=$($g.N) max_B=$($g.Max) unknown=$($g.Unknown) shape=$($g.Shape)"
    }

    if ($r.k_alias -eq $false -or $r.v_alias -eq $false) { $finding = 'OP_INPUT_NOT_ALIASED' }
    elseif ($r.status -ne 'PASS' -or $c.status -ne 'PASS' -or !$matched -or
            $r.k_alias -ne $true -or $r.v_alias -ne $true -or
            $r.profile.matches_unprofiled_output -ne $true -or !$deviceEvents -or !($fiaHost + $fiaDevice)) {
        $finding = 'INSUFFICIENT_EVIDENCE'
    } elseif ($max -ge $pool) { $finding = 'FULL_POOL_SIZED_DEVICE_CANDIDATE' }
    elseif ($max -ge $layer) { $finding = 'LAYER_SIZED_DEVICE_CANDIDATE' }
    elseif ($largestAny -ge $layer -or $sum -ge $layer -or $unknown -gt 0) {
        $finding = 'REVIEW_COPY_CANDIDATES'
    } elseif (!$fiaDevice) { $finding = 'INSUFFICIENT_DEVICE_OP_DETAIL' }
    elseif (($peak - $c.extra_peak_allocated_bytes) -ge $layer) { $finding = 'REVIEW_LAYOUT_PEAK_GROWTH' }
    else { $finding = 'NO_LAYER_OR_POOL_SIZED_COPY_VISIBLE' }
    Write-Output "  finding=$finding"
}

if (!(Test-Path -LiteralPath $Root -PathType Container)) { throw "Directory missing: $Root" }
$cardDirs = @(Get-ChildItem -LiteralPath $Root -Recurse -Directory -Filter 'card-0')
Write-Output '=== BEGIN KV COPY AUDIT ==='
foreach ($machine in @('910B3','910B4')) {
    $dirs = @($cardDirs | Where-Object { ($_.FullName -split '[\\/]') -match "^(results-)?$machine$" })
    if ($dirs.Count -ne 1) {
        Write-Output "$machine finding=INSUFFICIENT_EVIDENCE (found $($dirs.Count) matching card-0 directories)"
        continue
    }
    foreach ($case in @('w2','c4_h1')) {
        Write-Output "$machine/card-0/$case page=16"
        try { Show-Case $dirs[0].FullName $case }
        catch { Write-Output "  finding=INSUFFICIENT_EVIDENCE ($($_.Exception.Message))" }
    }
}
Write-Output '=== END KV COPY AUDIT ==='
