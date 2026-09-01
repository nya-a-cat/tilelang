[CmdletBinding()]
param(
    [switch]$Execute,
    [decimal]$MaximumHourlyUsd = 0.12,
    [int]$MaximumRuntimeMinutes = 30,
    [int]$PollSeconds = 20,
    [string]$EvidenceDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$VastCli = "C:\Users\element\.local\bin\vastai.exe"
$ExpectedGpuName = "RTX 5060 Ti"
$ExpectedComputeCapability = 1200
$DiskGb = 20
$Image = "vastai/pytorch@sha256:6ee5f68a3c11bd89e9364771bf6b929d5f266c4382fb3628d751b5e89241d462"
$ImageDigest = "sha256:6ee5f68a3c11bd89e9364771bf6b929d5f266c4382fb3628d751b5e89241d462"
$RunnerSourceSha = "4369b41eaac751fdbd5ba931de8c9e544cad6a3f"
$RunnerSha256 = "9a9a114cf515146a3950ea716ec428341c031a889ba5ab3aec1ef350f6dae3fd"
$RunnerUrl = "https://raw.githubusercontent.com/nya-a-cat/tilelang/$RunnerSourceSha/benchmark/research/run_layout_divergent_blackwell.py"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent (Split-Path -Parent $ScriptDirectory)
$WorkspaceRoot = Split-Path -Parent $RepositoryRoot
$OnstartPath = Join-Path $ScriptDirectory "vastai_layout_blackwell_onstart.sh"
$RunStamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$Label = "tilelang-layout-blackwell-$RunStamp"
$OfferQuery = "gpu_name=RTX_5060_Ti num_gpus=1 compute_cap=1200 reliability>=0.995 gpu_ram>=15000 cuda_vers>=12.8 driver_version>=580.00.00 disk_space>=30 disk_bw>=1000 inet_down>=250 inet_up>=100 inet_down_cost<=0.002 inet_up_cost<=0.002 direct_port_count>=2 cpu_arch=amd64"

if (-not (Test-Path -LiteralPath $VastCli -PathType Leaf)) {
    throw "Vast CLI is missing at $VastCli"
}
if (-not (Test-Path -LiteralPath $OnstartPath -PathType Leaf)) {
    throw "Onstart script is missing at $OnstartPath"
}
if ($MaximumRuntimeMinutes -lt 5 -or $MaximumRuntimeMinutes -gt 60) {
    throw "MaximumRuntimeMinutes must be between 5 and 60"
}
if ($PollSeconds -lt 10 -or $PollSeconds -gt 60) {
    throw "PollSeconds must be between 10 and 60"
}

if ([string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
    $EvidenceDirectory = Join-Path $WorkspaceRoot ".codex_tmp\vast-blackwell-$RunStamp"
}
$ResolvedEvidenceDirectory = [System.IO.Path]::GetFullPath($EvidenceDirectory)
[System.IO.Directory]::CreateDirectory($ResolvedEvidenceDirectory) | Out-Null
$RemoteEvidenceDirectory = Join-Path $ResolvedEvidenceDirectory "remote"
[System.IO.Directory]::CreateDirectory($RemoteEvidenceDirectory) | Out-Null

function Invoke-VastText {
    param([Parameter(Mandatory = $true)][string[]]$VastArguments)
    $output = & $VastCli @VastArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Vast CLI failed ($LASTEXITCODE): $($VastArguments -join ' ')"
    }
    return ($output -join "`n")
}

function Invoke-VastJson {
    param([Parameter(Mandatory = $true)][string[]]$VastArguments)
    $text = Invoke-VastText -VastArguments $VastArguments
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "Vast CLI returned empty JSON: $($VastArguments -join ' ')"
    }
    $parsed = $text | ConvertFrom-Json
    if ($null -eq $parsed) {
        return $null
    }
    $errorProperty = $parsed.PSObject.Properties["error"]
    if ($null -ne $errorProperty -and $errorProperty.Value -eq $true) {
        throw "Vast API error: $text"
    }
    return $parsed
}

function Get-ObjectProperty {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Name,
        $Default = $null
    )
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $Default
    }
    return $property.Value
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Get-MatchingInstances {
    $shown = Invoke-VastJson -VastArguments @("show", "instances", "--label", $Label, "--raw")
    if ($null -eq $shown) {
        return
    }
    return @($shown)
}

function Get-AllInstances {
    $shown = Invoke-VastJson -VastArguments @("show", "instances", "--raw")
    if ($null -eq $shown) {
        return
    }
    return @($shown)
}

function Save-ContainerLogs {
    param(
        [Parameter(Mandatory = $true)][long]$InstanceId,
        [Parameter(Mandatory = $true)][string]$Path
    )
    try {
        $logs = Invoke-VastText -VastArguments @("logs", [string]$InstanceId, "--tail", "1000")
        $logs | Set-Content -LiteralPath $Path -Encoding utf8
        return $logs
    }
    catch {
        ("log fetch failed: " + $_.Exception.Message) | Set-Content -LiteralPath $Path -Encoding utf8
        return ""
    }
}

function Test-RemoteManifest {
    param([Parameter(Mandatory = $true)][string]$Directory)
    $manifestPath = Join-Path $Directory "SHA256SUMS"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Remote evidence is missing SHA256SUMS"
    }
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
            throw "Malformed SHA256SUMS line: $line"
        }
        $expected = $Matches[1]
        $name = $Matches[2]
        $path = Join-Path $Directory $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Remote evidence is missing $name"
        }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "SHA-256 mismatch for ${name}: expected $expected, got $actual"
        }
    }
}

$existing = @(Get-MatchingInstances)
if ($existing.Count -ne 0) {
    throw "An instance already exists with label $Label"
}

$offersRaw = Invoke-VastJson -VastArguments @(
    "search", "offers", $OfferQuery,
    "--storage", [string]$DiskGb,
    "-o", "dph",
    "--limit", "20",
    "--raw"
)
if ($null -eq $offersRaw) {
    throw "Vast returned no RTX 5060 Ti offers"
}
$offers = @(@($offersRaw) | Where-Object {
    $_.rentable -eq $true -and
    $_.gpu_name -eq $ExpectedGpuName -and
    [int]$_.compute_cap -eq $ExpectedComputeCapability -and
    [decimal]$_.dph_total -le $MaximumHourlyUsd
} | Sort-Object -Property @{ Expression = { [decimal]$_.dph_total }; Ascending = $true }, @{ Expression = { [decimal]$_.reliability }; Descending = $true })

if ($offers.Count -eq 0) {
    throw "No eligible RTX 5060 Ti offer is available under `$$MaximumHourlyUsd/hour"
}
$offer = $offers[0]
$onstartSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OnstartPath).Hash.ToLowerInvariant()

$runnerProbe = Join-Path $ResolvedEvidenceDirectory "runner-download-probe.py"
Invoke-WebRequest -Uri $RunnerUrl -OutFile $runnerProbe -UseBasicParsing
$runnerProbeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runnerProbe).Hash.ToLowerInvariant()
if ($runnerProbeHash -ne $RunnerSha256) {
    throw "Published runner SHA-256 mismatch: expected $RunnerSha256, got $runnerProbeHash"
}

$preflight = [ordered]@{
    schema = "tilelang-vast-blackwell-controller-preflight-v1"
    execute = [bool]$Execute
    selected_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    label = $Label
    offer_query = $OfferQuery
    maximum_hourly_usd = $MaximumHourlyUsd
    maximum_runtime_minutes = $MaximumRuntimeMinutes
    disk_gb = $DiskGb
    image = $Image
    image_digest = $ImageDigest
    runner_source_sha = $RunnerSourceSha
    runner_sha256 = $RunnerSha256
    runner_url = $RunnerUrl
    onstart_path = $OnstartPath
    onstart_sha256 = $onstartSha256
    selected_offer = $offer
}
Write-JsonFile -Value $preflight -Path (Join-Path $ResolvedEvidenceDirectory "controller-preflight.json")

Write-Host "Selected offer $($offer.id): $($offer.gpu_name), reliability=$($offer.reliability), total=`$$($offer.dph_total)/hour"
Write-Host "Evidence directory: $ResolvedEvidenceDirectory"
if (-not $Execute) {
    Write-Host "Dry run complete. No Vast instance was created."
    return
}

$InstanceId = $null
$InstanceCreatedAt = $null
$CreateAttemptAt = $null
$Failure = $null
$DestroyVerified = $false
$CopyVerified = $false
$pollHistory = [System.Collections.Generic.List[object]]::new()
$controllerLogPath = Join-Path $ResolvedEvidenceDirectory "vast-container.log"

try {
    $environment = "-e TILELANG_VAST_OFFER_ID=$($offer.id)"
    $CreateAttemptAt = [DateTimeOffset]::UtcNow
    $create = Invoke-VastJson -VastArguments @(
        "create", "instance", [string]$offer.id,
        "--image", $Image,
        "--disk", [string]$DiskGb,
        "--ssh", "--direct",
        "--onstart", $OnstartPath,
        "--env", $environment,
        "--label", $Label,
        "--cancel-unavail",
        "--raw"
    )
    if ($null -eq $create) {
        throw "Vast create returned an empty response"
    }
    $createSuccess = Get-ObjectProperty -Value $create -Name "success" -Default $false
    $newContract = Get-ObjectProperty -Value $create -Name "new_contract"
    if ($createSuccess -ne $true -or $null -eq $newContract) {
        throw "Vast create did not return an instance ID: $($create | ConvertTo-Json -Depth 10)"
    }
    $InstanceId = [long]$newContract
    $InstanceCreatedAt = $CreateAttemptAt
    Write-Host "Created Vast instance $InstanceId"

    $deadline = $InstanceCreatedAt.AddMinutes($MaximumRuntimeMinutes)
    $completionSeen = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $allInstances = @(Get-AllInstances)
        $instance = $allInstances | Where-Object { [long]$_.id -eq $InstanceId } | Select-Object -First 1
        if ($null -eq $instance) {
            throw "Instance $InstanceId disappeared before evidence collection"
        }
        $pollHistory.Add([ordered]@{
            unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
            status = Get-ObjectProperty -Value $instance -Name "actual_status"
            cur_state = Get-ObjectProperty -Value $instance -Name "cur_state"
            status_msg = Get-ObjectProperty -Value $instance -Name "status_msg"
            dph_total = Get-ObjectProperty -Value $instance -Name "dph_total"
        })
        Write-JsonFile -Value @($pollHistory) -Path (Join-Path $ResolvedEvidenceDirectory "controller-polls.json")
        $logs = Save-ContainerLogs -InstanceId $InstanceId -Path $controllerLogPath
        if ($logs -match 'TILELANG_VAST_RUN_DONE status=(complete|failed) exit_code=([0-9]+)') {
            $completionSeen = $true
            Write-Host "Remote completion marker: status=$($Matches[1]) exit_code=$($Matches[2])"
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $completionSeen) {
        throw "Vast experiment exceeded the $MaximumRuntimeMinutes minute watchdog"
    }

    $remoteTarget = "local:" + ($RemoteEvidenceDirectory -replace '\\', '/')
    $copyError = $null
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        try {
            Invoke-VastText -VastArguments @("copy", "C.$InstanceId`:/workspace/evidence", $remoteTarget) | Out-Null
            $copyError = $null
            break
        }
        catch {
            $copyError = $_
            Start-Sleep -Seconds 10
        }
    }
    if ($null -ne $copyError) {
        throw $copyError
    }

    $copiedRoot = $RemoteEvidenceDirectory
    $nested = Join-Path $RemoteEvidenceDirectory "evidence"
    if (Test-Path -LiteralPath $nested -PathType Container) {
        $copiedRoot = $nested
    }
    Test-RemoteManifest -Directory $copiedRoot
    $lifecyclePath = Join-Path $copiedRoot "vast-lifecycle.json"
    $resultPath = Join-Path $copiedRoot "layout-divergent-blackwell-run.json"
    if (-not (Test-Path -LiteralPath $lifecyclePath -PathType Leaf)) {
        throw "Remote evidence is missing vast-lifecycle.json"
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Remote evidence is missing layout-divergent-blackwell-run.json"
    }
    $lifecycle = Get-Content -Raw -LiteralPath $lifecyclePath | ConvertFrom-Json
    $result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json
    if ($lifecycle.status -ne "complete" -or $result.status -ne "complete") {
        throw "Remote run was not complete: lifecycle=$($lifecycle.status), result=$($result.status)"
    }
    $CopyVerified = $true
}
catch {
    $Failure = $_
    Write-Warning $_.Exception.Message
}
finally {
    if ($null -eq $InstanceId) {
        try {
            $reconciled = @(Get-MatchingInstances)
            if ($reconciled.Count -gt 1) {
                throw "Multiple instances unexpectedly share label $Label"
            }
            if ($reconciled.Count -eq 1) {
                $InstanceId = [long]$reconciled[0].id
                $InstanceCreatedAt = if ($null -ne $CreateAttemptAt) { $CreateAttemptAt } else { [DateTimeOffset]::UtcNow }
                Write-Warning "Recovered instance ID $InstanceId from its unique label"
            }
            else {
                $DestroyVerified = $true
            }
        }
        catch {
            Write-Warning "Instance reconciliation failed: $($_.Exception.Message)"
        }
    }
    if ($null -ne $InstanceId) {
        Save-ContainerLogs -InstanceId $InstanceId -Path $controllerLogPath | Out-Null
        for ($attempt = 1; $attempt -le 4; $attempt++) {
            try {
                $allInstances = @(Get-AllInstances)
                $stillPresent = $allInstances | Where-Object { [long]$_.id -eq $InstanceId } | Select-Object -First 1
                if ($null -eq $stillPresent) {
                    $DestroyVerified = $true
                    break
                }
            }
            catch {
                Write-Warning "Instance status check $attempt failed: $($_.Exception.Message)"
            }
            try {
                Invoke-VastText -VastArguments @("destroy", "instance", [string]$InstanceId, "-y", "--raw") | Out-Null
            }
            catch {
                Write-Warning "Destroy attempt $attempt failed: $($_.Exception.Message)"
            }
            Start-Sleep -Seconds 5
        }
        if (-not $DestroyVerified) {
            try {
                $allInstances = @(Get-AllInstances)
                $DestroyVerified = $null -eq ($allInstances | Where-Object { [long]$_.id -eq $InstanceId } | Select-Object -First 1)
            }
            catch {
                Write-Warning "Final destruction verification failed: $($_.Exception.Message)"
            }
        }
    }

    $finishedAt = [DateTimeOffset]::UtcNow
    $elapsedHours = if ($null -ne $InstanceCreatedAt) { ($finishedAt - $InstanceCreatedAt).TotalHours } else { 0.0 }
    $summary = [ordered]@{
        schema = "tilelang-vast-blackwell-controller-summary-v1"
        label = $Label
        offer_id = $offer.id
        instance_id = $InstanceId
        instance_created_unix = if ($null -ne $InstanceCreatedAt) { $InstanceCreatedAt.ToUnixTimeSeconds() } else { $null }
        finished_unix = $finishedAt.ToUnixTimeSeconds()
        elapsed_hours = $elapsedHours
        hourly_usd = [decimal]$offer.dph_total
        estimated_compute_storage_cost_usd = $elapsedHours * [double]$offer.dph_total
        internet_down_cost_per_tb = Get-ObjectProperty -Value $offer -Name "internet_down_cost_per_tb"
        internet_up_cost_per_tb = Get-ObjectProperty -Value $offer -Name "internet_up_cost_per_tb"
        copy_verified = $CopyVerified
        destroy_verified = $DestroyVerified
        failure = if ($null -ne $Failure) { $Failure.Exception.Message } else { $null }
    }
    Write-JsonFile -Value $summary -Path (Join-Path $ResolvedEvidenceDirectory "controller-summary.json")
}

if (-not $DestroyVerified) {
    throw "CRITICAL: destruction of Vast instance $InstanceId could not be verified"
}
if ($null -ne $Failure) {
    throw $Failure
}
if (-not $CopyVerified) {
    throw "Remote evidence was not verified"
}

Write-Host "Vast run complete; evidence verified and instance $InstanceId destroyed."
