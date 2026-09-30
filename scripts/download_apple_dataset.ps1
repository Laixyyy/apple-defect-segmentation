param(
    [string]$ManifestPath = 'D:\something\edge download\apple-qualityv4icoco-segmentation.ndjson',
    [string]$OutputRoot = 'D:\BIT\fruits\data\raw_apple_quality'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Dataset manifest was not found: $ManifestPath"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$manifestCopy = Join-Path $OutputRoot 'apple-qualityv4icoco-segmentation.ndjson'
Copy-Item -LiteralPath $ManifestPath -Destination $manifestCopy -Force

$items = Get-Content -LiteralPath $ManifestPath | ForEach-Object { $_ | ConvertFrom-Json } |
    Where-Object { $_.type -eq 'image' }

$results = $items | ForEach-Object -Parallel {
    $item = $_
    $destinationDir = Join-Path $using:OutputRoot (Join-Path $item.split 'images')
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    $destination = Join-Path $destinationDir $item.file

    if ((Test-Path -LiteralPath $destination -PathType Leaf) -and (Get-Item -LiteralPath $destination).Length -gt 0) {
        return 'skipped'
    }

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        Remove-Item -LiteralPath $destination -Force
    }

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $item.url -OutFile $destination
            if ((Get-Item -LiteralPath $destination).Length -gt 0) {
                return 'downloaded'
            }
        }
        catch {
            if ($attempt -eq 3) { return "failed:$($item.file)" }
            Start-Sleep -Seconds $attempt
        }
    }
} -ThrottleLimit 8

$downloaded = @($results | Where-Object { $_ -eq 'downloaded' }).Count
$skipped = @($results | Where-Object { $_ -eq 'skipped' }).Count
$failed = @($results | Where-Object { $_ -like 'failed:*' })
Write-Output "Complete. Downloaded: $downloaded; already present: $skipped; failed: $($failed.Count); total: $($items.Count)."
if ($failed.Count -gt 0) { $failed }
