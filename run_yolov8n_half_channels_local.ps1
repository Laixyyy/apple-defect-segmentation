# Run in a PowerShell terminal to watch the half-channel experiment live.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$resultRoot = Join-Path $projectRoot 'local_results'
$experimentName = 'yolov8n_half_channels_160_rtx2050'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'Ultralytics'

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $venvPython"
}
if (Test-Path -LiteralPath (Join-Path $resultRoot $experimentName)) {
    throw "Refusing to overwrite existing results: $(Join-Path $resultRoot $experimentName)"
}
if (Test-Path -LiteralPath (Join-Path $resultRoot "${experimentName}_test")) {
    throw "Refusing to overwrite existing test results: $(Join-Path $resultRoot "${experimentName}_test")"
}

& $venvPython 'cloud_train\train_yolov8n_half_channels.py' `
    --weights 'models\yolov8n-seg.pt' `
    --epochs 100 `
    --batch 16 `
    --workers 2 `
    --device 0 `
    --project 'local_results' `
    --name $experimentName
