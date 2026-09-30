# Run this script from the VS Code PowerShell terminal to watch local GPU training live.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$resultRoot = Join-Path $projectRoot 'local_results'
$experimentName = 'mobilenetv2_160_rtx2050'
$env:YOLO_CONFIG_DIR = $projectRoot
$checkpointDir = Join-Path $env:USERPROFILE '.cache\torch\hub\checkpoints'

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $venvPython"
}
if (Test-Path -LiteralPath (Join-Path $resultRoot $experimentName)) {
    throw "Refusing to overwrite existing results: $(Join-Path $resultRoot $experimentName)"
}

# TorchVision looks for this official MobileNetV2 checkpoint in its cache.
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null
Copy-Item (Join-Path $projectRoot 'models\mobilenet_v2-7ebf99e0.pth') `
    (Join-Path $checkpointDir 'mobilenet_v2-7ebf99e0.pth') -Force

& $venvPython 'cloud_train\train_mobilenetv2_seg.py' `
    --epochs 100 `
    --batch 8 `
    --workers 2 `
    --device 0 `
    --project 'local_results' `
    --name $experimentName
