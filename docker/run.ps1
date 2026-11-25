# run.ps1 —— 在 ncnn 构建容器里跑一条命令
#
# 例子：
#   .\docker\run.ps1 -Cmd "cmake --version && python3 -c 'import pybind11;print(pybind11.get_include())'"
#   .\docker\run.ps1 -Torch -Cmd "cd /work/ncnn && ./build_pnnx.sh"
#
# 约定：
#   D:\ps\ncnn        -> /work/ncnn      （源码，读写；构建产物落在 build-* 目录里）
#   D:\ps\build       -> /work/build     （构建目录，避免污染源码）
#   桌面\量化回测      -> /work/quant     （脚本与测试文件，读写）

param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [switch]$Torch,
    [string]$Mem = "8g",
    [string]$Cpu = "6"
)

$ErrorActionPreference = "Stop"

$src  = "D:\ps\ncnn"
$out  = "D:\ps\build"
$quant = "C:\Users\<user>\Desktop\量化回测"

foreach ($p in @($src, $quant)) {
    if (-not (Test-Path $p)) { throw "缺少目录：$p" }
}
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

$image = if ($Torch) { "ncnn-build:24.04-torch" } else { "ncnn-build:24.04" }

$argv = @(
    "run", "--rm",
    "--memory", $Mem,
    "--cpus", $Cpu,
    "-v", "${src}:/work/ncnn",
    "-v", "${out}:/work/build",
    "-v", "${quant}:/work/quant",
    "-w", "/work",
    $image,
    "bash", "-lc", $Cmd
)

Write-Host "== docker $($argv -join ' ')" -ForegroundColor DarkGray
& docker @argv
exit $LASTEXITCODE
