# Gera dist\SceneFinder\ (PyInstaller) e dist\SceneFinder-Setup-<versao>.exe (Inno).
# Usa venv-build (sem torch) - com torch o pacote passaria de 800 MB.
$ErrorActionPreference = 'Stop'
$app = $PSScriptRoot
Set-Location $app

$py = Join-Path $app 'venv-build\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host ">>> criando venv-build"
    $base = @('C:\Python314\python.exe', (Get-Command python -EA SilentlyContinue).Source) |
            Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $base) { throw "nenhum python encontrado para criar o venv-build" }
    & $base -m venv venv-build
    # onnxruntime-directml, NAO onnxruntime: e o que da acesso a GPU (DirectML
    # funciona com AMD/NVIDIA/Intel). Os dois pacotes ocupam o mesmo diretorio
    # em site-packages e se sobrescrevem em silencio - quem instala por ultimo
    # vence -, por isso aqui so pode existir um.
    & (Join-Path $app 'venv-build\Scripts\pip.exe') install --quiet `
        onnxruntime-directml numpy pillow tokenizers pywebview pyinstaller `
        opencv-python-headless mutagen soundfile soxr
}

if (-not (Test-Path (Join-Path $app 'models\clap_audio.onnx'))) {
    Write-Warning "models\clap_audio.onnx ausente: o app sai SEM busca de audio por conteudo. Rode: venv\Scripts\python tools\export_clap.py"
}

# guarda: se alguem reinstalar onnxruntime por cima, o app volta para CPU
# silenciosamente e ninguem percebe ate cronometrar uma indexacao
& $py -c "import onnxruntime as o; ps=o.get_available_providers(); print('providers:', ps); assert 'DmlExecutionProvider' in ps, 'DirectML sumiu do venv-build: rode pip uninstall onnxruntime'"
if ($LASTEXITCODE -ne 0) { throw "venv-build sem DirectML" }

if (-not (Test-Path (Join-Path $app 'models\image.onnx'))) {
    throw "models\ vazio. Rode: venv\Scripts\python tools\export_onnx.py"
}
if (-not (Test-Path (Join-Path $app 'icon.ico'))) { & $py make_icon.py }

$versao = (Select-String -Path (Join-Path $app 'version.py') -Pattern '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Write-Host ">>> versao $versao"

Write-Host ">>> PyInstaller"
Remove-Item -Recurse -Force (Join-Path $app 'build'), (Join-Path $app 'dist\SceneFinder') -EA SilentlyContinue
& $py -m PyInstaller --noconfirm --clean scenefinder.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falhou" }

$mb = [math]::Round((Get-ChildItem (Join-Path $app 'dist\SceneFinder') -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
Write-Host ">>> dist\SceneFinder = $mb MB"

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
    Write-Warning "Inno Setup nao encontrado - so o dist\SceneFinder foi gerado."
    exit 0
}

Write-Host ">>> Inno Setup"
& $iscc "/DAppVersion=$versao" (Join-Path $app 'installer.iss') | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { throw "Inno Setup falhou" }

$setup = Join-Path $app "dist\SceneFinder-Setup-$versao.exe"
$smb = [math]::Round((Get-Item $setup).Length / 1MB)
Write-Host ">>> PRONTO: $setup ($smb MB)"
