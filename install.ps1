# Scene Finder - ambiente de DESENVOLVIMENTO (venv com torch, so para exportar
# os modelos ONNX em tools/export_onnx.py). Idempotente.
# Quem so quer usar o app baixa o instalador em Releases; quem quer empacotar
# roda build.ps1, que usa o venv-build enxuto.
$ErrorActionPreference = 'Stop'
$app = $PSScriptRoot
$venv = Join-Path $app 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
$pip = Join-Path $venv 'Scripts\pip.exe'

function Test-Deps {
    if (-not (Test-Path $python)) { return $false }
    & $python -c "import torch, sentence_transformers, webview, PIL, numpy" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Resolve-BasePython {
    # preferencia: 3.12 (wheels torch garantidos) > 3.13 > default
    foreach ($v in @('-3.12', '-3.13')) {
        try {
            $exe = & py $v -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }
    return (& py -c "import sys;print(sys.executable)").Trim()
}

function Install-Into-Venv([string]$base) {
    Write-Host ">>> venv com $base"
    if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
    & $base -m venv $venv
    & $python -m pip install --upgrade pip --quiet
    & $pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { return $false }
    & $pip install sentence-transformers pillow pywebview numpy
    return ($LASTEXITCODE -eq 0)
}

if (Test-Deps) {
    Write-Host ">>> dependencias OK, pulando instalacao"
} else {
    $base = Resolve-BasePython
    $ok = Install-Into-Venv $base
    if (-not $ok) {
        # ponytail: fallback unico - torch sem wheel pro python atual -> instala 3.12 user-scope
        Write-Host ">>> torch falhou em $base; instalando Python 3.12 via winget"
        winget install --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
        $base = & py -3.12 -c "import sys;print(sys.executable)"
        $ok = Install-Into-Venv $base.Trim()
        if (-not $ok) { throw 'instalacao falhou mesmo com Python 3.12' }
    }
    Write-Host ">>> baixando modelos CLIP (imagem + texto multilingual)"
    & $python -c "from sentence_transformers import SentenceTransformer as S; S('clip-ViT-B-32'); S('sentence-transformers/clip-ViT-B-32-multilingual-v1'); print('modelos OK')"
    if ($LASTEXITCODE -ne 0) { throw 'download dos modelos falhou' }
}

if (Test-Path (Join-Path $app 'make_icon.py')) {
    & $python (Join-Path $app 'make_icon.py')
}

# sem atalhos aqui de proposito: eles apontariam para o codigo-fonte e ficariam
# competindo com o app instalado pelo Inno Setup (mesmo nome, mesma porta).
Write-Host ">>> AMBIENTE DE DEV PRONTO"
Write-Host "    modelos:  venv\Scripts\python tools\export_onnx.py"
Write-Host "    rodar:    venv\Scripts\python app.py"
Write-Host "    empacotar: build.ps1"
