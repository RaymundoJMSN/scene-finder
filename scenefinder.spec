# PyInstaller spec - onedir (onefile extrairia os 225 MB de modelo a cada abertura)
from PyInstaller.utils.hooks import collect_all

datas = [
    ("index.html", "."),
    ("icon.ico", "."),
    ("models", "models"),
]
hiddenimports = [
    "webview.platforms.winforms", "clr", "clr_loader",
    "encoder", "indexer", "server", "ptbr", "updater", "version",
]
binaries = []
for mod in ("onnxruntime", "tokenizers"):
    d, b, h = collect_all(mod)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # o app roda em ONNX; nada de torch entra no pacote
    excludes=["torch", "transformers", "sentence_transformers", "scipy",
              "sklearn", "matplotlib", "pandas", "tkinter", "onnx",
              "onnxscript", "huggingface_hub", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="SceneFinder",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # sem janela de terminal
    icon="icon.ico",
    version_info=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, name="SceneFinder",
)
