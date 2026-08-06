"""Auto-update: compara a versao local com a ultima release do GitHub.

Nao instala nada sozinho - baixa e pergunta. Quem decide e o usuario, e o
instalador NSIS cuida de fechar/substituir o app.
"""
import json
import os
import re
import ssl
import tempfile
import threading
import urllib.request
from pathlib import Path

from version import REPO, __version__

API = f"https://api.github.com/repos/{REPO}/releases/latest"
UA = {"User-Agent": f"scene-finder/{__version__}"}
STATE = {"checked": False, "available": None, "downloading": False,
         "progress": 0, "file": None, "error": None}


def _num(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3] or [0])


def check():
    """-> dict da release se houver versao mais nova, senao None."""
    try:
        req = urllib.request.Request(API, headers=UA)
        with urllib.request.urlopen(req, timeout=8) as r:
            rel = json.loads(r.read())
        tag = rel.get("tag_name", "")
        if _num(tag) > _num(__version__):
            asset = next((a for a in rel.get("assets", [])
                          if a["name"].lower().endswith(".exe")), None)
            if asset:
                return {"version": tag, "url": asset["browser_download_url"],
                        "size": asset.get("size", 0),
                        "notes": (rel.get("body") or "")[:2000]}
    except Exception as e:
        STATE["error"] = str(e)
    return None


def check_async():
    def run():
        STATE["available"] = check()
        STATE["checked"] = True
    threading.Thread(target=run, daemon=True).start()


def download(url, on_progress=None):
    """Baixa o instalador para a pasta temporaria e devolve o caminho."""
    dest = Path(tempfile.gettempdir()) / "SceneFinder-update.exe"
    req = urllib.request.Request(url, headers=UA)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r, \
            open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while chunk := r.read(262144):
            f.write(chunk)
            got += len(chunk)
            if on_progress and total:
                on_progress(int(got * 100 / total))
    return dest


def download_async():
    if STATE["downloading"] or not STATE["available"]:
        return
    STATE["downloading"] = True
    STATE["progress"] = 0

    def run():
        try:
            def prog(p):
                STATE["progress"] = p
            STATE["file"] = str(download(STATE["available"]["url"], prog))
        except Exception as e:
            STATE["error"] = str(e)
        finally:
            STATE["downloading"] = False
    threading.Thread(target=run, daemon=True).start()


def install():
    """Roda o instalador baixado e encerra o app (o NSIS substitui os arquivos)."""
    if not STATE["file"] or not Path(STATE["file"]).exists():
        return False
    os.startfile(STATE["file"])
    return True
