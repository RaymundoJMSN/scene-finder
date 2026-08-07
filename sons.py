"""Scene Finder - indice de AUDIO (musicas e efeitos sonoros).

Tres sinais somados, espelhando o que funcionou no indice visual:
- semantica do NOME do arquivo, embedada pelo mesmo encoder de texto
  multilingue do indice de imagens ("chuva" acha "Rain_Loop_02.ogg")
- match literal de tokens do caminho
- conteudo do SOM via CLAP (audio<->texto), quando models/clap_*.onnx existem:
  acha o arquivo pelo que ele toca, mesmo com nome inutil ("track_17.wav")

Sem miniaturas: audio nao tem thumb; a UI mostra nome + duracao + player.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import encoder
import indexer

APP = indexer.APP
AUDIO_EXTS = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus"}
CHECKPOINT = 4000

# pesos dos sinais (cossenos em [-1,1]); nome domina porque os acervos de RPG
# tem nomes descritivos; CLAP complementa onde o nome nao ajuda
NAME_W = 1.0
LIT_W = 0.35
CLAP_W = 0.6

log = indexer.log


def pastas_audio(cfg):
    return [p for p in (cfg.get("audio_folders") or []) if Path(p).is_dir()]


def scan_audio(cfg):
    out, vistos = [], set()
    for folder in pastas_audio(cfg):
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = [d for d in dirnames if d.lower() != "__macosx"]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in AUDIO_EXTS:
                    continue
                chave = os.path.normcase(str(p))
                if chave not in vistos:
                    vistos.add(chave)
                    out.append(str(p))
    return out


def _duracao(p):
    try:
        import mutagen
        mf = mutagen.File(p)
        if mf is not None and mf.info and mf.info.length:
            return round(float(mf.info.length), 1)
    except Exception:
        pass
    try:
        import soundfile as sf
        return round(float(sf.info(p).duration), 1)
    except Exception:
        return 0.0


def load_audio(out=APP):
    meta_p = Path(out) / "audio_meta.json"
    nomes_p = Path(out) / "audio_names.npz"
    if not (meta_p.exists() and nomes_p.exists()):
        return [], np.zeros((0, encoder.dim()), np.float32), None
    try:
        items = json.loads(meta_p.read_text("utf-8"))["items"]
        nemb = np.load(nomes_p)["nemb"]
    except Exception as e:
        log.info("indice de audio ilegivel (%s): descartando", e)
        return [], np.zeros((0, encoder.dim()), np.float32), None
    if len(nemb) != len(items) or nemb.shape[1] != encoder.dim():
        log.info("indice de audio desalinhado: descartando")
        return [], np.zeros((0, encoder.dim()), np.float32), None
    cemb = None
    clap_p = Path(out) / "audio_clap.npz"
    if clap_p.exists():
        try:
            cemb = np.load(clap_p)["cemb"]
            if len(cemb) != len(items):
                cemb = None
        except Exception:
            cemb = None
    return items, nemb, cemb


def _salvar_audio(out, items, nemb):
    buf = tempfile.NamedTemporaryFile(delete=False, dir=out, suffix=".npz")
    np.savez(buf, nemb=nemb)
    buf.close()
    os.replace(buf.name, Path(out) / "audio_names.npz")
    indexer._atomic_write(Path(out) / "audio_meta.json",
                          json.dumps({"items": items}).encode("utf-8"))


def build_audio(cfg, out=APP, progress=None, plano=None):
    """Indice incremental de nomes de audio. Rapido: so texto, sem decodificar som."""
    out = Path(out)
    velhos_items, velho_nemb, velho_cemb = load_audio(out)
    velho = {os.path.normcase(it["p"]): i for i, it in enumerate(velhos_items)}

    arquivos = scan_audio(cfg)
    kept_rows, kept_items, novos = [], [], []
    for p in arquivos:
        i = velho.get(os.path.normcase(p))
        if i is not None and abs(velhos_items[i]["m"] - os.path.getmtime(p)) < 1e-6:
            kept_rows.append(i)
            kept_items.append(velhos_items[i])
        else:
            novos.append(p)

    counts = {"total": len(arquivos), "kept": len(kept_rows),
              "added": 0, "removed": len(velho) - len(kept_rows)}
    if plano:
        plano(len(kept_rows), len(novos))
    if progress:
        progress(len(kept_rows), len(arquivos))

    d = encoder.dim()
    nemb_partes = [velho_nemb[kept_rows].reshape(-1, d)]
    items = list(kept_items)
    done = len(kept_rows)
    for i in range(0, len(novos), CHECKPOINT):
        lote = novos[i:i + CHECKPOINT]
        nomes = [indexer._pretty_name(p) for p in lote]
        nemb_partes.append(encoder.encode_texts(nomes))
        for p in lote:
            items.append({"p": p, "m": os.path.getmtime(p), "d": _duracao(p)})
        done += len(lote)
        _salvar_audio(out, items, np.vstack(nemb_partes).astype(np.float32))
        if progress:
            progress(done, len(arquivos))

    counts["added"] = len(items) - len(kept_items)
    _salvar_audio(out, items, np.vstack(nemb_partes).astype(np.float32))

    # o CLAP e caro e roda em passe separado; aqui so realinha o que ja existe
    if velho_cemb is not None:
        cd = velho_cemb.shape[1]
        cemb = np.zeros((len(items), cd), np.float32)
        cemb[:len(kept_rows)] = velho_cemb[kept_rows]
        _salvar_clap(out, cemb)
    log.info("audio: %s", counts)
    return counts


def _salvar_clap(out, cemb):
    buf = tempfile.NamedTemporaryFile(delete=False, dir=out, suffix=".npz")
    np.savez(buf, cemb=cemb)
    buf.close()
    os.replace(buf.name, Path(out) / "audio_clap.npz")


def clap_disponivel():
    try:
        import clap
        return clap.disponivel()
    except Exception:
        return False


def build_clap(cfg, out=APP, progress=None):
    """Embeda o CONTEUDO dos audios que ainda nao tem vetor CLAP."""
    import clap
    out = Path(out)
    items, nemb, cemb = load_audio(out)
    if not items:
        return {"total": 0, "added": 0, "errors": 0}
    cd = clap.dim()
    if cemb is None or cemb.shape[1] != cd:
        cemb = np.zeros((len(items), cd), np.float32)
    falta = [i for i in range(len(items))
             if float(np.linalg.norm(cemb[i])) < 0.5]
    counts = {"total": len(items), "added": 0, "errors": 0}
    done = len(items) - len(falta)
    if progress:
        progress(done, len(items))
    for bloco in range(0, len(falta), 200):
        for i in falta[bloco:bloco + 200]:
            try:
                cemb[i] = clap.encode_audio(items[i]["p"])
                counts["added"] += 1
            except Exception as e:
                log.info("clap falhou: %s (%s)", items[i]["p"], e)
                counts["errors"] += 1
            done += 1
            if progress and done % 50 == 0:
                progress(done, len(items))
        _salvar_clap(out, cemb)
    _salvar_clap(out, cemb)
    log.info("clap: %s", counts)
    return counts


def search_audio(q, items, nemb, cemb=None, top_k=30, q_en=None):
    """-> [(score, indice)] ordenado. `q_en` e a consulta traduzida (p/ CLAP)."""
    if not items:
        return []
    if q_en and q_en.strip().lower() != q.strip().lower():
        # mesmo ensemble PT+EN do indice visual (+15% de MRR medido)
        dois = encoder.encode_texts([q, q_en])
        tq = dois.mean(axis=0)
        tq = (tq / max(float(np.linalg.norm(tq)), 1e-12)).astype(np.float32)
    else:
        tq = encoder.encode_texts([q])[0]
    scores = NAME_W * (nemb @ tq)

    qw = indexer._words(q)
    if qw:
        lit = np.fromiter(
            (len(qw & indexer._words(" ".join(Path(it["p"]).parts[-3:]))) / len(qw)
             for it in items), dtype=np.float32, count=len(items))
        scores = scores + LIT_W * lit

    if cemb is not None and clap_disponivel():
        try:
            import clap
            qa = clap.encode_text(q_en or q)
            scores = scores + CLAP_W * (cemb @ qa)
        except Exception as e:
            log.info("clap na busca falhou: %s", e)

    idx = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), int(i)) for i in idx]


def _check():
    """Self-test: wav sintetico -> indexa -> acha por nome, PT e EN."""
    import math
    import struct
    import wave
    tmp = Path(tempfile.mkdtemp(prefix="scenefinder-audio-"))
    for nome in ("Tavern_Ambience_Loop.wav", "Rain_Storm_Heavy.wav"):
        with wave.open(str(tmp / nome), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"".join(
                struct.pack("<h", int(8000 * math.sin(i / 20)))
                for i in range(8000)))
    cfg = {"audio_folders": [str(tmp)]}
    out = Path(tempfile.mkdtemp(prefix="scenefinder-audio-idx-"))
    c1 = build_audio(cfg, out=out)
    assert c1["added"] == 2, c1
    c2 = build_audio(cfg, out=out)
    assert c2["added"] == 0 and c2["kept"] == 2, f"incremental quebrado: {c2}"
    items, nemb, _ = load_audio(out)
    assert items[0]["d"] > 0, "duracao nao lida"
    top = search_audio("taverna", items, nemb, top_k=1)
    assert "Tavern" in Path(items[top[0][1]]["p"]).name, "PT nao achou o nome EN"
    top = search_audio("rain storm", items, nemb, top_k=1)
    assert "Rain" in Path(items[top[0][1]]["p"]).name
    print("SONS CHECK OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
