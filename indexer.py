"""Scene Finder - indexador: varre pastas, embeda com CLIP, gera thumbs.

Uso:  python indexer.py           -> indexa tudo (incremental)
      python indexer.py --check   -> self-test com 20 imagens reais
"""
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

import encoder

# mapas de VTT passam facil de 100MP; arquivos locais do proprio usuario
Image.MAX_IMAGE_PIXELS = None

APP = encoder.data_dir()   # indice, thumbs, config e logs
EXTS = {".webp", ".jpg", ".jpeg", ".png"}
# tokens/retratos/assets soltos nao sao cena, mas passam do filtro de tamanho
EXCL_DIRS = {"__macosx", "thumbs", "tokens", "subjects", "portraits",
             "icons", "ui", "fonts", "cards", "items", "spells", "covers"}
# ponytail: arquivos <60KB nunca sao mapa >=1024px; evita abrir header de milhares de icones
MIN_BYTES = 60 * 1024

logging.basicConfig(
    filename=APP / "indexer.log", level=logging.INFO,
    format="%(asctime)s %(message)s", encoding="utf-8")
log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "foundry_data": "",
    "folders": [],
    # pastas indexadas mas escondidas da busca: permite manter uma pasta no
    # indice e liga-la so quando precisar, sem reindexar de novo
    "folders_off": [],
    "sources_off": [],          # fontes online desligadas (mesma ideia)
    "min_side": 1024,
    "port": 8060,
    "top_k": 60,
    "subreddits": ["battlemaps", "dndmaps", "FantasyMaps"],
    "kemono": [],
}


def find_foundry():
    """Procura a pasta Data do Foundry nos lugares obvios (todos os drives)."""
    cands = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands.append(Path(local) / "FoundryVTT" / "Data")
    for d in "CDEFGHXYZ":
        cands += [Path(f"{d}:/FoundryVTT/Data"),
                  Path(f"{d}:/FoundryVTT/foundrydata/Data")]
    for c in cands:
        if c.is_dir():
            return c
    return None


def config_path():
    return APP / "config.json"


def save_config(cfg):
    _atomic_write(config_path(), json.dumps(cfg, indent=2).encode("utf-8"))


def load_config():
    """Cria o config no primeiro uso, achando o Foundry sozinho se der."""
    p = config_path()
    if p.exists():
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(json.loads(p.read_text("utf-8")))
        return cfg
    cfg = dict(DEFAULT_CONFIG)
    root = find_foundry()
    if root:
        cfg["foundry_data"] = str(root)
        cfg["folders"] = [str(root / sub) for sub in ("Assets", "modules")
                          if (root / sub).is_dir()] or [str(root)]
    save_config(cfg)
    return cfg


def scan(cfg):
    """Caminhos absolutos de imagens que parecem mapa (lado menor >= min_side)."""
    min_side = cfg.get("min_side", 1024)
    out = []
    for folder in cfg["folders"]:
        root = Path(folder)
        if not root.is_dir():
            log.info("pasta inexistente: %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in EXCL_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in EXTS or p.stem.lower().endswith("-thumb"):
                    continue
                try:
                    if p.stat().st_size < MIN_BYTES:
                        continue
                    with Image.open(p) as im:
                        if min(im.size) < min_side:
                            continue
                except Exception as e:
                    log.info("ilegivel no scan: %s (%s)", p, e)
                    continue
                out.append(str(p))
    return out


def _thumb_name(path):
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16] + ".webp"


def _make_thumb(path, thumbs_dir):
    name = _thumb_name(path)
    dest = thumbs_dir / name
    if not dest.exists():
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            im.save(dest, "WEBP", quality=70)
    return name


def _atomic_write(path, data: bytes):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def load_index(out=APP):
    meta_p, npz_p = Path(out) / "meta.json", Path(out) / "index.npz"
    if not (meta_p.exists() and npz_p.exists()):
        return np.zeros((0, encoder.dim()), dtype=np.float32), []
    items = json.loads(meta_p.read_text("utf-8"))["items"]
    emb = np.load(npz_p)["emb"]
    if emb.shape[1] != encoder.dim():
        # indice de outro modelo: inutil, sera reconstruido do zero
        log.info("indice com dim %s, modelo usa %s: descartando",
                 emb.shape[1], encoder.dim())
        return np.zeros((0, encoder.dim()), dtype=np.float32), []
    return emb, items


# marcas de variante do MESMO mapa: dia/noite, com grade, limpo, estacoes...
VARIANTE = {"day", "night", "dawn", "dusk", "original", "clean", "empty",
            "winter", "summer", "spring", "autumn", "fall", "rain", "rainy",
            "snow", "snowy", "fog", "foggy", "storm", "dust", "wind", "cloudy",
            "dark", "lit", "lights", "sunset", "sunrise", "overcast",
            "gridless", "gridded", "grid", "nogrid", "variant", "variants",
            "gm", "player", "dm", "hd", "low", "high", "res", "alt", "version",
            "top", "bottom", "upper", "lower", "floor", "level", "part"}


def map_key(path):
    """Identidade do MAPA, ignorando a variante.

    Um mesmo mapa costuma vir em 5-15 arquivos (Day/Night/Gridless/...). Sem
    isto, uma busca por 'wild west saloon' devolve 12 resultados que sao 3 mapas.
    A chave e a pasta-mae + as palavras do nome que nao sao marca de variante.
    """
    p = Path(path)
    # as PRIMEIRAS palavras nomeiam o mapa; o que vem depois e variante
    # (WildWestSaloon_NoSkull_Day, WildWestSaloon_SandStorm...). Listar todas as
    # variacoes possiveis seria infinito - o prefixo resolve sozinho.
    nome = [w for w in _word_list(p.stem) if w not in VARIANTE][:3]
    # junta tambem o mesmo mapa baixado em pastas diferentes (pacote solto +
    # modulo do Foundry, algo comum no acervo)
    if len(nome) >= 2:
        return " ".join(nome)
    nome = sorted(nome)
    # nome pobre ("map01", "01a"): so a pasta distingue
    pasta = p.parent
    if pasta.name.lower() in ("variants", "scenes", "gridless", "gridded",
                              "thumbs", "assets", "maps", "webp", "jpg", "png"):
        pasta = pasta.parent
    return f"{pasta.name.lower()}|{' '.join(nome)}"


def _pretty_name(path):
    """'czepeku-scenes-081-hellfire-prison/HellfirePrison_Glow' -> texto legivel."""
    parts = Path(path).parts[-3:]
    return " ".join(sorted(_words(" ".join(parts)), key=len, reverse=True))


def build_name_index(items, out=APP):
    """Embeda os nomes dos arquivos no MESMO espaco do CLIP (modelo de texto
    multilingual) -> busca em PT casa com nome em EN. Independe das imagens."""
    out = Path(out)
    names = [_pretty_name(it["p"]) for it in items]
    nemb = (encoder.encode_texts(names) if names
            else np.zeros((0, encoder.dim()), np.float32))
    buf = tempfile.NamedTemporaryFile(delete=False, dir=out, suffix=".npz")
    np.savez_compressed(buf, nemb=nemb)
    buf.close()
    os.replace(buf.name, out / "names.npz")
    return nemb


def load_name_index(items, out=APP):
    p = Path(out) / "names.npz"
    if p.exists():
        nemb = np.load(p)["nemb"]
        if len(nemb) == len(items):
            return nemb
    return build_name_index(items, out)


def build_index(cfg, out=APP, progress=None, limit=None):
    """Incremental: chave path+mtime. Retorna dict de contagens."""
    out = Path(out)
    thumbs_dir = out / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    old_emb, old_items = load_index(out)
    old = {it["p"]: (it["m"], i) for i, it in enumerate(old_items)}

    files = scan(cfg)
    if limit:
        files = files[:limit]

    kept_rows, kept_items, todo = [], [], []
    for p in files:
        m = os.path.getmtime(p)
        prev = old.get(p)
        if prev and abs(prev[0] - m) < 1e-6:
            kept_rows.append(prev[1])
            kept_items.append(old_items[prev[1]])
        else:
            todo.append((p, m))

    removed = len(old) - len(kept_rows)
    counts = {"total": len(files), "kept": len(kept_rows),
              "added": 0, "removed": removed, "errors": 0}
    done = len(kept_rows)
    if progress:
        progress(done, len(files))

    new_embs, new_items = [], []
    if todo:
        BATCH = 16
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            imgs, ok = [], []
            for p, m in batch:
                try:
                    with Image.open(p) as im:
                        imgs.append(im.convert("RGB"))
                    ok.append((p, m))
                except Exception as e:
                    log.info("ilegivel no embed: %s (%s)", p, e)
                    counts["errors"] += 1
            if imgs:
                vecs = encoder.encode_images(imgs, batch_size=BATCH)
                for (p, m), v in zip(ok, vecs):
                    try:
                        t = _make_thumb(p, thumbs_dir)
                    except Exception as e:
                        log.info("thumb falhou: %s (%s)", p, e)
                        counts["errors"] += 1
                        continue
                    new_embs.append(np.asarray(v, dtype=np.float32))
                    new_items.append({"p": p, "m": m, "t": t})
            done += len(batch)
            if progress:
                progress(min(done, len(files)), len(files))

    counts["added"] = len(new_items)
    d = encoder.dim()
    emb = np.vstack([old_emb[kept_rows].reshape(-1, d)] +
                    [np.array(new_embs, dtype=np.float32).reshape(-1, d)])
    items = kept_items + new_items

    # thumbs orfaos
    valid = {it["t"] for it in items}
    for f in thumbs_dir.glob("*.webp"):
        if f.name not in valid:
            f.unlink(missing_ok=True)

    buf = tempfile.NamedTemporaryFile(delete=False, dir=out, suffix=".npz")
    np.savez_compressed(buf, emb=emb)
    buf.close()
    os.replace(buf.name, out / "index.npz")
    _atomic_write(out / "meta.json",
                  json.dumps({"items": items}).encode("utf-8"))
    build_name_index(items, out)
    log.info("index: %s", counts)
    return counts


def _word_list(text):
    """Tokens minusculos sem acento, NA ORDEM, quebrando em camelCase."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(w) > 2]


def _words(text):
    """Mesmos tokens, como conjunto (para interseccao na busca literal)."""
    return set(_word_list(text))


def _name_words(item):
    if "w" not in item:
        # so o final do caminho importa: nome do arquivo + 2 pastas acima
        item["w"] = _words(" ".join(Path(item["p"]).parts[-3:]))
    return item["w"]


# CLIP acha mapas top-down parecidos entre si (scores empatados ~0.3), entao o
# nome do arquivo decide. Dois sinais de nome: semantico (cruza PT/EN) e literal
# (match exato vale mais que "parecido").
NAME_SEM_WEIGHT = 0.5
NAME_LIT_WEIGHT = 0.25


def search(query, emb, items, top_k=60, nemb=None, mask=None):
    """Semantica da imagem + do nome + match literal -> [(score, indice)].

    `mask`: array booleano do tamanho do indice; False esconde o item da busca
    sem tira-lo do indice (permite ligar/desligar pastas sem reindexar).
    """
    if not len(items):
        return []
    q = encoder.encode_texts([query])[0]
    scores = emb @ q

    if nemb is not None and len(nemb) == len(items):
        scores = scores + NAME_SEM_WEIGHT * (nemb @ q)

    qw = _words(query)
    if qw:
        lit = np.fromiter(
            (len(qw & _name_words(it)) / len(qw) for it in items),
            dtype=np.float32, count=len(items))
        scores = scores + NAME_LIT_WEIGHT * lit

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    idx = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), int(i)) for i in idx
            if np.isfinite(scores[i])]  # indice global p/ /api/open


def _check():
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="scenefinder-check-"))
    c1 = build_index(cfg, out=tmp, limit=20)
    assert c1["added"] == 20, f"esperava 20 novos, veio {c1}"
    emb, items = load_index(tmp)
    d = encoder.dim()
    assert len(items) == 20 and emb.shape == (20, d), \
        f"indice invalido: {len(items)} itens, shape {emb.shape}, esperado (20,{d})"
    c2 = build_index(cfg, out=tmp, limit=20)
    assert c2["added"] == 0 and c2["kept"] == 20, f"incremental quebrado: {c2}"
    nemb = load_name_index(items, tmp)
    assert nemb.shape == (20, d), f"indice de nomes invalido: {nemb.shape}"

    chaves = {map_key(it["p"]) for it in items}
    assert chaves, "map_key nao gerou nenhuma chave"
    res = search("tavern", emb, items, top_k=20, nemb=nemb)
    assert len(res) == 20, "busca nao retornou tudo"
    assert all(res[i][0] >= res[i + 1][0] for i in range(len(res) - 1)), "sem ordenacao"
    print("CHECK OK -", items[res[0][1]]["p"])


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        cfg = load_config()
        c = build_index(cfg, progress=lambda d, t: print(f"\r{d}/{t}", end=""))
        print("\n", c)
