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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import encoder

# mapas de VTT passam facil de 100MP; arquivos locais do proprio usuario
Image.MAX_IMAGE_PIXELS = None

APP = encoder.data_dir()   # indice, thumbs, config e logs
EXTS = {".webp", ".jpg", ".jpeg", ".png"}
# mapas animados: o embedding sai do frame central (uma cena com efeitos em
# movimento continua sendo a mesma cena; 3 frames mediados custariam 3x a
# inferencia por um ganho que nao se mede em mapas quase estaticos)
VIDEO_EXTS = {".webm", ".mp4", ".m4v", ".gif"}
# tokens/retratos/assets soltos nao sao cena, mas passam do filtro de tamanho
EXCL_DIRS = {"__macosx", "thumbs", "tokens", "subjects", "portraits",
             "icons", "ui", "fonts", "cards", "items", "spells", "covers"}
# excluidas mesmo com "tudo": nao sao conteudo, sao copias/metadados
MINIATURAS = {"__macosx", "thumbs"}
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
    "audio_folders": [],
    "indexar_ao_abrir": True,
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


def pastas(cfg):
    """Normaliza `folders` para dicionarios com opcoes proprias.

    Aceita os dois formatos: a string simples de sempre e o dicionario
    {"caminho", "min_side", "min_kb", "ignorar"}. Assim uma pasta de PECAS
    (sprites de decoracao, pequenos por natureza) pode entrar com um limite
    diferente do usado para cenas, sem afetar o resto do acervo.
    """
    saida = []
    for f in cfg.get("folders", []):
        d = {"caminho": f} if isinstance(f, str) else dict(f)
        if d.get("tudo"):
            # sem nenhum filtro: entra toda imagem, inclusive miniaturas,
            # tokens e retratos que normalmente sao descartados
            d.setdefault("min_side", 0)
            d.setdefault("min_kb", 0)
        d.setdefault("min_side", cfg.get("min_side", 1024))
        d.setdefault("min_kb", MIN_BYTES // 1024)
        d.setdefault("ignorar", [])
        d.setdefault("tudo", False)
        # "pecas" = sprites para montar mapa. Ficam no MESMO indice, mas a
        # busca ranqueia em secao separada: 148 mil armarios e bolos afogariam
        # qualquer cena no ranking misto (medido: "taverna a noite" devolvia
        # papel amarelado e bolo de chocolate no top-5)
        d.setdefault("tipo", "cenas")
        # categoria alimenta os filtros do app: scenes | maps | assets.
        # Sem declaracao, pecas viram assets e o resto scenes.
        d.setdefault("categoria",
                     "assets" if d["tipo"] == "pecas" else "scenes")
        saida.append(d)
    return saida


def caminho_de(f):
    return f if isinstance(f, str) else f.get("caminho", "")


def scan(cfg, conhecidos=None):
    """Caminhos absolutos de imagens que parecem mapa.

    `conhecidos`: {caminho normalizado: mtime} de itens que ja estao no indice.
    Para eles o filtro de tamanho e reaproveitado em vez de reabrir a imagem.
    """
    conhecidos = conhecidos or {}
    out, vistos = [], set()
    for opt in pastas(cfg):
        root = Path(opt["caminho"])
        min_side = opt["min_side"]
        min_bytes = opt["min_kb"] * 1024
        ignorar = {i.lower() for i in opt["ignorar"]}
        tudo = opt["tudo"]
        # mesmo no modo "tudo", pasta de miniatura fica de fora: ela so contem
        # copias pequenas de imagens que ja estao no indice, e elas competiam
        # com o mapa original na busca (e o caminho copiado nao servia para
        # jogar). Nunca ha conteudo novo ali.
        excluidas = MINIATURAS if tudo else EXCL_DIRS
        if not root.is_dir():
            log.info("pasta inexistente: %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in excluidas and d.lower() not in ignorar]
            for fn in filenames:
                p = Path(dirpath) / fn
                ext = p.suffix.lower()
                e_video = ext in VIDEO_EXTS
                if ext not in EXTS and not e_video:
                    continue
                if p.stem.lower().endswith("-thumb"):
                    continue          # miniatura solta, mesmo caso das pastas
                chave = os.path.normcase(str(p))
                if chave in vistos:      # pastas configuradas que se sobrepoem
                    continue
                if e_video:
                    # video e sempre conteudo (mapa animado/efeito); o filtro de
                    # dimensao exigiria abrir o container de cada um no scan
                    vistos.add(chave)
                    out.append(str(p))
                    continue
                try:
                    st = p.stat()
                    if st.st_size < min_bytes:
                        continue
                    # ja indexado e intacto: ele passou por este mesmo filtro
                    # antes, entao nao precisa abrir o arquivo de novo. E isto
                    # que faz "adicionar uma pasta" custar segundos em vez de
                    # reabrir o cabecalho de 22 mil imagens.
                    if abs(conhecidos.get(chave, -1) - st.st_mtime) > 1e-6:
                        with Image.open(p) as im:
                            if min(im.size) < min_side:
                                continue
                except Exception as e:
                    log.info("ilegivel no scan: %s (%s)", p, e)
                    continue
                vistos.add(chave)
                out.append(str(p))
    return out


def _thumb_name(path):
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16] + ".webp"


THUMB = 256
# Folga grande sobre os 224 do modelo. Medido: com 512 o pior cosseno cai para
# 0.9986 (abaixo do limite do projeto) e a velocidade e praticamente a mesma
# (1.43x contra 1.38x). 1024 mantem 0.9997 - o valor menor nao compensa.
DRAFT = 1024
# metade dos nucleos logicos: o onnxruntime tambem quer CPU para inferir
WORKERS = max(2, min(8, (os.cpu_count() or 8) // 2))
CHECKPOINT = 500        # imagens entre gravacoes parciais do indice


def _frame_video(caminho):
    """Frame central de um video como PIL RGB (cv2 le mp4/webm/m4v)."""
    import cv2
    cap = cv2.VideoCapture(caminho)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, fr = cap.read()
        if not ok and n > 1:          # container com contagem mentirosa
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        if not ok:
            raise ValueError("nenhum frame legivel")
        return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def _frame_gif(caminho):
    """Frame central de um GIF/imagem animada via PIL."""
    with Image.open(caminho) as im:
        n = getattr(im, "n_frames", 1)
        if n > 1:
            im.seek(n // 2)
        return im.convert("RGB")


def kind_de(caminho):
    return "v" if Path(caminho).suffix.lower() in VIDEO_EXTS else "i"


def _preparar(par):
    """Decodifica UMA vez e devolve (tensor pronto, miniatura).

    Antes o arquivo era aberto duas vezes - uma para o embedding e outra para a
    miniatura - e isso sozinho era ~27% do tempo de indexacao.
    """
    caminho = par[0]
    ext = Path(caminho).suffix.lower()
    try:
        if ext == ".gif":
            rgb = _frame_gif(caminho)
        elif ext in VIDEO_EXTS:
            rgb = _frame_video(caminho)
        else:
            with Image.open(caminho) as im:
                # JPEG decodifica direto em escala reduzida (DCT); folga sobre
                # 224 do modelo e 256 da miniatura para nao perder nitidez
                if im.format == "JPEG":
                    im.draft("RGB", (DRAFT, DRAFT))
                rgb = im.convert("RGB")
        arr = encoder._preprocess(rgb)
        thumb = rgb.copy()
        thumb.thumbnail((THUMB, THUMB))
        return arr, thumb
    except Exception as e:
        log.info("ilegivel: %s (%s)", caminho, e)
        return None


def _gravar_thumb(path, thumb, thumbs_dir):
    name = _thumb_name(path)
    dest = thumbs_dir / name
    if not dest.exists():
        thumb.save(dest, "WEBP", quality=70)
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
    try:
        items = json.loads(meta_p.read_text("utf-8"))["items"]
        emb = np.load(npz_p)["emb"]
    except Exception as e:
        log.info("indice ilegivel (%s): descartando", e)
        return np.zeros((0, encoder.dim()), dtype=np.float32), []
    if emb.shape[1] != encoder.dim():
        # indice de outro modelo: inutil, sera reconstruido do zero
        log.info("indice com dim %s, modelo usa %s: descartando",
                 emb.shape[1], encoder.dim())
        return np.zeros((0, encoder.dim()), dtype=np.float32), []
    if len(emb) != len(items):
        # desalinhado: cada score sairia casado com o item errado, em silencio
        log.info("indice desalinhado (%s vetores, %s itens): descartando",
                 len(emb), len(items))
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


def _carregar_nemb(out, n_esperado):
    """Nomes ja embedados, se casarem com o indice antigo. None = reconstruir."""
    p = Path(out) / "names.npz"
    if not p.exists():
        return None
    try:
        nemb = np.load(p)["nemb"]
    except Exception:
        return None
    if len(nemb) != n_esperado or nemb.shape[1] != encoder.dim():
        return None
    return nemb


def load_name_index(items, out=APP):
    nemb = _carregar_nemb(out, len(items))
    return build_name_index(items, out) if nemb is None else nemb


def _empilhar(base, novos, d):
    return np.vstack([base.reshape(-1, d),
                      np.array(novos, dtype=np.float32).reshape(-1, d)])


def _embed_seguro(arrays):
    """Um erro no lote (falta de memoria, imagem estranha) nao pode derrubar a
    indexacao inteira e reaparecer no mesmo ponto a cada tentativa.

    Devolve (vetores, ok) onde ok[i] diz se aquele item pode ser gravado. Item
    que falhou NAO entra no indice: gravar um vetor zero o marcaria como pronto
    e ele nunca mais seria reprocessado - uma falha de GPU no meio da passada
    envenenaria o resto do acervo em silencio.
    """
    try:
        return encoder.encode_prepared(np.stack(arrays)), [True] * len(arrays)
    except Exception as e:
        log.info("lote falhou (%s), tentando uma a uma", e)
        saida, ok = [], []
        for a in arrays:
            try:
                saida.append(encoder.encode_prepared(np.stack([a]))[0])
                ok.append(True)
            except Exception as e2:
                log.info("imagem falhou no embed: %s", e2)
                saida.append(np.zeros(encoder.dim(), np.float32))
                ok.append(False)
        return np.array(saida, dtype=np.float32), ok


def assinatura_criterio(cfg):
    """Resume as regras de filtro. Se mudarem, o atalho do scan nao vale mais."""
    return json.dumps(
        sorted([os.path.normcase(p["caminho"]), p["min_side"], p["min_kb"],
                sorted(x.lower() for x in p["ignorar"])] for p in pastas(cfg)))


def _criterio_salvo(out):
    """Regras com que o indice atual foi filtrado (None se desconhecido)."""
    try:
        return json.loads((Path(out) / "meta.json").read_text("utf-8")).get("criterio")
    except Exception:
        return None


def _salvar(out, kept_rows, kept_items, old_emb, new_embs, new_items,
            old_nemb=None, new_nembs=None, min_side=None):
    """Grava indice + nomes + meta de forma atomica. Serve tanto para o
    checkpoint quanto para o fim: o que esta em disco fica sempre consistente."""
    d = encoder.dim()
    emb = _empilhar(old_emb[kept_rows], new_embs, d)
    items = kept_items + new_items

    # savez cru, nao comprimido: a compressao levava ~3 s por gravacao e o
    # checkpoint roda varias vezes; custa ~10% de disco a mais
    buf = tempfile.NamedTemporaryFile(delete=False, dir=out, suffix=".npz")
    np.savez(buf, emb=emb)
    buf.close()
    os.replace(buf.name, out / "index.npz")

    if old_nemb is not None:
        nemb = _empilhar(old_nemb[kept_rows], new_nembs or [], d)
        if len(nemb) == len(items):
            buf = tempfile.NamedTemporaryFile(delete=False, dir=out, suffix=".npz")
            np.savez(buf, nemb=nemb)
            buf.close()
            os.replace(buf.name, out / "names.npz")

    _atomic_write(out / "meta.json",
                  json.dumps({"items": items, "criterio": min_side}).encode("utf-8"))
    return items


def build_index(cfg, out=APP, progress=None, limit=None, plano=None):
    """Incremental: chave path+mtime. Retorna dict de contagens."""
    out = Path(out)
    thumbs_dir = out / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    old_emb, old_items = load_index(out)
    # normcase: Windows nao diferencia maiuscula, e config com caixa diferente
    # faria o mesmo arquivo ser reindexado do zero
    old = {os.path.normcase(it["p"]): (it["m"], i)
           for i, it in enumerate(old_items)}

    # o filtro de tamanho so pode ser reaproveitado se as regras forem as
    # mesmas; se mudaram, todo arquivo precisa ser reavaliado
    criterio = assinatura_criterio(cfg)
    reaproveitar = (_criterio_salvo(out) == criterio)
    files = scan(cfg, {k: v[0] for k, v in old.items()} if reaproveitar else None)
    if limit:
        files = files[:limit]

    kept_rows, kept_items, todo = [], [], []
    for p in files:
        m = os.path.getmtime(p)
        prev = old.get(os.path.normcase(p))
        # tambem reprocessa se a miniatura sumiu, senao o card fica quebrado
        if (prev and abs(prev[0] - m) < 1e-6
                and (thumbs_dir / old_items[prev[1]]["t"]).exists()):
            kept_rows.append(prev[1])
            kept_items.append(old_items[prev[1]])
        else:
            todo.append((p, m))

    removed = len(old) - len(kept_rows)
    counts = {"total": len(files), "kept": len(kept_rows),
              "added": 0, "removed": removed, "errors": 0}
    done = len(kept_rows)
    if plano:                       # quantas serao realmente processadas
        plano(len(kept_rows), len(todo))
    if progress:
        progress(done, len(files))

    # nomes tambem sao incrementais: re-embedar os 22 mil a cada indexacao
    # custava dezenas de minutos por uma informacao que nao mudou
    old_nemb = _carregar_nemb(out, len(old_items))
    new_nembs = [] if old_nemb is not None else None

    new_embs, new_items = [], []
    if todo:
        BATCH = 16

        def gravar_parcial():
            """Checkpoint: sem isto, cair na imagem 20.000 joga fora horas."""
            if not new_items:
                return
            _salvar(out, kept_rows, kept_items, old_emb, new_embs, new_items,
                    old_nemb, new_nembs, criterio)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            desde_checkpoint = 0
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                # decodificar/redimensionar em varias threads enquanto o ONNX
                # infere o lote anterior: PIL solta o GIL nessas operacoes
                preparados = list(pool.map(_preparar, batch))
                arrays, prontos = [], []
                for (p, m), r in zip(batch, preparados):
                    if r is None:
                        counts["errors"] += 1
                        continue
                    arrays.append(r[0])
                    prontos.append((p, m, r[1]))
                if arrays:
                    vecs, validos = _embed_seguro(arrays)
                    gravados = []
                    for (p, m, thumb), v, bom in zip(prontos, vecs, validos):
                        if not bom:
                            counts["errors"] += 1
                            continue
                        try:
                            t = _gravar_thumb(p, thumb, thumbs_dir)
                        except Exception as e:
                            log.info("nao consegui gravar a miniatura: %s (%s)", p, e)
                            counts["errors"] += 1
                            continue
                        new_embs.append(np.asarray(v, dtype=np.float32))
                        new_items.append({"p": p, "m": m, "t": t,
                                          "k": kind_de(p)})
                        gravados.append(p)
                    if new_nembs is not None and gravados:
                        nv = encoder.encode_texts([_pretty_name(p) for p in gravados])
                        new_nembs.extend(np.asarray(x, np.float32) for x in nv)
                done += len(batch)
                desde_checkpoint += len(batch)
                if desde_checkpoint >= CHECKPOINT:
                    gravar_parcial()
                    desde_checkpoint = 0
                if progress:
                    progress(min(done, len(files)), len(files))

    counts["added"] = len(new_items)
    items = _salvar(out, kept_rows, kept_items, old_emb, new_embs, new_items,
                    old_nemb, new_nembs, criterio)

    # thumbs orfaos: so no fim da passada completa - se limpasse no checkpoint,
    # apagaria a thumb de item que ainda nao foi reprocessado
    valid = {it["t"] for it in items}
    for f in thumbs_dir.glob("*.webp"):
        if f.name not in valid:
            f.unlink(missing_ok=True)

    if old_nemb is None:      # indice antigo sem nomes: constroi uma vez
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


def search(query, emb, items, top_k=60, nemb=None, mask=None, q_en=None):
    """Semantica da imagem + do nome + match literal -> [(score, indice)].

    `mask`: array booleano do tamanho do indice; False esconde o item da busca
    sem tira-lo do indice (permite ligar/desligar pastas sem reindexar).
    `q_en`: consulta traduzida. Mediar o embedding PT com o EN melhora o MRR de
    0.828 para 0.950 no gabarito de nomes (tools/bench_ensemble.py) - o encoder
    e multilingue, mas o lado ingles casa melhor com os nomes dos arquivos.
    """
    if not len(items):
        return []
    if q_en and q_en.strip().lower() != query.strip().lower():
        dois = encoder.encode_texts([query, q_en])
        q = dois.mean(axis=0)
        q = (q / max(float(np.linalg.norm(q)), 1e-12)).astype(np.float32)
    else:
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

    # video e gif sinteticos: o _preparar tem que devolver tensor + thumb
    import cv2
    vdir = Path(tempfile.mkdtemp(prefix="scenefinder-video-"))
    mp4 = str(vdir / "t.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(20):
        vw.write(np.full((48, 64, 3), i * 10 % 255, np.uint8))
    vw.release()
    gif = vdir / "t.gif"
    quadros = [Image.new("RGB", (64, 48), (i * 30 % 255, 0, 0)) for i in range(6)]
    quadros[0].save(gif, save_all=True, append_images=quadros[1:])
    for arq in (mp4, str(gif)):
        r = _preparar((arq, 0))
        assert r is not None and r[0].shape[0] == 3, f"video/gif falhou: {arq}"
        assert kind_de(arq) == "v", arq
    print("video/gif OK")
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
