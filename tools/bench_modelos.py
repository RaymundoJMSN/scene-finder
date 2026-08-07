"""Compara modelos de imagem (CLIP atual x SigLIP2) no acervo real.

Nao adianta trocar de modelo por reputacao: aqui a troca so se justifica se ele
achar melhor ESTAS imagens. O gabarito sai dos proprios nomes de arquivo, e a
busca usa somente o embedding visual - o sinal de nome fica de fora de proposito,
senao mediria o texto e nao o modelo.

Metricas: acerto@1 e MRR (1/posicao do primeiro acerto).
"""
import collections
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

import indexer  # noqa: E402

random.seed(7)
N_GRUPOS = 40          # consultas
POR_GRUPO = 2          # imagens corretas por consulta
DISTRATORES = 400
# palavras que marcam variante do mesmo mapa, nao assunto diferente
VARIANTE = {"day", "night", "original", "clean", "empty", "winter", "summer",
            "rain", "snow", "fog", "gridless", "gridded", "variant", "variants",
            "scenes", "assets", "tier", "gl", "g", "thumb", "part", "gm", "player"}


def palavras(txt):
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", txt)
    return [w for w in re.split(r"[^A-Za-z]+", txt.lower()) if len(w) > 2]


def assunto(caminho):
    """Nome do mapa sem as marcas de variante -> serve de consulta e de gabarito."""
    p = Path(caminho)
    base = [w for w in palavras(p.stem) if w not in VARIANTE]
    if len(base) < 2:                       # nome pobre: usa a pasta
        base = [w for w in palavras(p.parent.name) if w not in VARIANTE]
    return " ".join(base[:4])


def montar():
    # varre o acervo direto: o indice pode estar vazio (troca de modelo)
    items = [{"p": p} for p in indexer.scan(indexer.load_config())]
    grupos = collections.defaultdict(list)
    for i, it in enumerate(items):
        a = assunto(it["p"])
        if len(a.split()) >= 2:
            grupos[a].append(i)
    bons = [(a, ix) for a, ix in grupos.items() if len(ix) >= POR_GRUPO]
    random.shuffle(bons)
    bons = bons[:N_GRUPOS]

    positivos, gabarito = [], {}
    for a, ix in bons:
        escolhidos = random.sample(ix, POR_GRUPO)
        gabarito[a] = set(range(len(positivos), len(positivos) + POR_GRUPO))
        positivos += escolhidos

    usados = set(positivos)
    resto = [i for i in range(len(items)) if i not in usados]
    ruido = random.sample(resto, min(DISTRATORES, len(resto)))
    todos = positivos + ruido
    return [items[i]["p"] for i in todos], gabarito, [a for a, _ in bons]


def avaliar(nome, vec_img, vec_txt, caminhos, gabarito, consultas):
    imgs = []
    for p in caminhos:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    E = vec_img(imgs)
    acerto1, mrr = 0, 0.0
    for c in consultas:
        q = vec_txt([c])[0]
        ordem = np.argsort(-(E @ q))
        certos = gabarito[c]
        if ordem[0] in certos:
            acerto1 += 1
        for pos, idx in enumerate(ordem[:50], 1):
            if idx in certos:
                mrr += 1 / pos
                break
    n = len(consultas)
    print(f"{nome:10s} acerto@1: {acerto1}/{n} ({acerto1/n:.0%})   MRR: {mrr/n:.3f}")
    return mrr / n


def main():
    caminhos, gabarito, consultas = montar()
    print(f"{len(consultas)} consultas, {len(caminhos)} imagens no pool\n")
    print("exemplos:", ", ".join(f"'{c}'" for c in consultas[:5]), "\n")

    import encoder  # CLIP atual, ONNX int8 (o que esta em producao)
    clip = avaliar("CLIP", lambda ims: encoder.encode_images(ims),
                   lambda t: encoder.encode_texts(t), caminhos, gabarito, consultas)

    import torch
    from transformers import AutoModel, AutoProcessor
    mid = "google/siglip2-base-patch16-224"
    proc = AutoProcessor.from_pretrained(mid)
    mod = AutoModel.from_pretrained(mid).eval()

    def tensor(v):
        # transformers 5 as vezes devolve o output completo em vez do tensor
        if hasattr(v, "pooler_output"):
            return v.pooler_output
        if hasattr(v, "last_hidden_state"):
            return v.last_hidden_state.mean(1)
        return v

    def s_img(ims):
        saidas = []
        for i in range(0, len(ims), 16):
            x = proc(images=ims[i:i + 16], return_tensors="pt")
            with torch.no_grad():
                v = tensor(mod.get_image_features(**x))
            saidas.append(v.numpy())
        v = np.vstack(saidas)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def s_txt(ts):
        x = proc(text=ts, return_tensors="pt", padding="max_length",
                 max_length=64, truncation=True)
        with torch.no_grad():
            v = tensor(mod.get_text_features(**x))
        v = v.numpy()
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    sig = avaliar("SigLIP2", s_img, s_txt, caminhos, gabarito, consultas)

    print(f"\nSigLIP2 {'GANHA' if sig > clip else 'PERDE'} "
          f"({(sig - clip) / max(clip, 1e-9):+.0%} de MRR)")

    # consultas em portugues: o multilingue precisa sobreviver a troca
    pt = {"taverna a noite": "tavern", "templo na selva": "temple",
          "navio pirata": "ship", "biblioteca": "library"}
    print("\nsanidade multilingue (similaridade PT x EN):")
    for p, e in pt.items():
        c1 = float(encoder.encode_texts([p])[0] @ encoder.encode_texts([e])[0])
        c2 = float(s_txt([p])[0] @ s_txt([e])[0])
        print(f"  '{p}' x '{e}':  CLIP {c1:.3f}   SigLIP2 {c2:.3f}")


if __name__ == "__main__":
    main()
