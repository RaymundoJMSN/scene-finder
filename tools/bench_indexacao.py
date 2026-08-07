"""Compara o pipeline de indexacao antigo com o novo, em imagens reais.

Mede duas coisas que precisam andar juntas: quanto mais rapido ficou e se o
embedding continua o mesmo. Velocidade que muda o resultado da busca nao serve.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

import encoder  # noqa: E402
import indexer  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48


def antigo(caminhos, thumbs_dir):
    """Como era: abre para o embedding, depois abre DE NOVO para a miniatura."""
    vecs = []
    for i in range(0, len(caminhos), 16):
        lote = caminhos[i:i + 16]
        imgs = []
        for p in lote:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        vecs.append(encoder.encode_images(imgs, batch_size=16))
        for p in lote:
            with Image.open(p) as im:
                th = im.convert("RGB")
                th.thumbnail((256, 256))
    return np.vstack(vecs)


def novo(caminhos, thumbs_dir):
    """Como ficou: uma decodificacao por imagem, em varias threads."""
    from concurrent.futures import ThreadPoolExecutor
    vecs = []
    with ThreadPoolExecutor(max_workers=indexer.WORKERS) as pool:
        for i in range(0, len(caminhos), 16):
            lote = caminhos[i:i + 16]
            prep = list(pool.map(indexer._preparar, [(p, 0) for p in lote]))
            arrays = [r[0] for r in prep if r]
            if arrays:
                vecs.append(encoder.encode_prepared(np.stack(arrays)))
    return np.vstack(vecs) if vecs else np.zeros((0, encoder.dim()), np.float32)


def main():
    cfg = indexer.load_config()
    todos = indexer.scan(cfg)
    amostra = todos[:: max(1, len(todos) // N)][:N]
    mb = sum(Path(p).stat().st_size for p in amostra) / 1e6
    print(f"amostra: {len(amostra)} imagens, {mb:.0f} MB no total")
    fmts = {}
    for p in amostra:
        e = Path(p).suffix.lower()
        fmts[e] = fmts.get(e, 0) + 1
    print(f"formatos: {fmts}")

    tdir = Path(indexer.APP) / "thumbs"
    tdir.mkdir(exist_ok=True)

    encoder.encode_texts(["aquecer"])          # tira o custo de carregar da conta

    t0 = time.perf_counter()
    va = antigo(amostra, tdir)
    ta = time.perf_counter() - t0

    t0 = time.perf_counter()
    vb = novo(amostra, tdir)
    tb = time.perf_counter() - t0

    print(f"\nantigo: {ta:6.1f}s  ({ta/len(amostra)*1000:5.0f} ms/img)")
    print(f"novo:   {tb:6.1f}s  ({tb/len(amostra)*1000:5.0f} ms/img)")
    print(f"ganho:  {ta/max(tb,1e-9):.1f}x")

    n = min(len(va), len(vb))
    cos = float(np.sum(va[:n] * vb[:n], axis=-1).mean())
    pior = float(np.sum(va[:n] * vb[:n], axis=-1).min())
    print(f"\ncosseno medio antigo x novo: {cos:.4f} (pior caso {pior:.4f})")
    if cos < 0.999:
        print("ATENCAO: o embedding mudou - revise o draft antes de aceitar")
    else:
        print("embeddings equivalentes")

    horas = 22507 * (tb / len(amostra)) / 3600
    print(f"\nprojecao para 22.507 imagens: {horas:.1f} h "
          f"(antes seriam {22507*(ta/len(amostra))/3600:.1f} h)")


if __name__ == "__main__":
    main()
