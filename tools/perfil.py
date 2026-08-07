"""Mede onde a indexacao gasta tempo, por etapa.

Otimizar sem isto e chute: se o gargalo for decodificar JPEG de 100 MP, acelerar
a inferencia (GPU) quase nao muda o total.
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

N = 24


def cron(f):
    t = time.perf_counter()
    r = f()
    return r, time.perf_counter() - t


def main():
    cfg = indexer.load_config()
    todos = indexer.scan(cfg)
    print(f"acervo: {len(todos)} imagens")
    amostra = todos[:: max(1, len(todos) // N)][:N]

    tam = [Path(p).stat().st_size for p in amostra]
    print(f"amostra: {len(amostra)} | mediana {np.median(tam)/1e6:.1f} MB | "
          f"maior {max(tam)/1e6:.1f} MB")

    # 1) abrir + decodificar + converter (como o indexer faz hoje)
    def abrir_tudo():
        out = []
        for p in amostra:
            with Image.open(p) as im:
                out.append(im.convert("RGB"))
        return out
    imgs, t_dec = cron(abrir_tudo)
    px = sum(i.size[0] * i.size[1] for i in imgs)
    print(f"\n1. decodificar        {t_dec:6.2f}s  ({t_dec/len(amostra)*1000:5.0f} ms/img)"
          f"  [{px/1e6:.0f} MP no total]")

    # 2) preprocess (resize + normalizacao)
    _, t_pre = cron(lambda: np.stack([encoder._preprocess(i) for i in imgs]))
    print(f"2. preparar (resize)  {t_pre:6.2f}s  ({t_pre/len(amostra)*1000:5.0f} ms/img)")

    # 3) inferencia ONNX
    lote = np.stack([encoder._preprocess(i) for i in imgs])
    sess = encoder._session("image.onnx")
    saida = encoder._saida(sess)
    _, t_inf = cron(lambda: sess.run(None, {"pixel_values": lote})[saida])
    print(f"3. inferencia (CPU)   {t_inf:6.2f}s  ({t_inf/len(amostra)*1000:5.0f} ms/img)")

    # 4) miniatura - hoje o indexer ABRE O ARQUIVO DE NOVO para isto
    tdir = Path(indexer.APP) / "thumbs"

    def thumbs():
        for p in amostra:
            with Image.open(p) as im:
                im2 = im.convert("RGB")
                im2.thumbnail((256, 256))
    _, t_thumb = cron(thumbs)
    print(f"4. miniatura (reabre) {t_thumb:6.2f}s  ({t_thumb/len(amostra)*1000:5.0f} ms/img)")

    total = t_dec + t_pre + t_inf + t_thumb
    print(f"\nTOTAL                 {total:6.2f}s  ({total/len(amostra)*1000:5.0f} ms/img)")
    for nome, t in [("decodificar", t_dec), ("preparar", t_pre),
                    ("inferencia", t_inf), ("miniatura", t_thumb)]:
        print(f"   {nome:12s} {t/total*100:4.0f}%")

    # 5) o mesmo decode usando draft (JPEG decodifica menor via DCT)
    def abrir_draft():
        out = []
        for p in amostra:
            im = Image.open(p)
            im.draft("RGB", (224, 224))     # so tem efeito em JPEG
            out.append(im.convert("RGB"))
        return out
    imgs_d, t_draft = cron(abrir_draft)
    print(f"\n5. decodificar com draft {t_draft:6.2f}s "
          f"({t_draft/len(amostra)*1000:5.0f} ms/img)  "
          f"= {t_dec/max(t_draft,1e-9):.1f}x mais rapido")
    print(f"   tamanhos apos draft: {[i.size for i in imgs_d[:3]]}")

    # o draft muda o embedding? (o que importa e o cosseno, nao o pixel)
    a = encoder.encode_images(imgs[:8])
    b = encoder.encode_images(imgs_d[:8])
    cos = float(np.sum(a * b, axis=-1).mean())
    print(f"   cosseno normal x draft: {cos:.4f}")

    print("\nprovedores ONNX disponiveis:", encoder.ort.get_available_providers())


if __name__ == "__main__":
    main()
