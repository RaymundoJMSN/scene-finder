"""Prova que o ONNX int8 nao degrada a busca em relacao ao torch.

Criterios (o de imagem e mais frouxo de proposito):
- texto >= 0.99: a query passa por ele a cada busca, e ele e comparado contra um
  indice que pode ter sido gerado por outra versao. Precisa ser fiel.
- imagem >= 0.95: a quantizacao int8 desvia ~0.97 do fp32, mas como o indice e
  reconstruido com o proprio ONNX o desvio some - todos os vetores saem do mesmo
  encoder. O que precisa se manter e o RANKING, medido abaixo.
- top-10 >= 8/10 em comum com o ranking do torch: este e o teste que importa.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

import encoder  # noqa: E402
import indexer  # noqa: E402

QUERIES = ["taverna a noite", "templo na selva", "navio pirata",
           "biblioteca real", "jungle temple", "desert city at night"]


def cos(a, b):
    return float(np.sum(a * b, axis=-1).mean())


def main():
    emb, items = indexer.load_index(ROOT)
    print(f"indice: {len(items)} itens")

    # --- texto: ONNX vs torch ---
    t_onnx = encoder.encode_texts(QUERIES)
    from sentence_transformers import SentenceTransformer  # so nesta ferramenta
    t_torch = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1").encode(
        QUERIES, normalize_embeddings=True).astype(np.float32)
    c_txt = cos(t_onnx, t_torch)
    print(f"texto  cosseno medio: {c_txt:.4f}")

    # --- imagem: ONNX vs os embeddings gravados no indice (torch fp32) ---
    idx = np.linspace(0, len(items) - 1, 24, dtype=int)
    imgs = [Image.open(items[i]["p"]) for i in idx]
    i_onnx = encoder.encode_images(imgs)
    c_img = cos(i_onnx, emb[idx])
    print(f"imagem cosseno medio: {c_img:.4f}")

    # --- o que importa de verdade: o ranking muda? ---
    piores = []
    for q, qt, qo in zip(QUERIES, t_torch, t_onnx):
        top_t = np.argsort(-(emb @ qt))[:10]
        top_o = np.argsort(-(emb @ qo))[:10]
        overlap = len(set(top_t.tolist()) & set(top_o.tolist()))
        piores.append(overlap)
        print(f"  '{q}': {overlap}/10 iguais no top-10")

    ok = c_txt >= 0.99 and c_img >= 0.95 and min(piores) >= 8
    print("\nVERIFY", "OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
