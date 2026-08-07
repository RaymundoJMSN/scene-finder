"""Confere que o ONNX int8 reproduz o modelo original (torch) antes de reindexar.

Uma diferenca aqui nao aparece como erro: aparece como busca pior, semanas depois.
Criterio: cosseno medio >= 0.95 nos dois encoders e ordenacao preservada num
conjunto pequeno de consultas.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

import encoder  # noqa: E402

CONSULTAS = ["tavern at night", "taverna a noite", "jungle temple",
             "pirate ship deck", "desert city", "biblioteca real"]


def main():
    import torch
    from transformers import AutoModel, AutoProcessor

    mid = encoder.config()["model_id"]
    proc = AutoProcessor.from_pretrained(mid)
    mod = AutoModel.from_pretrained(mid).eval()

    def puro(v):
        return v.pooler_output if hasattr(v, "pooler_output") else v

    def norm(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    # texto
    x = proc(text=CONSULTAS, return_tensors="pt", padding="max_length",
             max_length=encoder.config()["max_len"], truncation=True)
    with torch.no_grad():
        t_torch = norm(puro(mod.get_text_features(input_ids=x["input_ids"])).numpy())
    t_onnx = encoder.encode_texts(CONSULTAS)
    c_txt = float(np.sum(t_onnx * t_torch, axis=-1).mean())
    print(f"texto  cosseno medio: {c_txt:.4f}")

    # imagens REAIS do acervo: imagem sintetica fica fora da distribuicao do
    # modelo e a divergencia medida nela nao diz nada sobre o uso real
    import indexer
    caminhos = indexer.scan(indexer.load_config())
    if len(caminhos) < 12:
        print("acervo pequeno demais para verificar")
        return 1
    passo = max(1, len(caminhos) // 12)
    imgs = [Image.open(p).convert("RGB") for p in caminhos[::passo][:12]]
    px = proc(images=imgs, return_tensors="pt")
    with torch.no_grad():
        i_torch = norm(puro(mod.get_image_features(**px)).numpy())
    i_onnx = encoder.encode_images(imgs)
    c_img = float(np.sum(i_onnx * i_torch, axis=-1).mean())
    print(f"imagem cosseno medio: {c_img:.4f}")

    # a ordenacao entre torch e onnx precisa bater
    iguais = 0
    for q_o, q_t in zip(t_onnx, t_torch):
        if np.argmax(i_onnx @ q_o) == np.argmax(i_torch @ q_t):
            iguais += 1
    print(f"melhor imagem igual em {iguais}/{len(CONSULTAS)} consultas")

    ok = c_txt >= 0.95 and c_img >= 0.95 and iguais >= len(CONSULTAS) - 2
    print("\nVERIFY", "OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
