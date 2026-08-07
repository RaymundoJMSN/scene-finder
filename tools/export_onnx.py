"""Exporta o SigLIP2 para ONNX int8 (roda so na maquina de dev).

Por que SigLIP2 e nao CLIP: medido em tools/bench_modelos.py sobre o acervo real,
acerto@1 subiu de 15% para 28% e o MRR +49%, mantendo o portugues (o text encoder
do SigLIP2 e multilingue).

Saida em models/: image.onnx, text.onnx, tokenizer.json, preprocess.json
"""
import json
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
MODEL_ID = "google/siglip2-base-patch16-224"
MAX_LEN = 64


def _saida(v):
    if hasattr(v, "pooler_output"):
        return v.pooler_output
    if hasattr(v, "last_hidden_state"):
        return v.last_hidden_state.mean(1)
    return v


class Imagem(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return _saida(self.m.get_image_features(pixel_values=pixel_values))


class Texto(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return _saida(self.m.get_text_features(input_ids=input_ids))


def _quant(raw, final):
    quantize_dynamic(raw, final, weight_type=QuantType.QInt8)
    raw.unlink(missing_ok=True)
    for extra in raw.parent.glob(raw.name + ".data"):
        extra.unlink(missing_ok=True)
    return final.stat().st_size / 1e6


def main():
    OUT.mkdir(exist_ok=True)
    for velho in ("clip_image.onnx", "clip_text.onnx", "image_preprocess.json",
                  "text_config.json"):
        (OUT / velho).unlink(missing_ok=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    modelo = AutoModel.from_pretrained(MODEL_ID).eval()

    # IMAGEM FICA EM FP32 DE PROPOSITO.
    # Quantizar este ViT destroi o encoder: cosseno contra o torch cai para 0.72
    # (testado int8 simetrico, uint8 e per-channel: todos iguais) e a busca fica
    # PIOR que o CLIP antigo - MRR 0.169 contra 0.258. Em fp32 o cosseno e 0.997
    # e o custo e so ~30% de tempo por imagem. Nao "otimize" isto sem rodar
    # tools/bench_modelos.py antes.
    print("exportando imagem (fp32, ver comentario)...")
    img_path = OUT / "image.onnx"
    torch.onnx.export(
        Imagem(modelo).eval(), (torch.randn(1, 3, 224, 224),), str(img_path),
        input_names=["pixel_values"], output_names=["embedding"],
        opset_version=17, dynamo=False,
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}})
    print(f"  image.onnx  {img_path.stat().st_size / 1e6:.0f} MB")

    # TEXTO TAMBEM EM FP32.
    # Quantizado ele fica em 0.954 de cosseno, o que parece pouco dano, mas
    # medido no acervo derruba o MRR de 0.443 para 0.382 (-14%). O projeto
    # prioriza qualidade de busca sobre tamanho de download.
    print("exportando texto (fp32, ver comentario)...")
    txt_path = OUT / "text.onnx"
    torch.onnx.export(
        Texto(modelo).eval(), (torch.ones(1, MAX_LEN, dtype=torch.long),),
        str(txt_path),
        input_names=["input_ids"], output_names=["embedding"],
        opset_version=17, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch"}, "embedding": {0: "batch"}})
    print(f"  text.onnx   {txt_path.stat().st_size / 1e6:.0f} MB")

    proc.tokenizer.backend_tokenizer.save(str(OUT / "tokenizer.json"))
    ip = proc.image_processor
    cfg = {
        "size": 224,
        # SigLIP redimensiona direto para o quadrado: nao ha center crop
        "mean": list(ip.image_mean),
        "std": list(ip.image_std),
        "max_len": MAX_LEN,
        "pad_id": proc.tokenizer.pad_token_id,
        "dim": modelo.config.text_config.hidden_size,
        "model_id": MODEL_ID,
    }
    (OUT / "preprocess.json").write_text(json.dumps(cfg, indent=2), "utf-8")
    print("  tokenizer.json + preprocess.json ok")
    print(f"total models/: {sum(f.stat().st_size for f in OUT.iterdir()) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
