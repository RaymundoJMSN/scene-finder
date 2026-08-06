"""Converte os 2 modelos CLIP para ONNX int8 (roda so na maquina de dev).

Motivo: torch + modelos = ~1,5 GB no instalador e ~10 s de startup. ONNX int8
da ~250 MB e ~1 s, e o PyInstaller para de brigar com o torch.

Saida em models/: clip_image.onnx, clip_text.onnx, tokenizer.json, dense.npy
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
IMG_MODEL = "clip-ViT-B-32"
TXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


class ImageEncoder(torch.nn.Module):
    """pixel_values -> embedding 512d (nao normalizado)."""

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values):
        return self.clip.get_image_features(pixel_values=pixel_values)


class TextEncoder(torch.nn.Module):
    """input_ids/attention_mask -> mean pooling -> Dense 768->512."""

    def __init__(self, xlmr, dense):
        super().__init__()
        self.xlmr = xlmr
        self.dense = dense

    def forward(self, input_ids, attention_mask):
        tok = self.xlmr(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(tok.dtype)
        mean = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.dense(mean)


def _quant(raw, final):
    quantize_dynamic(raw, final, weight_type=QuantType.QInt8)
    raw.unlink()
    return final.stat().st_size / 1e6


def export_image():
    st = SentenceTransformer(IMG_MODEL)
    clip = st[0].model if hasattr(st[0], "model") else st[0].auto_model
    enc = ImageEncoder(clip).eval()
    dummy = torch.randn(1, 3, 224, 224)
    raw = OUT / "clip_image_fp32.onnx"
    torch.onnx.export(
        enc, (dummy,), str(raw), input_names=["pixel_values"],
        output_names=["embedding"], opset_version=17, dynamo=False,
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}})
    mb = _quant(raw, OUT / "clip_image.onnx")
    print(f"  clip_image.onnx  {mb:.0f} MB")

    # config do preprocessamento, para nao depender do transformers em runtime
    proc = st[0].processor if hasattr(st[0], "processor") else None
    ip = getattr(proc, "image_processor", proc)
    cfg = {
        "size": 224,
        "mean": list(getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073])),
        "std": list(getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711])),
    }
    (OUT / "image_preprocess.json").write_text(json.dumps(cfg), "utf-8")
    return st


def export_text():
    st = SentenceTransformer(TXT_MODEL)
    xlmr = st[0].auto_model
    # o modulo Dense recebe/devolve dict; para exportar so interessa a matriz
    dense = st[2].linear if hasattr(st[2], "linear") else st[2]
    enc = TextEncoder(xlmr, dense).eval()
    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    raw = OUT / "clip_text_fp32.onnx"
    torch.onnx.export(
        enc, (ids, mask), str(raw),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        opset_version=17, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "embedding": {0: "batch"}})
    mb = _quant(raw, OUT / "clip_text.onnx")
    print(f"  clip_text.onnx   {mb:.0f} MB")

    tok_dir = Path(st[0].tokenizer.name_or_path)
    src = tok_dir / "tokenizer.json"
    if src.exists():
        shutil.copy(src, OUT / "tokenizer.json")
    else:
        st[0].tokenizer.backend_tokenizer.save(str(OUT / "tokenizer.json"))
    print(f"  tokenizer.json   ok (max_seq={st.max_seq_length})")
    (OUT / "text_config.json").write_text(
        json.dumps({"max_seq": st.max_seq_length}), "utf-8")
    return st


if __name__ == "__main__":
    print("exportando imagem...")
    st_img = export_image()
    print("exportando texto...")
    st_txt = export_text()
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"total models/: {total:.0f} MB")
