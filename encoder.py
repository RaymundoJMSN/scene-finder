"""Inferencia CLIP via onnxruntime - substitui torch + transformers no app.

Mesmos pesos do sentence-transformers, so que exportados (tools/export_onnx.py).
Sem isto o app empacotado passaria de 800 MB e demoraria ~10 s para abrir.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer


def frozen():
    return getattr(sys, "frozen", False)


def app_dir():
    """Onde ficam modelos e html. Empacotado, e o _MEIPASS (subpasta _internal
    no modo onedir) - NAO a pasta do exe."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def data_dir():
    """Onde ficam indice, thumbs e config. Instalado em Program Files, o app nao
    pode escrever ao lado do exe - vai para LOCALAPPDATA."""
    if frozen():
        d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SceneFinder"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return app_dir()


MODELS = app_dir() / "models"
_SESS_OPTS = ort.SessionOptions()
_SESS_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_cache = {}


def _session(name):
    if name not in _cache:
        _cache[name] = ort.InferenceSession(
            str(MODELS / name), _SESS_OPTS, providers=["CPUExecutionProvider"])
    return _cache[name]


def _emb_out(sess):
    """Indice da saida (batch, 512): o grafo exportado tambem expoe estados
    intermediarios, e a ordem deles nao e estavel entre exports."""
    key = id(sess)
    if key not in _cache:
        for i, o in enumerate(sess.get_outputs()):
            if len(o.shape) == 2 and o.shape[-1] == 512:
                _cache[key] = i
                break
        else:
            raise RuntimeError(
                f"nenhuma saida (batch,512) em {[o.shape for o in sess.get_outputs()]}")
    return _cache[key]


def _norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


# ---------- imagem ----------

def _preprocess(img, cfg):
    """Resize do lado menor -> center crop -> normalize (igual CLIPImageProcessor)."""
    size = cfg["size"]
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(cfg["mean"], dtype=np.float32)) / np.array(cfg["std"], dtype=np.float32)
    return a.transpose(2, 0, 1)  # HWC -> CHW


def encode_images(images, batch_size=16):
    cfg = json.loads((MODELS / "image_preprocess.json").read_text("utf-8"))
    sess = _session("clip_image.onnx")
    out = []
    for i in range(0, len(images), batch_size):
        batch = np.stack([_preprocess(im, cfg) for im in images[i:i + batch_size]])
        out.append(sess.run(None, {"pixel_values": batch})[_emb_out(sess)])
    return _norm(np.vstack(out).astype(np.float32))


# ---------- texto ----------

def _tokenizer():
    if "tok" not in _cache:
        t = Tokenizer.from_file(str(MODELS / "tokenizer.json"))
        max_seq = json.loads((MODELS / "text_config.json").read_text("utf-8"))["max_seq"]
        t.enable_truncation(max_length=max_seq)
        t.enable_padding()
        _cache["tok"] = t
    return _cache["tok"]


def encode_texts(texts, batch_size=64):
    tok = _tokenizer()
    sess = _session("clip_text.onnx")
    out = []
    for i in range(0, len(texts), batch_size):
        encs = tok.encode_batch(texts[i:i + batch_size])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        out.append(sess.run(None, {"input_ids": ids,
                                   "attention_mask": mask})[_emb_out(sess)])
    return _norm(np.vstack(out).astype(np.float32))
