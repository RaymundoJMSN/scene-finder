"""Inferencia SigLIP2 via onnxruntime - o app nao carrega torch.

Modelo exportado por tools/export_onnx.py. Trocamos o CLIP pelo SigLIP2 depois de
medir no acervo real (tools/bench_modelos.py): acerto@1 15% -> 28%, MRR +49%.
"""
import json
import os
import sys
from functools import lru_cache
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


@lru_cache(maxsize=1)
def config():
    return json.loads((MODELS / "preprocess.json").read_text("utf-8"))


def dim():
    return config()["dim"]


def _session(nome):
    if nome not in _cache:
        _cache[nome] = ort.InferenceSession(
            str(MODELS / nome), _SESS_OPTS, providers=["CPUExecutionProvider"])
    return _cache[nome]


def _saida(sess):
    """Indice da saida (batch, dim): o grafo exportado tambem expoe estados
    intermediarios, e a ordem deles nao e estavel entre exports."""
    chave = ("out", id(sess))
    if chave not in _cache:
        d = dim()
        for i, o in enumerate(sess.get_outputs()):
            if len(o.shape) == 2 and o.shape[-1] == d:
                _cache[chave] = i
                break
        else:
            raise RuntimeError(
                f"nenhuma saida (batch,{d}) em {[o.shape for o in sess.get_outputs()]}")
    return _cache[chave]


def _norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


# ---------- imagem ----------

def _preprocess(img):
    """SigLIP: redimensiona direto para o quadrado (sem center crop) e normaliza."""
    cfg = config()
    s = cfg["size"]
    img = img.convert("RGB").resize((s, s), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return a.transpose(2, 0, 1)  # HWC -> CHW


def encode_images(images, batch_size=16):
    sess = _session("image.onnx")
    saida = []
    for i in range(0, len(images), batch_size):
        lote = np.stack([_preprocess(im) for im in images[i:i + batch_size]])
        saida.append(sess.run(None, {"pixel_values": lote})[_saida(sess)])
    return _norm(np.vstack(saida).astype(np.float32))


# ---------- texto ----------

def _tokenizer():
    if "tok" not in _cache:
        cfg = config()
        t = Tokenizer.from_file(str(MODELS / "tokenizer.json"))
        t.enable_truncation(max_length=cfg["max_len"])
        # SigLIP espera comprimento fixo: sem attention_mask, o padding faz parte
        t.enable_padding(length=cfg["max_len"], pad_id=cfg["pad_id"], pad_token="<pad>")
        _cache["tok"] = t
    return _cache["tok"]


def encode_texts(texts, batch_size=64):
    tok = _tokenizer()
    sess = _session("text.onnx")
    saida = []
    for i in range(0, len(texts), batch_size):
        encs = tok.encode_batch(texts[i:i + batch_size])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        saida.append(sess.run(None, {"input_ids": ids})[_saida(sess)])
    return _norm(np.vstack(saida).astype(np.float32))
