"""Inferencia SigLIP2 via onnxruntime - o app nao carrega torch.

Modelo exportado por tools/export_onnx.py. Trocamos o CLIP pelo SigLIP2 depois de
medir no acervo real (tools/bench_modelos.py): acerto@1 15% -> 28%, MRR +49%.
"""
import json
import os
import sys
import threading
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

# O DirectML NAO e thread-safe: duas chamadas a Run() ao mesmo tempo derrubam o
# processo com ACCESS_VIOLATION (0xC0000005), e ate sessoes DIFERENTES (imagem e
# texto) colidem entre si - as vezes chegando a resetar o driver de video. Isso
# acontece pelo caminho normal do app: basta pesquisar enquanto a indexacao roda.
# Por isso o lock e GLOBAL, nao por sessao. No CPU seria dispensavel, mas custa
# quase nada perto do tempo de inferencia.
_GPU_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def config():
    return json.loads((MODELS / "preprocess.json").read_text("utf-8"))


def dim():
    return config()["dim"]


def _session(nome):
    if nome not in _cache:
        # DirectML usa qualquer GPU DX12 (AMD/NVIDIA/Intel). Medido na RX 9070 XT:
        # inferencia ~33x mais rapida que CPU e embeddings identicos (cosseno
        # 0.9999999999). O filtro evita o aviso do ORT em builds sem DML.
        provedores = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                      if p in ort.get_available_providers()]
        _cache[nome] = ort.InferenceSession(
            str(MODELS / nome), _SESS_OPTS,
            providers=provedores or ["CPUExecutionProvider"])
    return _cache[nome]


def provedor_ativo():
    """Provider que sera usado. Nao carrega modelo so para responder isto."""
    for nome, sess in _cache.items():
        if isinstance(nome, str) and nome.endswith(".onnx"):
            return sess.get_providers()[0]
    disp = ort.get_available_providers()
    return "DmlExecutionProvider" if "DmlExecutionProvider" in disp else disp[0]


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


def _rodar(sess, entradas):
    """Toda inferencia passa por aqui - ver o comentario do _GPU_LOCK."""
    with _GPU_LOCK:
        return sess.run(None, entradas)[_saida(sess)]


def encode_images(images, batch_size=16):
    sess = _session("image.onnx")
    saida = []
    for i in range(0, len(images), batch_size):
        lote = np.stack([_preprocess(im) for im in images[i:i + batch_size]])
        saida.append(_rodar(sess, {"pixel_values": lote}))
    return _norm(np.vstack(saida).astype(np.float32))


def encode_prepared(lote):
    """Igual a encode_images, mas recebe o tensor (N,3,224,224) ja pronto.

    Permite que a indexacao prepare as imagens em varias threads enquanto o
    ONNX infere, em vez de decodificar e inferir alternadamente numa thread so.
    """
    if not len(lote):
        return np.zeros((0, dim()), np.float32)
    v = _rodar(_session("image.onnx"),
               {"pixel_values": np.ascontiguousarray(lote)})
    return _norm(v.astype(np.float32))


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
        saida.append(_rodar(sess, {"input_ids": ids}))
    return _norm(np.vstack(saida).astype(np.float32))
