"""CLAP em runtime: acha o som pelo CONTEUDO ("chuva" -> um .wav sem nome util).

Modelos exportados por tools/export_clap.py e validados por tools/verify_clap.py
(cosseno 1.000000 vs original, acerto@5 8/8 no retrieval com efeitos reais).

Roda em CPU DE PROPOSITO: medido, o DirectML da ganho marginal no encoder de
audio (166->149 ms) e PIORA o de texto (3->9 ms) - o HTSAT e pequeno. Em CPU as
sessoes sao thread-safe, entao nao passa pelo _GPU_LOCK e nao disputa a GPU com
o SigLIP2 das imagens.
"""
from pathlib import Path

import numpy as np

import encoder

MODELS = encoder.MODELS
SR = 48000
N_FFT = 1024
HOP = 480
N_MELS = 64
FMIN, FMAX = 50.0, 14000.0
W = SR * 10                      # 480000 amostras = 10 s
FRACOES = (0.10, 0.50, 0.90)     # inicio das 3 janelas em audio longo

_cache = {}


def disponivel():
    return (MODELS / "clap_audio.onnx").exists() and \
           (MODELS / "clap_text.onnx").exists()


def dim():
    return 512


def _sess(nome):
    if nome not in _cache:
        import onnxruntime as ort
        _cache[nome] = ort.InferenceSession(
            str(MODELS / nome), providers=["CPUExecutionProvider"])
    return _cache[nome]


def _tok():
    if "tok" not in _cache:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(str(MODELS / "clap_tokenizer.json"))
        t.enable_padding(pad_id=1, pad_token="<pad>")
        t.enable_truncation(max_length=64)
        _cache["tok"] = t
    return _cache["tok"]


# ---------- replica numpy do ClapFeatureExtractor (diff max 7.6e-6) ----------

def _hz_para_mel(freq):
    freq = np.asarray(freq, dtype=np.float64)
    return np.where(freq >= 1000.0,
                    15.0 + np.log(np.maximum(freq, 1e-10) / 1000.0)
                    * (27.0 / np.log(6.4)),
                    3.0 * freq / 200.0)


def _mel_para_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    return np.where(mels >= 15.0,
                    1000.0 * np.exp((np.log(6.4) / 27.0) * (mels - 15.0)),
                    200.0 * mels / 3.0)


def _filtro_mel():
    if "fb" not in _cache:
        nb = N_FFT // 2 + 1
        mel_freqs = np.linspace(_hz_para_mel(FMIN), _hz_para_mel(FMAX), N_MELS + 2)
        ff = _mel_para_hz(mel_freqs)
        fft_freqs = np.linspace(0, SR // 2, nb)
        slopes = ff[None, :] - fft_freqs[:, None]
        diff = np.diff(ff)
        fb = np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1],
                                        slopes[:, 2:] / diff[1:]))
        fb *= (2.0 / (ff[2:N_MELS + 2] - ff[:N_MELS]))[None, :]
        _cache["fb"] = fb
        _cache["janela"] = np.hanning(N_FFT + 1)[:-1]
    return _cache["fb"], _cache["janela"]


def _log_mel(wav):
    fb, janela = _filtro_mel()
    wav = np.asarray(wav, dtype=np.float64)
    if wav.shape[0] < W:                        # repeatpad do transformers
        wav = np.tile(wav, W // wav.shape[0])
        wav = np.pad(wav, (0, W - wav.shape[0]))
    wav = np.pad(wav, (N_FFT // 2, N_FFT // 2), mode="reflect")
    quadros = np.lib.stride_tricks.sliding_window_view(wav, N_FFT)[::HOP]
    pot = np.abs(np.fft.rfft(quadros * janela, axis=1)) ** 2.0
    mel = np.maximum(1e-10, pot @ fb)
    return (10.0 * np.log10(mel)).astype(np.float32)


def _resample(wav, sr):
    if sr == SR:
        return wav
    import soxr
    # interp linear numpy foi REPROVADA na verificacao (cos min 0.838)
    return soxr.resample(wav, sr, SR, quality="HQ").astype(np.float32)


def _preparar(caminho):
    """Arquivo -> (n_janelas, 1, 1001, 64). Audio longo: 3 janelas via seek."""
    import soundfile as sf
    info = sf.info(str(caminho))
    w_orig = info.samplerate * 10
    if info.frames <= w_orig:
        dados, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
        janelas = [_resample(dados.mean(axis=1), sr)[:W]]
    else:
        janelas = []
        with sf.SoundFile(str(caminho)) as f:
            for frac in FRACOES:
                f.seek(int((info.frames - w_orig) * frac))
                trecho = f.read(w_orig + N_FFT, dtype="float32", always_2d=True)
                wav = _resample(trecho.mean(axis=1), info.samplerate)
                if wav.shape[0] < W:
                    wav = np.pad(wav, (0, W - wav.shape[0]))
                janelas.append(wav[:W])
    return np.stack([_log_mel(j) for j in janelas])[:, None]


def encode_audio(caminho):
    """-> embedding (512,) L2-normalizado (media das janelas)."""
    feats = _preparar(caminho)
    e = _sess("clap_audio.onnx").run(None, {"input_features": feats})[0]
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)
    e = e.mean(axis=0)
    return (e / max(float(np.linalg.norm(e)), 1e-12)).astype(np.float32)


def encode_text(texto):
    tok = _tok()
    enc = tok.encode_batch([texto])
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    e = _sess("clap_text.onnx").run(
        None, {"input_ids": ids, "attention_mask": mask})[0][0]
    return (e / max(float(np.linalg.norm(e)), 1e-12)).astype(np.float32)
