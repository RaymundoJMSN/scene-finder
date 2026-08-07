"""Verifica o CLAP ONNX exportado (tools/export_clap.py) SEM torch/transformers.

Roda no venv-build:  .\\venv-build\\Scripts\\python.exe tools\\verify_clap.py

Checa tres coisas:
  a) paridade ONNX vs torch, contra a referencia congelada em models/clap_ref.npz
     (gerada pelo export no venv dev): cosseno >= 0.999 em 12 audios reais e
     12 textos, e espectrograma da replica numpy vs ClapFeatureExtractor
     (diff max <= 1e-4);
  b) retrieval real: 8 consultas x ~40 efeitos do acervo, acerto@5 pelo AUDIO
     (o nome do arquivo e so gabarito, nao entra no ranking);
  c) tempo por audio no CPU e no DML (se DmlExecutionProvider existir).

Receita de runtime (o que o app vai replicar — parametros em clap_config.json):
  soundfile le (ogg/wav/flac/mp3/opus; m4a NAO) -> mono (media dos canais) ->
  soxr para 48 kHz (interp linear numpy REPROVADA: cos medio 0.956, min 0.838)
  -> ate 3 janelas de 10 s (10%/50%/90%; MRR 0.917 vs 0.854 da janela central)
  -> log-mel 1001x64 (replica numpy do ClapFeatureExtractor, diff ~4e-6) ->
  clap_audio.onnx -> embeddings das janelas L2-norm, media, re-norm.
  Texto: clap_tokenizer.json (tokenizers) -> clap_text.onnx. Saidas ja saem
  L2-normalizadas (norma 1), busca = produto escalar.

ATENCAO DirectML: nao e thread-safe — no app, toda chamada ao session.run
passa pelo encoder._GPU_LOCK, igual aos encoders de imagem/texto.
"""
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
AUDIO_DIR = Path(r"X:\FoundryVTT\Data\Assets\Audio")

# ---- parametros do ClapFeatureExtractor (laion/clap-htsat-unfused) ----
SR = 48000
N_FFT = 1024
HOP = 480
N_MELS = 64
FMIN = 50.0
FMAX = 14000.0
W = SR * 10                      # 480000 amostras = 10 s
FRACOES = (0.10, 0.50, 0.90)     # inicio das 3 janelas em audio longo


def _hz_para_mel_slaney(freq):
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    return np.where(freq >= 1000.0,
                    15.0 + np.log(np.maximum(freq, 1e-10) / 1000.0) * (27.0 / np.log(6.4)),
                    mels)


def _mel_para_hz_slaney(mels):
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    return np.where(mels >= 15.0,
                    1000.0 * np.exp((np.log(6.4) / 27.0) * (mels - 15.0)),
                    freq)


def _filtro_mel():
    """(513, 64) igual a transformers.mel_filter_bank(norm='slaney', mel_scale='slaney')."""
    nb_bins = N_FFT // 2 + 1
    mel_freqs = np.linspace(_hz_para_mel_slaney(FMIN), _hz_para_mel_slaney(FMAX), N_MELS + 2)
    filter_freqs = _mel_para_hz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, SR // 2, nb_bins)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    diff = np.diff(filter_freqs)
    fb = np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]))
    fb *= (2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS]))[None, :]
    return fb


_FB = _filtro_mel()
_JANELA = np.hanning(N_FFT + 1)[:-1]  # hann periodica, igual window_function(1024, "hann")


def log_mel(waveform):
    """Waveform mono 48 kHz de ate 480000 amostras -> (1001, 64) float32 em dB.

    Replica bit-a-bit o ClapFeatureExtractor (rand_trunc + repeatpad, filtros
    slaney/slaney): diff max medido 3.8e-6 contra o transformers.
    """
    wav = np.asarray(waveform, dtype=np.float64)
    if wav.shape[0] > W:
        raise ValueError("log_mel recebe janelas de ate 10 s; use preparar_audio")
    if wav.shape[0] < W:                       # repeatpad do transformers
        wav = np.tile(wav, W // wav.shape[0])
        wav = np.pad(wav, (0, W - wav.shape[0]))
    wav = np.pad(wav, (N_FFT // 2, N_FFT // 2), mode="reflect")
    quadros = np.lib.stride_tricks.sliding_window_view(wav, N_FFT)[::HOP]  # (1001, 1024)
    potencia = np.abs(np.fft.rfft(quadros * _JANELA, axis=1)) ** 2.0
    mel = np.maximum(1e-10, potencia @ _FB)
    return (10.0 * np.log10(mel)).astype(np.float32)  # power_to_db ref=1, sem top_db


def _resample(wav, sr):
    if sr == SR:
        return wav
    return soxr.resample(wav, sr, SR, quality="HQ").astype(np.float32)


def carregar_48k(caminho):
    """Le o arquivo INTEIRO em mono 48 kHz (so para audio curto ou referencia)."""
    dados, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
    return _resample(dados.mean(axis=1), sr)


def preparar_audio(caminho):
    """Arquivo de audio -> input_features (n_janelas, 1, 1001, 64) float32.

    Audio <= 10 s: 1 janela (repeatpad). Audio > 10 s: 3 janelas de 10 s
    comecando em 10%/50%/90% do excedente — le SO esses trechos via seek
    (musica de 1 h nao cabe decodificada inteira na RAM).
    """
    info = sf.info(str(caminho))
    w_orig = info.samplerate * 10
    if info.frames <= w_orig:
        janelas = [carregar_48k(caminho)]
    else:
        janelas = []
        with sf.SoundFile(str(caminho)) as f:
            for frac in FRACOES:
                f.seek(int((info.frames - w_orig) * frac))
                trecho = f.read(w_orig + N_FFT, dtype="float32", always_2d=True)
                wav = _resample(trecho.mean(axis=1), info.samplerate)
                if wav.shape[0] < W:           # borda de arredondamento do resample
                    wav = np.pad(wav, (0, W - wav.shape[0]))
                janelas.append(wav[:W])
    return np.stack([log_mel(j) for j in janelas])[:, None]


def embed_janelas(sess, feats):
    """(n, 1, 1001, 64) -> embedding (512,) L2-normalizado (media das janelas)."""
    e = sess.run(None, {"input_features": feats})[0]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    e = e.mean(axis=0)
    return e / np.linalg.norm(e)


def embed_audio(sess, caminho):
    return embed_janelas(sess, preparar_audio(caminho))


def abrir_tokenizer():
    tok = Tokenizer.from_file(str(MODELS / "clap_tokenizer.json"))
    tok.enable_padding(pad_id=1, pad_token="<pad>")
    tok.enable_truncation(max_length=64)
    return tok


def tokenizar(tok, frases):
    enc = tok.encode_batch(list(frases))
    return (np.array([e.ids for e in enc], dtype=np.int64),
            np.array([e.attention_mask for e in enc], dtype=np.int64))


def embed_textos(sess, tok, frases):
    ids, mask = tokenizar(tok, frases)
    e = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# teste de retrieval: 8 consultas x ~40 efeitos reais do acervo
# ---------------------------------------------------------------------------
GABARITO = {  # consulta em ingles -> regex de nome que conta como acerto
    "rain": "rain",
    "sword battle": "sword|battle",
    "tavern crowd": "tavern",
    "fire": "fire",
    "thunder": "thunder",
    "footsteps": "footsteps",
    "door": "door",
    "wind": "wind",
}

ARQUIVOS_RETRIEVAL = [
    # gabaritos (nome contem o termo)
    "01-rain-urban-gentle-1-ms-stereo.ogg",
    "1H_Sword_Hit_Flesh_01.ogg",
    "knife-sword-hit-multiple-fienup-003.ogg",
    "battle-cry-charge-screams-army-attacking-running-1.ogg",
    "boardwalk-crowd-passing-through-tavern-featuring-traditional-greek-music-warm-day.ogg",
    "campfire.ogg",
    "firecracklinginawoodstove.ogg",
    "ambient-thunder-clap-distant-with-rain-mono.ogg",
    "185-soundscape-mountain-forest-thunderstorm-close-birds-distant-summer.ogg",
    "0002-footsteps-walking-crusty-snow-male-shoes-normal-pace.ogg",
    "015-foley-footsteps-asphalt-boot-walk-fast-run-jog-close.ogg",
    "1707-footsteps-flip-flops-140-fpm-loop.ogg",
    "0002-door-thick-open-close-squeaky.ogg",
    "03-wind-grass-strong-hurricane-blast-3.ogg",
    "0004-wind-gusty-buffeted-mountainside-birds.ogg",
    # distratores
    "0001-pencil-draw-circles.ogg",
    "0002-ambience-forest-crows-birds-chirping.ogg",
    "0003-buckle-snap-and-unsnap-2.ogg",
    "0004-bicycle-pedaling-on-dirt.ogg",
    "0005-hammer-on-wood.ogg",
    "0005-water-river-medium-flow-medium-distance.ogg",
    "0006-glass-jar-medium-size-metal-lid-screw-on-off-1.ogg",
    "0010-zipper-backpack-variation.ogg",
    "0021-tape-measure-2.ogg",
    "0024-typing-keyboard-variation.ogg",
    "0019-shotgun-pump-action-1.ogg",
    "0023-shotgun-safety-switch-on-off-2.ogg",
    "05-crickets-owls-summer-night.ogg",
    "206700-railroad-crossing-signal-bell-loop.ogg",
    "08-small-beach-gentle-waves-echo-1.ogg",
    "0026-pouring-liquid-in-to-bowl-5.ogg",
    "0094-pour-cereal-into-glass-bowl-3.ogg",
    "0014-water-large-stream-1ft-waterfall-base.ogg",
    "0008-water-small-drainpipe-close-to-opening.ogg",
    "0010-reciprocating-saw.ogg",
    "0008-miter-chop-saw-2.ogg",
    "0007-snowboard-passby-5.ogg",
    "02-water-mill-wheel-outdoor-mid-field-stereo.ogg",
    "13-barcelona-metro-platform-crowd-walla-paannounce-bleep.ogg",
    "35-barcelona-burgerbar-manytourists-crowd-walla.ogg",
]


def _checar_paridade(sess_a, sess_t, tok):
    ref = np.load(MODELS / "clap_ref.npz")

    diff_mel = np.abs(log_mel(ref["mel_wav"]) - ref["mel_ref"]).max()
    print(f"a1) espectrograma numpy vs ClapFeatureExtractor: diff max {diff_mel:.2e}")
    assert diff_mel <= 1e-4, "espectrograma fora da tolerancia"

    cos_a = []
    for caminho, e_ref in zip(ref["audio_paths"], ref["audio_embs"]):
        e = embed_audio(sess_a, caminho)
        cos_a.append(float(e @ e_ref / np.linalg.norm(e_ref)))
    print(f"a2) audio ONNX vs torch ({len(cos_a)} arquivos): "
          f"cos min {min(cos_a):.6f} medio {np.mean(cos_a):.6f}")
    assert min(cos_a) >= 0.999, "audio abaixo de 0.999"

    E = embed_textos(sess_t, tok, ref["textos"])
    e_ref = ref["text_embs"] / np.linalg.norm(ref["text_embs"], axis=1, keepdims=True)
    cos_t = (E * e_ref).sum(axis=1)
    print(f"a3) texto ONNX vs torch ({len(cos_t)} frases): "
          f"cos min {cos_t.min():.6f} medio {cos_t.mean():.6f}")
    assert cos_t.min() >= 0.999, "texto abaixo de 0.999"


def _checar_retrieval(sess_a, sess_t, tok):
    import re
    nomes, embs = [], []
    for nome in ARQUIVOS_RETRIEVAL:
        p = AUDIO_DIR / nome
        if not p.exists():
            print(f"   (pulado, nao existe: {nome})")
            continue
        nomes.append(nome)
        embs.append(embed_audio(sess_a, p))
    E = np.stack(embs)
    T = embed_textos(sess_t, tok, list(GABARITO))
    sims = T @ E.T

    acertos = 0
    for qi, (consulta, rx) in enumerate(GABARITO.items()):
        top5 = [nomes[i] for i in np.argsort(-sims[qi])[:5]]
        ok = any(re.search(rx, n, re.I) for n in top5)
        acertos += ok
        print(f"   {'OK' if ok else 'X '} {consulta:14s} top1: {top5[0][:60]}")
    print(f"b) acerto@5: {acertos}/{len(GABARITO)} ({len(nomes)} audios no indice)")
    assert acertos >= 6, "retrieval degradou (esperado 8/8 na exportacao original)"


def _medir_tempo(providers, rotulo, tok):
    import onnxruntime as ort
    sess_a = ort.InferenceSession(str(MODELS / "clap_audio.onnx"), providers=providers)
    sess_t = ort.InferenceSession(str(MODELS / "clap_text.onnx"), providers=providers)
    amostra = [AUDIO_DIR / n for n in ARQUIVOS_RETRIEVAL[:8] if (AUDIO_DIR / n).exists()]

    feats = [preparar_audio(p) for p in amostra]
    embed_janelas(sess_a, feats[0])                      # aquecimento
    t0 = time.perf_counter()
    for f in feats:
        embed_janelas(sess_a, f)
    dt = (time.perf_counter() - t0) / len(feats)
    n_jan = sum(f.shape[0] for f in feats) / len(feats)

    t0 = time.perf_counter()
    for p in amostra:
        preparar_audio(p)
    dt_prep = (time.perf_counter() - t0) / len(amostra)

    embed_textos(sess_t, tok, ["warmup"])
    t0 = time.perf_counter()
    for _ in range(5):
        embed_textos(sess_t, tok, list(GABARITO))
    dt_txt = (time.perf_counter() - t0) / (5 * len(GABARITO))
    print(f"c) {rotulo}: inferencia {dt*1000:.0f} ms/audio ({n_jan:.1f} janelas em media), "
          f"preparo {dt_prep*1000:.0f} ms/audio, texto {dt_txt*1000:.1f} ms/consulta")


def main():
    import onnxruntime as ort
    sess_a = ort.InferenceSession(str(MODELS / "clap_audio.onnx"),
                                  providers=["CPUExecutionProvider"])
    sess_t = ort.InferenceSession(str(MODELS / "clap_text.onnx"),
                                  providers=["CPUExecutionProvider"])
    tok = abrir_tokenizer()

    _checar_paridade(sess_a, sess_t, tok)
    _checar_retrieval(sess_a, sess_t, tok)
    _medir_tempo(["CPUExecutionProvider"], "CPU", tok)
    if "DmlExecutionProvider" in ort.get_available_providers():
        # single-thread aqui; no app o session.run do DML exige encoder._GPU_LOCK
        _medir_tempo(["DmlExecutionProvider", "CPUExecutionProvider"], "DML", tok)
    print("VERIFICACAO OK")


if __name__ == "__main__":
    main()
