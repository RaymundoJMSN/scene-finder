"""Exporta o CLAP (audio<->texto) para ONNX fp32 - roda so na maquina de dev.

Modelo: laion/clap-htsat-unfused (HTSAT-tiny + RoBERTa-base, embedding 512).
Decisoes MEDIDAS nesta exportacao (nao presumidas):
- fp32 SEM quantizar: int8 ja destruiu o ViT do SigLIP2 neste projeto
  (tools/export_onnx.py); aqui nem foi tentado, qualidade > tamanho.
- variante unfused: no teste de retrieval real (8 consultas x 40 efeitos do
  acervo, verify_clap.py) o modelo base ja faz acerto@5 = 8/8 — o teste satura,
  nao ha o que uma variante maior (larger_clap_general) melhoraria de medivel.
- unfused ignora is_longer (enable_fusion=False), entao o grafo exportado so
  recebe input_features (batch, 1, 1001, 64); audio longo vira 3 janelas de
  10 s em 10%/50%/90% mediadas (MRR 0.917 vs 0.854 da janela central unica).
- resample p/ 48 kHz: soxr HQ (tem wheel abi3 p/ Python 3.14 do venv-build);
  interpolacao linear numpy REPROVADA (cos medio 0.956, min 0.838 vs soxr).

Saida em models/: clap_audio.onnx (~116 MB), clap_text.onnx (~501 MB),
clap_tokenizer.json, clap_config.json e clap_ref.npz (embeddings torch de
12 audios reais + 12 textos + 1 espectrograma de referencia, para o
verify_clap.py provar a paridade no venv-build SEM torch).

Rodar: .\\venv\\Scripts\\python.exe tools\\export_clap.py
Depois: .\\venv-build\\Scripts\\python.exe tools\\verify_clap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import ClapFeatureExtractor, ClapModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
MODEL_ID = "laion/clap-htsat-unfused"
AUDIO_DIR = Path(r"X:\FoundryVTT\Data\Assets\Audio")
MUSICA_DIR = Path(r"X:\FoundryVTT\Data\Assets\Musicas")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 12 audios reais variados (efeitos curtos/longos, ambiencias, musica em wav e
# mp3) para a referencia de paridade torch vs ONNX
AUDIOS_REF = [
    AUDIO_DIR / "01-rain-urban-gentle-1-ms-stereo.ogg",
    AUDIO_DIR / "1H_Sword_Hit_Flesh_01.ogg",
    AUDIO_DIR / "boardwalk-crowd-passing-through-tavern-featuring-traditional-greek-music-warm-day.ogg",
    AUDIO_DIR / "campfire.ogg",
    AUDIO_DIR / "ambient-thunder-clap-distant-with-rain-mono.ogg",
    AUDIO_DIR / "0002-footsteps-walking-crusty-snow-male-shoes-normal-pace.ogg",
    AUDIO_DIR / "0002-door-thick-open-close-squeaky.ogg",
    AUDIO_DIR / "03-wind-grass-strong-hurricane-blast-3.ogg",
    AUDIO_DIR / "0024-typing-keyboard-variation.ogg",
    AUDIO_DIR / "05-crickets-owls-summer-night.ogg",
    MUSICA_DIR / "23716-orchestral-build-up-120b.wav",
    MUSICA_DIR / "52.-escape-｜-rpg-chase-theme-｜-fantasy-music-｜-background-music-｜-dnd.mp3",
]

TEXTOS_REF = [
    "rain", "sword battle", "tavern crowd", "fire", "thunder", "footsteps",
    "door", "wind", "epic orchestral battle music",
    "creaking wooden door opening slowly", "ocean waves crashing on a beach",
    "a blacksmith hammering metal",
]


def _garantir_audios_ref():
    """Sem o acervo local (CI), gera 12 wavs sinteticos variados: a paridade
    torch vs ONNX se prova igualmente bem com qualquer sinal."""
    global AUDIOS_REF
    if AUDIOS_REF[0].exists():
        return
    import math
    import struct
    import tempfile
    import wave
    tmp = Path(tempfile.mkdtemp(prefix="clap-ref-"))
    novos = []
    for k in range(len(AUDIOS_REF)):
        p = tmp / f"ref_{k}.wav"
        freq = 110 * (k + 1)
        dur = 2 + (k % 3) * 6            # 2s a 14s: cobre 1 e 3 janelas
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / 16000)
                                      * (0.5 + 0.5 * math.sin(i / 3000))))
                for i in range(16000 * dur)))
        novos.append(p)
    AUDIOS_REF = novos
    print(f"(acervo local ausente: referencia com {len(novos)} wavs sinteticos)")


class Audio(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_features):
        # pooler_output ja sai L2-normalizado pelo get_audio_features
        return self.m.get_audio_features(input_features=input_features).pooler_output


class Texto(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask).pooler_output


def main():
    _garantir_audios_ref()
    OUT.mkdir(exist_ok=True)
    modelo = ClapModel.from_pretrained(MODEL_ID).eval()
    assert not modelo.config.audio_config.enable_fusion, "esperado unfused"

    # AUDIO E TEXTO EM FP32 DE PROPOSITO — ver docstring; nao quantizar.
    print("exportando audio (fp32)...")
    a_path = OUT / "clap_audio.onnx"
    torch.onnx.export(
        Audio(modelo).eval(), (torch.randn(1, 1, 1001, 64),), str(a_path),
        input_names=["input_features"], output_names=["embedding"],
        opset_version=17, dynamo=False,
        dynamic_axes={"input_features": {0: "batch"}, "embedding": {0: "batch"}})
    print(f"  clap_audio.onnx {a_path.stat().st_size / 1e6:.0f} MB")

    print("exportando texto (fp32)...")
    t_path = OUT / "clap_text.onnx"
    torch.onnx.export(
        Texto(modelo).eval(),
        (torch.ones(2, 12, dtype=torch.long), torch.ones(2, 12, dtype=torch.long)),
        str(t_path),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        opset_version=17, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "embedding": {0: "batch"}})
    print(f"  clap_text.onnx  {t_path.stat().st_size / 1e6:.0f} MB")

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL_ID).backend_tokenizer.save(
        str(OUT / "clap_tokenizer.json"))

    cfg = {
        "model_id": MODEL_ID,
        "dim": 512,
        "sampling_rate": 48000,
        "n_fft": 1024,
        "hop_length": 480,
        "n_mels": 64,
        "frequency_min": 50.0,
        "frequency_max": 14000.0,
        "mel_scale": "slaney",
        "mel_norm": "slaney",
        "janela_stft": "hann periodica: np.hanning(n_fft+1)[:-1]",
        "stft_center": True,
        "stft_pad_mode": "reflect",
        "power": 2.0,
        "log": "10*log10(clip(mel, 1e-10, None))  # power_to_db ref=1, sem top_db",
        "max_samples": 480000,
        "frames": 1001,
        "audio_curto": "repeatpad: tile inteiro + zeros ate 480000 (feito no log_mel)",
        "audio_longo": {
            "janelas": [0.10, 0.50, 0.90],
            "regra": "3 janelas de 10 s comecando nessas fracoes do excedente; "
                     "embeddings L2-norm por janela, media, re-norm",
            "medido": "MRR 0.917 vs 0.854 da janela central unica (acerto@5 8/8 nos dois)",
        },
        "resample": {
            "lib": "soxr", "qualidade": "HQ",
            "motivo": "wheel abi3 funciona no Python 3.14 do venv-build; interp "
                      "linear numpy reprovada (cos medio 0.956, min 0.838 vs soxr)",
        },
        "audio_input": {"nome": "input_features", "shape": ["batch", 1, 1001, 64],
                        "dtype": "float32"},
        "texto_input": {"nomes": ["input_ids", "attention_mask"], "dtype": "int64",
                        "pad_id": 1, "pad_token": "<pad>"},
        "saida": "embedding (batch, 512) ja L2-normalizado; busca = produto escalar",
        "consulta": "em ingles ('rain' e 'the sound of rain' empatam 8/8); "
                    "traduzir pt->en com ptbr.py como nas fontes online",
        "formatos": ["ogg", "wav", "flac", "mp3", "opus"],
        "limitacoes": [
            "m4a nao suportado (libsndfile: Format not recognised)",
            "DirectML nao e thread-safe: session.run sempre sob encoder._GPU_LOCK",
        ],
        "referencia_runtime": "tools/verify_clap.py (preparar_audio/embed_*)",
    }
    (OUT / "clap_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), "utf-8")
    print("  clap_tokenizer.json + clap_config.json ok")

    # ---- referencia congelada p/ o verify rodar sem torch ----
    import verify_clap as vc

    print("gerando clap_ref.npz (embeddings torch de audios/textos reais)...")
    audio_mod, texto_mod = Audio(modelo), Texto(modelo)
    embs_a = []
    for p in AUDIOS_REF:
        feats = vc.preparar_audio(p)                    # mesma receita do runtime
        with torch.inference_mode():
            e = audio_mod(torch.from_numpy(feats)).numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        e = e.mean(axis=0)
        embs_a.append(e / np.linalg.norm(e))

    ids, mask = vc.tokenizar(vc.abrir_tokenizer(), TEXTOS_REF)
    with torch.inference_mode():
        embs_t = texto_mod(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()

    # espectrograma de referencia: 10 s reais pelo ClapFeatureExtractor oficial
    wav = vc.carregar_48k(AUDIOS_REF[0])
    meio = max(0, (wav.shape[0] - vc.W) // 2)
    mel_wav = wav[meio:meio + vc.W]
    fe = ClapFeatureExtractor.from_pretrained(MODEL_ID)
    mel_ref = fe(mel_wav, sampling_rate=48000, return_tensors="np")["input_features"][0, 0]

    np.savez_compressed(
        OUT / "clap_ref.npz",
        audio_paths=np.array([str(p) for p in AUDIOS_REF]),
        audio_embs=np.stack(embs_a).astype(np.float32),
        textos=np.array(TEXTOS_REF),
        text_embs=embs_t.astype(np.float32),
        mel_wav=mel_wav.astype(np.float32),
        mel_ref=mel_ref.astype(np.float32))
    print(f"  clap_ref.npz {(OUT / 'clap_ref.npz').stat().st_size / 1e6:.1f} MB")

    # ---- prova rapida aqui mesmo (a prova oficial e o verify no venv-build) ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(a_path), providers=["CPUExecutionProvider"])
    e = vc.embed_audio(sess, AUDIOS_REF[0])
    cos = float(e @ embs_a[0])
    sess_t = ort.InferenceSession(str(t_path), providers=["CPUExecutionProvider"])
    et = vc.embed_textos(sess_t, vc.abrir_tokenizer(), TEXTOS_REF[:2])
    ref_t = embs_t[:2] / np.linalg.norm(embs_t[:2], axis=1, keepdims=True)
    cos_t = float((et * ref_t).sum(1).min())
    print(f"sanidade ONNX vs torch: audio {cos:.6f}, texto {cos_t:.6f}")
    assert cos >= 0.999 and cos_t >= 0.999
    print("EXPORT OK — rode tools/verify_clap.py no venv-build")


if __name__ == "__main__":
    main()
