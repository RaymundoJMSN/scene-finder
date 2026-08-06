"""Checagem rapida do encoder ONNX recem-exportado (usada pelo CI).

Nao precisa de indice: so confirma que os dois modelos carregam, que a saida tem
a forma certa e que o modelo de texto continua multilingue - se a exportacao
quebrar o multilingual, buscar em portugues para de funcionar em silencio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import encoder  # noqa: E402


def main():
    t = encoder.encode_texts(["tavern at night", "taverna a noite", "spaceship"])
    assert t.shape == (3, 512), f"forma inesperada no texto: {t.shape}"

    pt_en = float(t[0] @ t[1])
    nada_a_ver = float(t[0] @ t[2])
    print(f"PT x EN (mesmo sentido): {pt_en:.3f}")
    print(f"PT x assunto diferente : {nada_a_ver:.3f}")
    # embeddings de texto do CLIP vivem numa faixa estreita e alta: o contraste
    # entre assuntos e de centesimos, nao de decimos. Medido: 0.90 vs 0.81.
    assert pt_en > 0.8, "modelo multilingue quebrado: PT nao casa com EN"
    assert pt_en > nada_a_ver + 0.05, "sem contraste entre assuntos diferentes"

    img = Image.new("RGB", (900, 600), (90, 70, 50))
    v = encoder.encode_images([img])
    assert v.shape == (1, 512), f"forma inesperada na imagem: {v.shape}"
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-3, "embedding nao normalizado"

    print("SMOKE OK")


if __name__ == "__main__":
    main()
