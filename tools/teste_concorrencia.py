"""Prova que buscar durante a indexacao nao derruba o app.

O DirectML nao aceita duas chamadas simultaneas: sem o lock global do encoder,
este teste mata o processo com ACCESS_VIOLATION (0xC0000005) em segundos - e o
usuario so ve a janela sumir, porque o app roda sem console.

Reproduz o caminho real: uma thread indexando (imagem + texto, como o
build_index faz) e outra pesquisando ao mesmo tempo.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

import encoder  # noqa: E402
import indexer  # noqa: E402

SEGUNDOS = 25
erros = []
paradas = threading.Event()


def indexando(caminhos):
    """Imita o loop do build_index: prepara em paralelo, infere, embeda nomes."""
    try:
        while not paradas.is_set():
            for i in range(0, len(caminhos), 8):
                if paradas.is_set():
                    return
                lote = caminhos[i:i + 8]
                prep = [indexer._preparar((p, 0)) for p in lote]
                arr = [r[0] for r in prep if r]
                if arr:
                    encoder.encode_prepared(np.stack(arr))
                    encoder.encode_texts([Path(p).stem for p in lote])
    except Exception as e:
        erros.append(f"indexacao: {type(e).__name__}: {e}")


def pesquisando():
    consultas = ["taverna a noite", "jungle temple", "navio pirata", "deserto"]
    try:
        n = 0
        while not paradas.is_set():
            encoder.encode_texts([consultas[n % len(consultas)]])
            n += 1
            time.sleep(0.4)
    except Exception as e:
        erros.append(f"busca: {type(e).__name__}: {e}")


def main():
    print("provider:", encoder.provedor_ativo())
    caminhos = indexer.scan(indexer.load_config())[:64]
    if len(caminhos) < 8:
        print("acervo pequeno demais"); return 1

    threads = [threading.Thread(target=indexando, args=(caminhos,), daemon=True),
               threading.Thread(target=pesquisando, daemon=True),
               threading.Thread(target=pesquisando, daemon=True)]
    for t in threads:
        t.start()
    print(f"indexando e pesquisando ao mesmo tempo por {SEGUNDOS}s...")
    time.sleep(SEGUNDOS)
    paradas.set()
    for t in threads:
        t.join(timeout=30)

    if erros:
        print("FALHOU:")
        for e in erros:
            print("  ", e)
        return 1
    print("CONCORRENCIA OK - processo sobreviveu, sem excecao")
    return 0


if __name__ == "__main__":
    sys.exit(main())
