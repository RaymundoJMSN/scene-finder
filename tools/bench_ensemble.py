"""Mede se somar a consulta traduzida (PT+EN mediados) melhora o ranking.

Gabarito automatico: para cada palavra PT do dicionario do ptbr ("taverna"),
sao relevantes os arquivos cujo NOME contem a traducao EN ("tavern"). A meta e
ver se a consulta em portugues alcanca melhor os nomes em ingles.
Metrica: MRR e recall@20 sobre o indice de nomes real.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import encoder  # noqa: E402
import indexer  # noqa: E402
import ptbr  # noqa: E402

D = Path(r"C:\Users\rayna\AppData\Local\SceneFinder")
indexer.APP = D

CONSULTAS = [
    ("taverna", "tavern"), ("floresta", "forest"), ("caverna", "cave"),
    ("igreja", "church"), ("ponte", "bridge"), ("mercado", "market"),
    ("biblioteca", "library"), ("prisao", "prison"), ("navio", "ship"),
    ("deserto", "desert"), ("pantano", "swamp"), ("castelo", "castle"),
    ("cachoeira", "waterfall"), ("cemiterio", "graveyard"), ("moinho", "mill"),
    ("farol", "lighthouse"), ("celeiro", "barn"), ("templo", "temple"),
    ("torre", "tower"), ("padaria", "bakery"),
]


def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def avaliar(nemb, itens_lower, variante):
    mrr, rec = 0.0, 0
    usadas = 0
    for pt, en in CONSULTAS:
        alvo = np.array([en in nome for nome in itens_lower])
        if not alvo.any():
            continue
        usadas += 1
        if variante == "pt":
            q = encoder.encode_texts([pt])[0]
        elif variante == "en":
            q = encoder.encode_texts([en])[0]
        else:  # ensemble
            dois = encoder.encode_texts([pt, en])
            q = _norm(dois.mean(axis=0, keepdims=True))[0]
        ordem = np.argsort(-(nemb @ q))
        pos = next((r + 1 for r, j in enumerate(ordem[:2000]) if alvo[j]), None)
        if pos:
            mrr += 1 / pos
        rec += int(alvo[ordem[:20]].any())
    return mrr / usadas, rec / usadas, usadas


def main():
    emb, items = indexer.load_index(D)
    nemb = indexer._carregar_nemb(D, len(items))
    print(f"indice: {len(items)} itens")
    itens_lower = [Path(it["p"]).stem.lower() for it in items]
    for v in ("pt", "ensemble", "en"):
        mrr, rec, n = avaliar(nemb, itens_lower, v)
        print(f"  {v:9s} MRR {mrr:.3f}  recall@20 {rec:.2f}  ({n} consultas)")


if __name__ == "__main__":
    main()
