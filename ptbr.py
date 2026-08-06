"""PT->EN para as fontes online (Reddit/kemono/czepeku so casam texto literal).

A busca LOCAL nao usa isto - o CLIP multilingual ja entende portugues. Aqui e so
para nao mandar "taverna" pro Reddit, onde todo titulo esta em ingles.
"""
import re
import unicodedata

# termos de cenario de RPG; o que nao estiver aqui passa direto
PT_EN = {
    "taverna": "tavern", "estalagem": "inn", "hospedaria": "inn",
    "masmorra": "dungeon", "calabouco": "dungeon", "cripta": "crypt",
    "castelo": "castle", "fortaleza": "fortress", "forte": "fort",
    "torre": "tower", "muralha": "wall", "portao": "gate", "ponte": "bridge",
    "templo": "temple", "igreja": "church", "catedral": "cathedral",
    "capela": "chapel", "santuario": "shrine", "mosteiro": "monastery",
    "altar": "altar", "biblioteca": "library", "laboratorio": "laboratory",
    "forja": "forge", "ferreiro": "blacksmith", "mercado": "market",
    "feira": "market", "loja": "shop", "armazem": "warehouse",
    "taberna": "tavern", "padaria": "bakery", "estabulo": "stable",
    "moinho": "mill", "fazenda": "farm", "celeiro": "barn",
    "vila": "village", "vilarejo": "village", "aldeia": "village",
    "cidade": "city", "povoado": "town", "acampamento": "camp",
    "porto": "port", "doca": "docks", "cais": "docks", "farol": "lighthouse",
    "navio": "ship", "barco": "boat", "veleiro": "sailing ship",
    "pirata": "pirate", "naufragio": "shipwreck", "convés": "deck",
    "floresta": "forest", "mata": "woods", "bosque": "grove",
    "selva": "jungle", "pantano": "swamp", "brejo": "marsh",
    "deserto": "desert", "oasis": "oasis", "duna": "dunes",
    "montanha": "mountain", "caverna": "cave", "gruta": "cavern",
    "mina": "mine", "penhasco": "cliff", "desfiladeiro": "canyon",
    "praia": "beach", "ilha": "island", "lago": "lake", "rio": "river",
    "cachoeira": "waterfall", "campo": "field", "planicie": "plains",
    "tundra": "tundra", "gelo": "ice", "neve": "snow", "vulcao": "volcano",
    "esgoto": "sewer", "cemiterio": "graveyard", "tumba": "tomb",
    "ruinas": "ruins", "arena": "arena", "coliseu": "colosseum",
    "prisao": "prison", "cadeia": "jail", "cela": "cell",
    "trono": "throne", "salao": "hall", "palacio": "palace",
    "mansao": "mansion", "casa": "house", "cabana": "cabin",
    "quarto": "bedroom", "cozinha": "kitchen", "porao": "basement",
    "jardim": "garden", "praca": "square", "rua": "street", "beco": "alley",
    "inferno": "hell", "abismo": "abyss", "covil": "lair", "toca": "lair",
    "ninho": "nest", "labirinto": "labyrinth", "observatorio": "observatory",
    "nave": "spaceship", "espacial": "space", "estacao": "station",
    "laboratorio": "lab", "reator": "reactor", "hangar": "hangar",
    "noite": "night", "dia": "day", "amanhecer": "dawn", "poente": "sunset",
    "inverno": "winter", "verao": "summer", "outono": "autumn",
    "chuva": "rain", "tempestade": "storm", "nevoa": "fog",
    "assombrado": "haunted", "assombrada": "haunted",
    "abandonado": "abandoned", "abandonada": "abandoned",
    "subterraneo": "underground", "subterranea": "underground",
    "antigo": "ancient", "antiga": "ancient", "magico": "magic",
    "magica": "magic", "sagrado": "sacred", "sagrada": "sacred",
    "sombrio": "dark", "sombria": "dark", "escuro": "dark",
    "dragao": "dragon", "anao": "dwarven", "elfo": "elven",
    "elfico": "elven", "goblin": "goblin", "orc": "orc", "bruxa": "witch",
    "mago": "wizard", "feiticeiro": "sorcerer", "vampiro": "vampire",
    "mortos": "undead", "fantasma": "ghost", "demonio": "demon",
    "batalha": "battle", "guerra": "war", "mapa": "map", "cena": "scene",
    "interior": "interior", "exterior": "exterior", "entrada": "entrance",
}
STOP = {"de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "com", "sem", "para", "por", "um", "uma", "o", "a", "os", "as", "e"}


def _strip(w):
    w = unicodedata.normalize("NFKD", w)
    return "".join(c for c in w if not unicodedata.combining(c)).lower()


def to_en(query):
    """Traduz palavra a palavra o que reconhecer; devolve query original se nada bater."""
    words = [w for w in re.split(r"\s+", query.strip()) if w]
    out, hit = [], False
    for w in words:
        k = _strip(w)
        if k in STOP:
            continue
        if k in PT_EN:
            out.append(PT_EN[k])
            hit = True
        else:
            out.append(w)
    return " ".join(out) if hit and out else query


if __name__ == "__main__":
    assert to_en("taverna a noite") == "tavern night", to_en("taverna a noite")
    assert to_en("templo na selva") == "temple jungle"
    assert to_en("navio pirata") == "ship pirate"
    assert to_en("jungle temple") == "jungle temple"      # ingles passa intacto
    assert to_en("Ravenloft") == "Ravenloft"              # nome proprio intacto
    assert to_en("floresta assombrada") == "forest haunted"
    print("ptbr OK")
