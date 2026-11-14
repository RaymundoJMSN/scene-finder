"""Traduz a consulta para as fontes online, que so casam titulo em ingles.

A busca LOCAL nao usa isto - o encoder de texto ja e multilingue. Aqui e so para
nao mandar "loja de pedras" para o Reddit, onde todo titulo esta em ingles.

Duas camadas: um dicionario embutido (instantaneo e offline, cobre o vocabulario
de cenario de RPG) e, quando ele nao da conta da frase inteira, um tradutor
online com cache em disco. Sem internet, o dicionario continua valendo.
"""
import json
import re
import threading
import unicodedata
import urllib.parse
import urllib.request

# termos de cenario de RPG; o que nao estiver aqui vai para o tradutor online
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
    "pirata": "pirate", "naufragio": "shipwreck", "conves": "deck",
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
    "reator": "reactor", "hangar": "hangar", "oficina": "workshop",
    "pedra": "stone", "pedras": "stones", "joia": "jewel", "joias": "jewelry",
    "ourives": "jeweler", "relojoeiro": "clockmaker", "alquimista": "alchemist",
    "boticario": "apothecary", "armeiro": "armorer", "carpinteiro": "carpenter",
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
    # objetos e materiais - vocabulario dos ACERVOS DE PECAS (Forgotten
    # Adventures e afins sao 148 mil sprites de movel, planta e traste; o
    # dicionario so tinha nome de LUGAR, entao "porta" nunca virava "door")
    "porta": "door", "portas": "doors", "janela": "window",
    "janelas": "windows", "parede": "wall", "piso": "floor", "chao": "floor",
    "telhado": "roof", "escada": "stairs", "escadaria": "stairs",
    "coluna": "pillar", "arco": "arch", "grade": "grate", "cerca": "fence",
    "portal": "portal", "tapume": "fence", "corrimao": "railing",
    "mesa": "table", "cadeira": "chair", "banco": "bench", "cama": "bed",
    "colchao": "mattress", "travesseiro": "pillow", "tapete": "rug",
    "cortina": "curtain", "armario": "wardrobe", "comoda": "dresser",
    "prateleira": "shelf", "estante": "bookshelf", "bau": "chest",
    "caixa": "crate", "caixote": "crate", "barril": "barrel",
    "tonel": "barrel", "saco": "sack", "cesta": "basket", "jarro": "jug",
    "vaso": "vase", "garrafa": "bottle", "copo": "cup", "caneca": "mug",
    "prato": "plate", "panela": "pot", "caldeirao": "cauldron",
    "balde": "bucket", "bigorna": "anvil", "fornalha": "furnace",
    "lareira": "fireplace", "fogueira": "campfire", "tocha": "torch",
    "vela": "candle", "lampiao": "lantern", "lanterna": "lantern",
    "livro": "book", "livros": "books", "pergaminho": "scroll",
    "mapa": "map", "bandeira": "banner", "estandarte": "banner",
    "quadro": "painting", "espelho": "mirror", "estatua": "statue",
    "busto": "bust", "fonte": "fountain", "poco": "well", "placa": "sign",
    "espada": "sword", "escudo": "shield", "machado": "axe",
    "lanca": "spear", "adaga": "dagger", "martelo": "hammer",
    "cajado": "staff", "varinha": "wand", "armadura": "armor",
    "elmo": "helmet", "besta": "crossbow", "flecha": "arrow",
    "corda": "rope", "corrente": "chain", "chave": "key",
    "moeda": "coin", "moedas": "coins", "tesouro": "treasure",
    "gema": "gem", "cristal": "crystal",
    "pocao": "potion", "frasco": "flask", "carroca": "wagon",
    "carruagem": "carriage", "roda": "wheel",
    "tambor": "drum", "ferramenta": "tool", "ferramentas": "tools",
    "arvore": "tree", "arvores": "trees", "tronco": "log",
    "toco": "stump", "galho": "branch", "arbusto": "bush", "moita": "bush",
    "grama": "grass", "flor": "flower", "flores": "flowers",
    "planta": "plant", "plantas": "plants", "cogumelo": "mushroom",
    "cogumelos": "mushrooms", "raiz": "root", "trepadeira": "vine",
    "cacto": "cactus", "palmeira": "palm tree", "folha": "leaf",
    "folhas": "leaves", "rocha": "rock", "rochas": "rocks",
    "cristais": "crystals", "estalagmite": "stalagmite",
    "osso": "bone", "ossos": "bones", "cranio": "skull",
    "esqueleto": "skeleton", "caixao": "coffin", "lapide": "tombstone",
    "teia": "cobweb", "sangue": "blood",
    "madeira": "wood", "ferro": "iron", "aco": "steel", "ouro": "gold",
    "prata": "silver", "bronze": "bronze", "tijolo": "brick",
    "marmore": "marble", "tecido": "cloth", "couro": "leather",
    "vidro": "glass", "comida": "food", "pao": "bread", "carne": "meat",
    "fruta": "fruit", "peixe": "fish", "vinho": "wine", "cerveja": "beer",
    "quebrado": "broken", "quebrada": "broken", "velho": "old",
    "velha": "old", "pequeno": "small", "pequena": "small",
    "grande": "large", "redondo": "round", "redonda": "round",
    "aberto": "open", "aberta": "open", "fechado": "closed",
    "fechada": "closed", "vermelho": "red", "vermelha": "red",
    "azul": "blue", "verde": "green", "amarelo": "yellow",
    "branco": "white", "branca": "white", "preto": "black",
    "preta": "black", "marrom": "brown", "roxo": "purple",
}
STOP = {"de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "com", "sem", "para", "por", "um", "uma", "o", "a", "os", "as", "e"}

API = "https://api.mymemory.translated.net/get"
UA = {"User-Agent": "scene-finder/1.0"}
_cache = {}
_cache_path = None
_lock = threading.Lock()


def usar_cache(caminho):
    """Aponta o cache para um arquivo (o app passa a pasta de dados)."""
    global _cache_path
    _cache_path = caminho
    try:
        _cache.update(json.loads(caminho.read_text("utf-8")))
    except Exception:
        pass


def _gravar_cache():
    if not _cache_path:
        return
    try:
        _cache_path.write_text(json.dumps(_cache, ensure_ascii=False), "utf-8")
    except Exception:
        pass


def _strip(w):
    w = unicodedata.normalize("NFKD", w)
    return "".join(c for c in w if not unicodedata.combining(c)).lower()


def pelo_dicionario(query):
    """Palavra a palavra. Devolve (texto, faltou_alguma_palavra)."""
    palavras = [w for w in re.split(r"\s+", query.strip()) if w]
    saida, traduziu, faltou = [], False, False
    for w in palavras:
        k = _strip(w).strip(".,;:!?")
        if k in STOP:
            continue
        if k in PT_EN:
            saida.append(PT_EN[k])
            traduziu = True
        else:
            saida.append(w)
            faltou = True
    if not saida:
        return query, True
    return (" ".join(saida) if traduziu else query), faltou


def _online(query):
    url = f"{API}?q={urllib.parse.quote(query)}&langpair=pt-BR|en"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=5) as r:
        dados = json.loads(r.read())
    txt = (dados.get("responseData") or {}).get("translatedText") or ""
    txt = txt.strip()
    # a API devolve avisos em maiusculas quando estoura a cota
    if not txt or "MYMEMORY WARNING" in txt.upper() or "QUERY LENGTH" in txt.upper():
        raise ValueError("resposta invalida do tradutor")
    return txt


def to_en(query):
    """Consulta em ingles. Nunca levanta excecao: sem rede, cai no dicionario."""
    query = (query or "").strip()
    if not query:
        return query
    with _lock:
        if query in _cache:
            return _cache[query]

    texto, faltou = pelo_dicionario(query)
    # o dicionario deu conta da frase inteira: nem precisa de rede
    if not faltou:
        with _lock:
            _cache[query] = texto
            _gravar_cache()
        return texto

    try:
        texto = _online(query)
    except Exception:
        pass  # offline ou cota estourada: fica o que o dicionario conseguiu
    with _lock:
        _cache[query] = texto
        _gravar_cache()
    return texto


_aquecendo = set()


def _aquecer(query):
    try:
        to_en(query)
    finally:
        with _lock:
            _aquecendo.discard(query)


def to_en_rapido(query):
    """Nunca bloqueia: responde com dicionario+cache na hora e, se sobrou
    palavra desconhecida, aquece a traducao online em background para a
    PROXIMA busca. A primeira consulta com palavra fora do dicionario travava
    ~4 s esperando a API - inaceitavel no meio de uma sessao de jogo."""
    query = (query or "").strip()
    if not query:
        return query
    with _lock:
        if query in _cache:
            return _cache[query]
    texto, faltou = pelo_dicionario(query)
    if faltou:
        with _lock:
            novo = query not in _aquecendo
            if novo:
                _aquecendo.add(query)
        if novo:
            threading.Thread(target=_aquecer, args=(query,), daemon=True).start()
    else:
        with _lock:
            _cache[query] = texto
            _gravar_cache()
    return texto


if __name__ == "__main__":
    assert pelo_dicionario("taverna a noite")[0] == "tavern night"
    assert pelo_dicionario("templo na selva")[0] == "temple jungle"
    assert pelo_dicionario("jungle temple") == ("jungle temple", True)
    assert pelo_dicionario("Ravenloft") == ("Ravenloft", True)
    # sem palavra desconhecida, nao consulta a rede
    assert to_en("taverna a noite") == "tavern night"
    print("dicionario OK")
    try:
        r = to_en("loja de pedras preciosas")
        print("com rede: 'loja de pedras preciosas' ->", repr(r))
        assert "shop" in r.lower() or "store" in r.lower() or "loja" in r.lower()
    except Exception as e:
        print("sem rede (esperado offline):", e)
    print("ptbr OK")
