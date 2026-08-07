"""Scene Finder - servidor local (127.0.0.1) + fontes online.

Uso:  python server.py --dev   -> so o servidor, abre no navegador
      (uso normal e via app.py, que embute a janela)
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

import encoder
import indexer
import ptbr
import updater
from version import __version__

APP = encoder.app_dir()        # index.html, models
DATA = encoder.data_dir()      # thumbs, indice, cache
CFG = indexer.load_config()
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
PROXY_HOSTS = ("redd.it", "reddit.com", "redditmedia.com",
               "kemono.cr", "czepeku.com", "encounterkit.com")
CZEPEKU_CACHE = DATA / "czepeku_cache.json"
ptbr.usar_cache(DATA / "traducoes.json")   # traducoes ficam junto dos dados

CZEPEKU_CATS = ["fantasy/scenes", "fantasy/maps", "scifi/scenes", "scifi/maps"]

STATE = {
    "emb": None, "items": [], "nemb": None, "ready": False,   # busca local
    "mask": None,          # pastas visiveis (None = todas)
    "idx": {"running": False, "done": 0, "total": 0, "error": None},
}
SEARCH_LOCK = threading.Lock()
# repetir a mesma busca nao pode re-bater nas APIs (Reddit devolve 429)
ONLINE_CACHE = {}


def _norm_dir(p):
    return str(p).replace("/", "\\").rstrip("\\").lower()


def atualizar_mascara():
    """Recalcula quais itens a busca enxerga, a partir de folders_off."""
    off = [_norm_dir(f) for f in (CFG.get("folders_off") or [])]
    itens = STATE["items"]
    if not off or not itens:
        STATE["mask"] = None
        return
    STATE["mask"] = np.array(
        [not any(_norm_dir(it["p"]).startswith(o + "\\") for o in off)
         for it in itens], dtype=bool)


def contar_por_pasta():
    """Quantos itens do indice caem em cada pasta configurada."""
    contas = {f: 0 for f in CFG.get("folders", [])}
    normal = {_norm_dir(f): f for f in contas}
    for it in STATE["items"]:
        p = _norm_dir(it["p"])
        for n, original in normal.items():
            if p.startswith(n + "\\"):
                contas[original] += 1
                break
    return contas


def _boot():
    STATE["emb"], STATE["items"] = indexer.load_index()
    indexer.encoder.encode_texts(["warmup"])  # carrega a sessao ONNX antes da 1a busca
    STATE["nemb"] = indexer.load_name_index(STATE["items"])
    atualizar_mascara()
    STATE["ready"] = True
    # sem indice mas com pastas configuradas = primeira execucao ou troca de
    # modelo (o indice antigo vira incompativel). Reindexa sem o usuario pedir.
    if not STATE["items"] and CFG.get("folders"):
        start_reindex()


def _http_get(url, timeout=5, headers=None):
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


# ---------- fontes online ----------

def src_reddit(q):
    out, threads, errs = [], [], []

    def entries(sub, term):
        url = (f"https://www.reddit.com/r/{sub}/search.rss?"
               f"q={urllib.parse.quote(term)}&restrict_sr=1")
        data, _ = _http_get(url)
        ns = {"a": "http://www.w3.org/2005/Atom",
              "m": "http://search.yahoo.com/mrss/"}
        return ElementTree.fromstring(data).findall("a:entry", ns), ns

    def one(sub):
        try:
            found, ns = entries(sub, q)
            # ordem das palavras importa no Reddit; sem nada, usa a mais especifica
            words = [w for w in q.split() if len(w) > 2]
            if not found and len(words) > 1:
                found, ns = entries(sub, max(words, key=len))
            for e in found:
                title = e.findtext("a:title", "", ns)
                link = e.find("a:link", ns)
                thumb = e.find("m:thumbnail", ns)
                if link is None:
                    continue
                out.append({"title": title, "url": link.get("href"),
                            "thumb": thumb.get("url") if thumb is not None else "",
                            "source": f"r/{sub}"})
        except urllib.error.HTTPError as e:
            errs.append("limite do Reddit, tente em 1 min" if e.code == 429
                        else f"HTTP {e.code}")
        except Exception as e:
            errs.append(str(e))

    for sub in CFG.get("subreddits", []):
        t = threading.Thread(target=one, args=(sub,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=8)
    if not out and errs:
        raise RuntimeError(errs[0])  # so avisa se NADA veio
    return out


IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _kemono_posts(c, q):
    url = (f"https://kemono.cr/api/v1/{c['service']}/user/{c['id']}/posts"
           f"?q={urllib.parse.quote(q)}")
    # DDoS-guard do kemono exige Accept: text/css em requests de API
    data, _ = _http_get(url, headers={"Accept": "text/css"})
    return json.loads(data)


def src_kemono(q, i):
    c = CFG["kemono"][i]
    posts = _kemono_posts(c, q)
    # a busca do kemono e AND: "tavern night" da 0, "tavern" da 50.
    # sem resultado, tenta so a palavra mais especifica (a mais longa).
    words = [w for w in q.split() if len(w) > 2]
    if not posts and len(words) > 1:
        posts = _kemono_posts(c, max(words, key=len))
    out = []
    for p in posts:
        path = (p.get("file") or {}).get("path") or ""
        if not path.lower().endswith(IMG_EXT):
            for a in p.get("attachments") or []:
                if (a.get("path") or "").lower().endswith(IMG_EXT):
                    path = a["path"]
                    break
        if not path.lower().endswith(IMG_EXT):
            continue
        out.append({
            "title": p.get("title") or "(sem titulo)",
            "url": f"https://kemono.cr/{c['service']}/user/{c['id']}/post/{p['id']}",
            "thumb": f"https://img.kemono.cr/thumbnail/data{path}",
            "source": c.get("name") or "kemono",
        })
    return out


# suba quando o formato dos itens mudar, senao o cache antigo (sem os campos
# novos) continua sendo servido ate expirar
CZEPEKU_CACHE_V = 2


def _czepeku_catalog():
    if CZEPEKU_CACHE.exists():
        try:
            cache = json.loads(CZEPEKU_CACHE.read_text("utf-8"))
            if (cache.get("v") == CZEPEKU_CACHE_V
                    and time.time() - cache["fetched"] < 86400):
                return cache["items"]
        except Exception:
            pass
    items = []
    for page in CZEPEKU_CATS:
        try:
            html, _ = _http_get(f"https://www.czepeku.com/{page}", timeout=15)
            html = html.decode("utf-8", "ignore")
            for m in re.finditer(
                    r'href="/(fantasy|scifi)/(scenes|maps)/([a-z0-9-]+)"', html):
                cat, kind, slug = m.groups()
                chunk = html[m.end():m.end() + 4000]
                src = re.search(
                    r'src="(https://content\.encounterkit\.com/[^"]+)"', chunk)
                items.append({
                    "title": slug.replace("-", " ").title(),
                    "url": f"https://www.czepeku.com/{cat}/{kind}/{slug}",
                    "thumb": (src.group(1).replace("width=1920", "width=480")
                              if src else ""),
                    "source": f"czepeku {cat}/{kind}",
                    "cat": f"{cat}/{kind}",
                })
        except Exception:
            continue
    seen, uniq = set(), []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"])
            uniq.append(it)
    if uniq:
        CZEPEKU_CACHE.write_text(
            json.dumps({"v": CZEPEKU_CACHE_V, "fetched": time.time(),
                        "items": uniq}), "utf-8")
    return uniq


def src_czepeku(q, cat=None):
    """Ranqueia por quantos termos batem no titulo (frase inteira raramente bate).

    `cat` limita a uma das quatro secoes do site (fantasy/scifi x scenes/maps).
    """
    words = [w.lower() for w in q.split() if len(w) > 2] or [q.lower()]
    hits = []
    for it in _czepeku_catalog():
        if cat and it.get("cat") != cat:
            continue
        t = it["title"].lower()
        n = sum(1 for w in words if w in t)
        if n:
            hits.append((n, it))
    hits.sort(key=lambda h: -h[0])
    return [it for _, it in hits[:40]]


# ---------- reindex ----------

def start_reindex():
    if STATE["idx"]["running"]:
        return
    STATE["idx"] = {"running": True, "done": 0, "total": 0, "error": None}

    def run():
        try:
            def prog(d, t):
                STATE["idx"]["done"], STATE["idx"]["total"] = d, t
            indexer.build_index(CFG, progress=prog)
            STATE["emb"], STATE["items"] = indexer.load_index()
        except Exception as e:
            STATE["idx"]["error"] = str(e)
        finally:
            STATE["idx"]["running"] = False

    threading.Thread(target=run, daemon=True).start()


# ---------- handler ----------

def _card(score, gi):
    it = STATE["items"][gi]
    p = it["p"]
    try:
        foundry = str(Path(p).relative_to(CFG["foundry_data"])).replace("\\", "/")
    except (ValueError, KeyError):
        foundry = p.replace("\\", "/")
    return {"i": gi, "path": p, "foundry": foundry,
            "thumb": f"/thumbs/{it['t']}", "score": round(score, 3),
            "name": Path(p).stem, "dir": "/".join(Path(p).parts[-3:-1])}


def _formatar(res, agrupar):
    """Um card por MAPA; as variantes vao dentro dele.

    Sem agrupar, um unico mapa com 12 variantes ocupa a tela inteira e esconde
    as outras opcoes."""
    if not agrupar:
        return [_card(s, i) for s, i in res]
    grupos, ordem = {}, []
    for score, gi in res:
        k = indexer.map_key(STATE["items"][gi]["p"])
        if k not in grupos:
            grupos[k] = _card(score, gi)
            grupos[k]["variants"] = []
            ordem.append(k)
        elif len(grupos[k]["variants"]) < 40:
            grupos[k]["variants"].append(_card(score, gi))
    return [grupos[k] for k in ordem]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8",
              cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=604800")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/reindex":
            start_reindex()
            return self._json({"ok": True})

        if path == "/api/config":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                novo = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            CFG.update({k: v for k, v in novo.items() if k in indexer.DEFAULT_CONFIG})
            indexer.save_config(CFG)
            atualizar_mascara()
            return self._json({"ok": True, "config": CFG})

        if path == "/api/toggle":
            # liga/desliga pasta local ou fonte online sem reindexar nada
            n = int(self.headers.get("Content-Length") or 0)
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
                tipo, alvo, ligado = d["tipo"], d["alvo"], bool(d["on"])
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            chave = "folders_off" if tipo == "pasta" else "sources_off"
            desligados = [x for x in (CFG.get(chave) or [])]
            igual = (_norm_dir if tipo == "pasta" else str)
            desligados = [x for x in desligados if igual(x) != igual(alvo)]
            if not ligado:
                desligados.append(alvo)
            CFG[chave] = desligados
            indexer.save_config(CFG)
            if tipo == "pasta":
                atualizar_mascara()
            return self._json({"ok": True, chave: desligados})

        if path == "/api/update/download":
            updater.download_async()
            return self._json({"ok": True})

        if path == "/api/update/install":
            return self._json({"ok": updater.install()})

        self._json({"error": "?"}, 404)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        q = (qs.get("q") or [""])[0].strip()

        if u.path == "/":
            return self._send(200, (APP / "index.html").read_bytes(),
                              "text/html; charset=utf-8")

        if u.path.startswith("/thumbs/"):
            f = DATA / "thumbs" / Path(u.path).name
            if f.exists():
                return self._send(200, f.read_bytes(), "image/webp", cache=True)
            return self._json({"error": "thumb"}, 404)

        if u.path == "/api/search":
            if not STATE["ready"]:
                return self._json({"ready": False, "results": []})
            alvo = CFG.get("top_k", 60)
            with SEARCH_LOCK:
                # busca fundo porque o agrupamento colapsa muitos resultados
                # num card so; sem isso a tela ficaria com poucos mapas
                res = indexer.search(q, STATE["emb"], STATE["items"],
                                     alvo * 8, nemb=STATE["nemb"],
                                     mask=STATE["mask"]) if q else []
            return self._json({"ready": True,
                               "results": _formatar(res, True)[:alvo]})

        if u.path == "/api/similar":
            # "mais como esta": usa o vetor da propria imagem, ja indexado
            if not STATE["ready"]:
                return self._json({"ready": False, "results": []})
            try:
                gi = int((qs.get("i") or ["-1"])[0])
                alvo = CFG.get("top_k", 60)
                with SEARCH_LOCK:
                    scores = STATE["emb"] @ STATE["emb"][gi]
                    if STATE["mask"] is not None:
                        scores = np.where(STATE["mask"], scores, -np.inf)
                    ordem = [j for j in np.argsort(-scores)[:alvo * 10]
                             if np.isfinite(scores[j])]
                # tira o proprio mapa: quem pede "parecidos" quer OUTROS mapas,
                # nao as 40 variantes deste
                base_key = indexer.map_key(STATE["items"][gi]["p"])
                res = [(float(scores[j]), int(j)) for j in ordem
                       if indexer.map_key(STATE["items"][j]["p"]) != base_key]
                return self._json({"ready": True,
                                   "results": _formatar(res, True)[:alvo],
                                   "base": _card(1.0, gi)})
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if u.path == "/api/online":
            src = (qs.get("src") or [""])[0]
            qe = ptbr.to_en(q)  # titulos das fontes sao em ingles
            ck = (src, qe)
            hit = ONLINE_CACHE.get(ck)
            if hit and time.time() - hit[0] < 300:
                return self._json({"results": hit[1], "q": qe})
            try:
                if src == "reddit":
                    r = src_reddit(qe)
                elif src.startswith("kemono:"):
                    r = src_kemono(qe, int(src.split(":")[1]))
                elif src.startswith("czepeku:"):
                    r = src_czepeku(qe, src.split(":", 1)[1])
                elif src == "czepeku":
                    r = src_czepeku(qe)
                else:
                    return self._json({"error": "src"}, 400)
                ONLINE_CACHE[ck] = (time.time(), r)
                return self._json({"results": r, "q": qe})
            except Exception as e:
                return self._json({"results": [], "q": qe, "error": str(e)})

        if u.path == "/api/img":
            url = (qs.get("u") or [""])[0]
            host = urllib.parse.urlparse(url).hostname or ""
            if not any(host == h or host.endswith("." + h) for h in PROXY_HOSTS):
                return self._json({"error": "host"}, 403)
            try:
                headers = {"Referer": "https://kemono.cr/"} if "kemono" in host else {}
                data, ctype = _http_get(url, headers=headers)
                return self._send(200, data, ctype or "image/jpeg", cache=True)
            except Exception as e:
                return self._json({"error": str(e)}, 502)

        if u.path == "/api/open":
            try:
                it = STATE["items"][int((qs.get("i") or ["-1"])[0])]
                subprocess.Popen(["explorer", "/select,", it["p"]])
                return self._json({"ok": True})
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if u.path == "/api/view":
            # abre a imagem no visualizador padrao (ver em tamanho real)
            try:
                it = STATE["items"][int((qs.get("i") or ["-1"])[0])]
                os.startfile(it["p"])
                return self._json({"ok": True})
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if u.path == "/api/ext":
            url = (qs.get("u") or [""])[0]
            if urllib.parse.urlparse(url).scheme in ("http", "https"):
                os.startfile(url)  # navegador padrao
                return self._json({"ok": True})
            return self._json({"error": "url"}, 400)

        if u.path == "/api/sources":
            off = set(CFG.get("sources_off") or [])
            srcs = [{"id": "reddit", "label": "Reddit"}]
            for i, c in enumerate(CFG.get("kemono", [])):
                srcs.append({"id": f"kemono:{i}",
                             "label": c.get("name") or f"kemono {i}"})
            for cat in CZEPEKU_CATS:
                srcs.append({"id": f"czepeku:{cat}", "label": f"Czepeku {cat}"})
            for s in srcs:
                s["on"] = s["id"] not in off
            # a busca so dispara as ligadas; a lista completa alimenta os filtros
            return self._json({"todas": srcs,
                               "ativas": [s for s in srcs if s["on"]]})

        if u.path == "/api/folders":
            contas = contar_por_pasta()
            off = {_norm_dir(f) for f in (CFG.get("folders_off") or [])}
            return self._json([
                {"path": f, "itens": contas.get(f, 0),
                 "on": _norm_dir(f) not in off}
                for f in CFG.get("folders", [])])

        if u.path == "/api/config":
            sug = indexer.find_foundry()
            return self._json({"config": CFG, "sugestao": str(sug) if sug else "",
                               "version": __version__,
                               "data_dir": str(DATA)})

        if u.path == "/api/update":
            return self._json({"version": __version__, **updater.STATE})

        if u.path == "/api/pick":
            # dialogo nativo so existe quando rodando dentro da janela pywebview
            win = globals().get("WINDOW")
            if win is None:
                return self._json({"path": "", "sem_dialogo": True})
            try:
                import webview
                sel = win.create_file_dialog(webview.FOLDER_DIALOG)
                return self._json({"path": sel[0] if sel else ""})
            except Exception as e:
                return self._json({"path": "", "error": str(e)})

        if u.path == "/api/index/status":
            st = dict(STATE["idx"])
            st["indexed"] = len(STATE["items"])
            return self._json(st)

        self._json({"error": "?"}, 404)


def serve():
    threading.Thread(target=_boot, daemon=True).start()
    updater.check_async()
    port = CFG.get("port", 8060)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    return httpd, port


if __name__ == "__main__":
    httpd, port = serve()
    print(f"Scene Finder em http://127.0.0.1:{port}/  (Ctrl+C encerra)")
    httpd.serve_forever()
