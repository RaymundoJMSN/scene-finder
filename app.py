"""Scene Finder - app: servidor embutido + janela nativa (WebView2)."""
import os
import socket
import threading

import webview

import server


def main():
    port = server.CFG.get("port", 8060)
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
        probe.close()
    except OSError:
        # ja tem instancia rodando: so abre outra janela apontando pra ela
        webview.create_window("Scene Finder", f"http://127.0.0.1:{port}/",
                              width=1280, height=800,
                              background_color="#111214")
        webview.start()
        return

    httpd, _ = server.serve()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # o server usa esta referencia para abrir o seletor de pastas nativo
    server.WINDOW = webview.create_window(
        "Scene Finder", f"http://127.0.0.1:{port}/",
        width=1280, height=800, background_color="#111214")
    webview.start()
    os._exit(0)  # janela fechou -> derruba servidor junto


if __name__ == "__main__":
    main()
