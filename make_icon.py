"""Gera icon.ico: lupa verde sobre grid de mapa, fundo escuro arredondado."""
from pathlib import Path

from PIL import Image, ImageDraw

APP = Path(__file__).resolve().parent
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([8, 8, 248, 248], 52, fill=(17, 18, 20, 255))
for v in range(56, 220, 40):  # grid de mapa
    d.line([(v, 32), (v, 224)], fill=(58, 60, 66, 255), width=5)
    d.line([(32, v), (224, v)], fill=(58, 60, 66, 255), width=5)
d.ellipse([60, 60, 164, 164], outline=(74, 222, 128, 255), width=18)
d.line([(152, 152), (210, 210)], fill=(74, 222, 128, 255), width=24)
img.save(APP / "icon.ico",
         sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                (128, 128), (256, 256)])
print("icon.ico OK")
