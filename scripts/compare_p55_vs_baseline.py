#!/usr/bin/env python3
"""Compare the prob=0.55 test against the existing prob=0.5929 output on frames 273-300."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_villas_273_300_p55")
BASE_MANIFEST = os.path.join(ROOT, "data", "para_subir", "villas_263_300", "manifest.json")
BASE_RESTO = os.path.join(ROOT, "data", "stardist_resto", "stardist_resto_report.json")

rep = json.load(open(os.path.join(SRC, "stardist_273_300_p55_report.json")))
# 273-279: manifest del primer lote (mapeo correcto hasta 279).
# 280-300: reporte de la corrida corregida (mapeo real de meta.json).
base = json.load(open(BASE_MANIFEST))["per_frame"].copy()
base.update(json.load(open(BASE_RESTO)))

rows = []
for f in range(273, 301):
    k = str(f)
    rows.append((f, rep[k]['n_poligonos'], base.get(k, {}).get('n_poligonos', 0)))
total_p55 = sum(r[1] for r in rows)
total_base = sum(r[2] for r in rows)

doc = os.path.join(ROOT, "docs", "TEST_PROB_055_VILLAS_273_300.md")
lines = [
    "# Test prob_thresh=0.55 — Villas frames 273-300",
    "",
    "Mismo modelo y mapeo que la producción; único cambio: `prob_thresh=0.55` en vez de 0.5929.",
    "NMS 0.02. Comparación contra los polígonos ya generados con 0.5929.",
    "",
    "| Frame | baseline 0.5929 | test 0.55 | delta |",
    "|---:|---:|---:|---:|",
]
for f, p55, b in rows:
    lines.append(f"| {f} | {b} | {p55} | {p55-b:+d} |")
lines += [
    "",
    f"**Total:** baseline {total_base} → test {total_p55} ({total_p55-total_base:+d}).",
    "",
    "Esto es sólo un conteo de polígonos, no una evaluación contra oro: "
    "no sabemos cuáles son aciertos y cuáles falsos positivos sin revisar los overlays.",
    "",
]

imgs = []
for f in range(273, 301):
    name = rep[str(f)]['image']
    p = os.path.join(SRC, f"nms0.02_{name}")
    if os.path.exists(p):
        imgs.append((f, p))
if imgs:
    cols, rows_n = 7, (len(imgs) + 6) // 7
    fig, axs = plt.subplots(rows_n, cols, figsize=(18, 2.8 * rows_n))
    for ax in axs.flat:
        ax.axis("off")
    for ax, (f, p) in zip(axs.flat, imgs):
        ax.imshow(Image.open(p))
        ax.set_title(f"frame {f} · {rep[str(f)]['n_poligonos']} pol.", fontsize=8)
        ax.axis("off")
    fig.suptitle("StarDist prob=0.55 — Villas 273-300", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(SRC, "overlays_p55.png"), dpi=110)
    plt.close(fig)
    lines.append("Overlays: `data/stardist_villas_273_300_p55/overlays_p55.png`.\n")

with open(doc, "w") as fh:
    fh.write("\n".join(lines))
print("escrito", doc)
print("baseline", total_base, "p55", total_p55, "delta", total_p55 - total_base)
