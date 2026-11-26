#!/usr/bin/env python3
"""Render the Villas 273-300 prob_thresh sweep and compare with the baseline."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_villas_273_300_sweep")
DOC = os.path.join(ROOT, "docs", "SWEEP_PROB_VILLAS_273_300.md")
SHOW = [273, 280, 288, 294, 300]

rep = json.load(open(os.path.join(SRC, "stardist_villas_273_300_sweep.json")))
probs = rep["probs"]
totals = {p: sum(v["n_poligonos"] for v in rep["per_prob"][f"{p:.4f}"].values())
          for p in probs}
baseline = totals[probs[-1]]

lines = [
    "# Barrido de prob_thresh — Villas 273-300",
    "",
    f"Modelo StarDist v1 · {rep['n_frames']} frames · NMS {rep['nms']} · "
    f"mapeo real de `meta.json`.",
    "",
    "| prob_thresh | Polígonos | Δ vs baseline |",
    "|---:|---:|---:|",
]
for p in probs:
    lines.append(f"| {p:.4f} | {totals[p]} | {totals[p]-baseline:+d} |")

lines += [
    "",
    "## Por frame",
    "",
    "| Frame | " + " | ".join(f"{p:.2f}" for p in probs) + " |",
    "|---:|" + "---:|" * len(probs),
]
for f in range(273, 301):
    vals = [rep["per_prob"][f"{p:.4f}"][str(f)]["n_poligonos"] for p in probs]
    lines.append(f"| {f} | " + " | ".join(str(v) for v in vals) + " |")

lines += [
    "",
    "Es un conteo de polígonos, **no una evaluación contra oro**. "
    "La serie conviene leerla junto al barrido con gold en `docs/PROB_THRESH_SWEEP.md`, "
    "donde bajar el umbral reducía FN pero subía FP y empeoraba F1.",
    "",
]

cols, rows_n = len(SHOW), len(probs)
fig, axs = plt.subplots(rows_n, cols, figsize=(15, 2.9 * rows_n))
for r, p in enumerate(probs):
    tag = f"{p:.4f}"
    for c, f in enumerate(SHOW):
        ax = axs[r, c]
        name = rep["per_prob"][tag][str(f)]["image"]
        path = os.path.join(SRC, tag, f"ov_{name}")
        if os.path.exists(path):
            ax.imshow(Image.open(path))
            ax.set_title(f"prob {p:.2f} · frame {f}", fontsize=8)
        ax.axis("off")
fig.suptitle("Villas 273-300 — umbral de probabilidad (NMS 0.02)", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(SRC, "sweep_overlays.png"), dpi=105)
plt.close(fig)

with open(DOC, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("escrito", DOC)
print("totales", {f"{p:.2f}": totals[p] for p in probs})
