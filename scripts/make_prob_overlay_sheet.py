#!/usr/bin/env python3
"""Assemble the per-prob_thresh overlays into one comparison sheet."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_prob_overlays")
PROBS = ["0.3000", "0.3500", "0.4000", "0.4500", "0.5000", "0.5500", "0.5929"]
NAMES = ["Limon_Magdala_00946.png", "Limon_Magdala_01027.png",
         "Limon_Magdala_00866.png", "Limon_Magdala_01012.png",
         "Limon_Magdala_01042.png", "Limon_Magdala_01043.png"]

summary = json.load(open(os.path.join(SRC, "prob_overlays_summary.json")))
fig, axs = plt.subplots(len(PROBS), len(NAMES), figsize=(18, 21))
for r, prob in enumerate(PROBS):
    for c, name in enumerate(NAMES):
        ax = axs[r, c]
        p = os.path.join(SRC, f"ov_prob{prob}_{name}")
        if os.path.exists(p):
            ax.imshow(Image.open(p))
            s = summary[prob][name]
            ax.set_title(f"prob {prob}\n{name[-9:-4]} · FN {s['fn']} · FP {s['fp']}",
                         fontsize=7)
        ax.axis("off")
fig.suptitle("StarDist — umbral de probabilidad (NMS 0.10) · rojo=FN, amarillo=TP, azul=FP, verde=oro",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(SRC, "overlays_por_prob.png"), dpi=105)
plt.close(fig)
print("hoja:", os.path.join(SRC, "overlays_por_prob.png"))
