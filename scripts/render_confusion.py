#!/usr/bin/env python3
"""Render confusion matrices from the StarDist-vs-gold_v2 evaluation report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_confusion", "stardist_confusion.json")
OUTDIR = os.path.join(ROOT, "data", "stardist_confusion")
DOC = os.path.join(ROOT, "docs", "MATRIZ_CONFUSION_STARDUST.md")

report = json.load(open(SRC))
buckets_all = report["buckets"]

ORDER = [
    ("dev", "Dev (no visto)"),
    ("holdout", "Holdout (no visto)"),
    ("farm:LasVillas", "LasVillas (todas)"),
    ("farm:Magdala", "Magdala (todas)"),
    ("LasVillas/dev", "LasVillas / dev"),
    ("LasVillas/holdout", "LasVillas / holdout"),
    ("Magdala/dev", "Magdala / dev"),
    ("Magdala/holdout", "Magdala / holdout"),
    ("train", "Train (visto)"),
    ("ALL", "Todas"),
]


def matrix(b):
    return np.array([[b["tp_px"], b["fp_px"]],
                     [b["fn_px"], b["tn_px"]]], dtype=float)


def draw(mat, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1))
    ax.set_xticks([0, 1], ["pred copa", "pred fondo"])
    ax.set_yticks([0, 1], ["real copa", "real fondo"])
    ax.set_title(title, fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center",
                    fontsize=10, color="white" if mat[i, j] > mat.max() * 0.05 else "black")
    fig.colorbar(im, ax=ax, label="pixeles (escala log)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


lines = [
    "# Matriz de confusión — StarDist vs gold_v2",
    "",
    f"Modelo: `{report['model']}` · dataset: `{report['dataset']}` · "
    f"`prob_thresh={report['prob_thresh']:.4f}` · matching de instancias IoU ≥ {report['iou_match_threshold']}.",
    "",
    f"Imágenes: {report['n_images']} ({', '.join(report['farms'])}). "
    f"Splits: {report['splits']}.",
    "",
    "**Matriz de píxeles**: filas = clase real, columnas = clase predicha. "
    "Se construye con la unión de máscaras de copa por imagen (fondo = el resto del tile 512×512). "
    "**Instancias**: TP/FP/FN por matching greedy IoU ≥ 0.5.",
    "",
    "> Ojo: `train` son imágenes vistas durante el entrenamiento; "
    "`dev`/`holdout` son las honestas. Las filas `farm:*` mezclan todos los splits.",
    "",
]

for nms_key in sorted(buckets_all):
    bs = buckets_all[nms_key]
    lines += [f"## {nms_key}", ""]
    lines += [
        "| Grupo | TP | FP | FN | Precision | Recall | F1 | IoU píxel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in ORDER:
        if key not in bs:
            continue
        b = bs[key]
        lines.append(
            f"| {label} | {b['tp']} | {b['fp']} | {b['fn']} | "
            f"{b['precision']:.4f} | {b['recall']:.4f} | {b['F1']:.4f} | {b['iou_px']:.4f} |"
        )
    lines += [""]

    lines += [
        "| Grupo | Matriz de píxeles (TP/FP · FN/TN) |",
        "|---|---|",
    ]
    for key, label in ORDER[:8]:
        if key not in bs:
            continue
        m = matrix(bs[key])
        lines.append(
            f"| {label} | `{int(m[0,0]):,}` / `{int(m[0,1]):,}` · "
            f"`{int(m[1,0]):,}` / `{int(m[1,1]):,}` |"
        )
        draw(m, f"{nms_key} — {label}",
             os.path.join(OUTDIR, f"matriz_{nms_key}_{key.replace(':', '').replace('/', '_')}.png"))
    lines += [""]

with open(DOC, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("escrito", DOC)
print("PNG", len([f for f in os.listdir(OUTDIR) if f.startswith("matriz_")]))
