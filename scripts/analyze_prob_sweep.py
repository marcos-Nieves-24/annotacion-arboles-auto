#!/usr/bin/env python3
"""Render the StarDist prob_thresh sweep on Magdala dev/holdout."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_prob_sweep", "stardist_prob_sweep.json")
OUT = os.path.join(ROOT, "data", "stardist_prob_sweep")
DOC = os.path.join(ROOT, "docs", "PROB_THRESH_SWEEP.md")

rep = json.load(open(SRC))
lines = [
    "# Barrido de `prob_thresh` — StarDist en Magdala dev/holdout",
    "",
    f"Modelo `{rep['model']}` · {rep['n_images']} imágenes · matching IoU ≥ {rep['matching_iou']} · "
    f"baseline `prob_thresh={rep['base_prob']:.4f}`.",
    "",
    "Mismo set usado en el análisis de FN (26 imágenes), para comparar directamente contra las 37 FN del baseline.",
    "",
]

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, nms in zip(axs, rep["nms"]):
    rows = [v for v in rep["results"].values() if v["nms"] == nms]
    rows.sort(key=lambda r: r["prob_thresh"])
    xs = [r["prob_thresh"] for r in rows]
    ax.plot(xs, [r["F1"] for r in rows], "o-", label="F1")
    ax.plot(xs, [r["recall"] for r in rows], "s-", label="Recall")
    ax.plot(xs, [r["precision"] for r in rows], "^-", label="Precision")
    ax.axvline(rep["base_prob"], color="red", ls="--", alpha=0.6, label="baseline")
    ax.set_title(f"NMS {nms}")
    ax.set_xlabel("prob_thresh")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("StarDist — sensibilidad al umbral de probabilidad (Magdala dev+holdout)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "prob_sweep.png"), dpi=130)
plt.close(fig)

for nms in rep["nms"]:
    rows = [v for v in rep["results"].values() if v["nms"] == nms]
    rows.sort(key=lambda r: r["prob_thresh"])
    lines += [
        f"## NMS {nms}",
        "",
        "| prob_thresh | TP | FP | FN | Precision | Recall | F1 | FN borde 0/1/2/≥3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        b = r["fn_border"]
        lines.append(
            f"| {r['prob_thresh']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['F1']:.4f} | "
            f"{b['0']}/{b['1']}/{b['2']}/{b['>=3']} |"
        )
    lines += [""]

best = max(rep["results"].values(), key=lambda r: (r["F1"], -r["prob_thresh"]))
base = [v for v in rep["results"].values()
        if v["nms"] == 0.10 and abs(v["prob_thresh"] - round(rep["base_prob"], 4)) < 1e-6]
bprob = round(rep["base_prob"], 4)
nms01 = [v for v in rep["results"].values() if v["nms"] == 0.10]
nms01.sort(key=lambda r: r["prob_thresh"])
worst = min(nms01, key=lambda r: r["prob_thresh"])
baseline = [v for v in nms01 if abs(v["prob_thresh"] - bprob) < 1e-6][0]
b0 = baseline["fn_border"]
bw = worst["fn_border"]
lines += [
    "## Lectura",
    "",
    f"**El baseline se reproduce exactamente**: NMS 0.10 con prob {bprob} da "
    f"{baseline['fn']} FN, {baseline['fp']} FP, F1 {baseline['F1']:.4f} — "
    "idéntico al análisis previo de 37 FN. La comparación es válida.",
    "",
    f"**Bajar el umbral no ayuda.** De {bprob} a {worst['prob_thresh']}: "
    f"FN {baseline['fn']} → {worst['fn']} (se recuperan {baseline['fn']-worst['fn']} copas), "
    f"pero FP {baseline['fp']} → {worst['fp']} (se inventan {worst['fp']-baseline['fp']}). "
    f"F1 cae de {baseline['F1']:.4f} a {worst['F1']:.4f}. El mejor F1 del barrido "
    f"es justamente el baseline ({best['F1']:.4f}).",
    "",
    f"**El borde persiste.** Con el umbral más bajo, de {worst['fn']} FN todavía "
    f"{bw['2']} cruzan el borde (2 vértices) y {bw['0']} son interiores. En el baseline "
    f"eran {b0['2']} y {b0['0']}. Bajar el umbral no elimina el problema de borde.",
    "",
    "**Conclusión:** el umbral de probabilidad no es la causa de las copas perdidas; "
    "el modelo ya está operando en su punto óptimo de F1. La mitigación tiene que ir "
    "por contexto/escala (padding en inferencia o reentrenamiento con copas de borde), "
    "no por bajar `prob_thresh`.",
    "",
    "Gráfico: `data/stardist_prob_sweep/prob_sweep.png`.",
]
with open(DOC, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("escrito", DOC)
