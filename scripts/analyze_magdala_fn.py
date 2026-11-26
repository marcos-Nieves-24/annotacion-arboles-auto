#!/usr/bin/env python3
"""Analyze missed Magdala crowns from the StarDist FN report."""
import json, os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "stardist_magdala_fn", "magdala_fn_report.json")
OUTDIR = os.path.join(ROOT, "data", "stardist_magdala_fn")
DOC = os.path.join(ROOT, "docs", "MAGDALA_FN_ANALISIS.md")

report = json.load(open(SRC))
items = report["items"]
lines = [
    "# Magdala — copas que se escapan a StarDist",
    "",
    f"Modelo `{report['model']}` · `prob_thresh={report['prob_thresh']:.4f}` · "
    f"matching IoU ≥ {report['matching_iou']} · "
    f"{report['images']} imágenes Magdala de dev/holdout.",
    "",
    "**Sin escritura en CVAT.** Los FN son copas del oro que StarDist no alcanzó a detectar.",
    "",
]

def pct(part, total):
    return f"{100*part/max(total,1):.1f}%"

for nms in (0.02, 0.1):
    sel = [i for i in items if i["nms"] == nms]
    st = report["nms"][str(nms)]
    by_kind = Counter(i["kind"] for i in sel)
    edge = sum(i["edge_touch"] for i in sel)
    areas = np.array([i["area_px"] for i in sel], dtype=float)
    ious = np.array([i["best_iou"] for i in sel], dtype=float)
    dists = [i["nearest_pred_dist"] for i in sel if i["nearest_pred_dist"] is not None]

    lines += [
        f"## NMS {nms}",
        "",
        f"TP {st['tp']} · FP {st['fp']} · **FN {st['fn']}**",
        "",
        "| Diagnóstico | Valor |",
        "|---|---:|",
        f"| No detectadas (`best IoU < 0.1`) | {by_kind['not_detected']} ({pct(by_kind['not_detected'], len(sel))}) |",
        f"| Sub-umbral (`0.1 ≤ best IoU < 0.5`) | {by_kind['sub_threshold']} ({pct(by_kind['sub_threshold'], len(sel))}) |",
        f"| Tocan borde del tile (≤8 px) | {edge} ({pct(edge, len(sel))}) |",
        f"| Área mediana (px) | {int(np.median(areas)) if len(areas) else 0} |",
        f"| Área p25 / p75 | {int(np.percentile(areas,25)) if len(areas) else 0} / {int(np.percentile(areas,75)) if len(areas) else 0} |",
        f"| Mejor IoU mediano | {np.median(ious):.3f} |",
        f"| Distancia mediana a la predicción más cercana (px) | {np.median(dists):.1f} |" if dists else "| Distancia a predicción | n/a |",
        "",
    ]

    cedge = Counter()
    for i in sel:
        x0, y0, x1, y1 = i["bbox"]
        for flag, cond in (("arriba", y0 <= 8), ("abajo", y1 >= 503),
                           ("izquierda", x0 <= 8), ("derecha", x1 >= 503)):
            if cond:
                cedge[flag] += 1
        if not (y0 <= 8 or y1 >= 503 or x0 <= 8 or x1 >= 503):
            cedge["centro del tile"] += 1
    lines += ["| Borde del tile | Copas perdidas |", "|---|---:|"]
    for k in ("arriba", "abajo", "izquierda", "derecha", "centro del tile"):
        lines.append(f"| {k} | {cedge[k]} ({pct(cedge[k], len(sel))}) |")
    lines += [""]

    # Mapa de calor de centroides dentro del tile
    xs = [i["centroid"][0] for i in sel if i["centroid"][0] is not None]
    ys = [i["centroid"][1] for i in sel if i["centroid"][1] is not None]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if xs:
        h, xe, ye = np.histogram2d(xs, ys, bins=8, range=[[0, 512], [0, 512]])
        ax.imshow(h.T, origin="lower", cmap="hot", extent=[0, 512, 512, 0])
        ax.set_title(f"FN Magdala NMS {nms} — {len(xs)} centroides")
        ax.set_xlabel("x (px en el tile)")
        ax.set_ylabel("y (px en el tile)")
        for i in sel:
            if i["centroid"][0] is not None:
                ax.plot(i["centroid"][0], i["centroid"][1], "w+", ms=4, alpha=0.5)
        fig.colorbar(ax.images[0], ax=ax, label="copas perdidas")
    fig.tight_layout()
    heat = os.path.join(OUTDIR, f"fn_heatmap_nms{nms}.png")
    fig.savefig(heat, dpi=130)
    plt.close(fig)

    top_img = Counter(i["image"] for i in sel).most_common(8)
    lines += ["| Imagen | split | FN |", "|---|---|---:|"]
    for img, cnt in top_img:
        sp = next(i["split"] for i in sel if i["image"] == img)
        lines.append(f"| `{img}` | {sp} | {cnt} |")
    lines += [
        "",
        "| Imagen | split | ann_id | centroide (x,y) | área px | mejor IoU | tipo | borde |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for i in sorted(sel, key=lambda z: (-z["best_iou"], z["image"])):
        cx, cy = i["centroid"]
        lines.append(
            f"| `{i['image']}` | {i['split']} | {i['ann_id']} | ({cx}, {cy}) | "
            f"{i['area_px']} | {i['best_iou']:.3f} | {i['kind']} | {'sí' if i['edge_touch'] else 'no'} |"
        )
    lines += [""]

lines += [
    "## Lectura",
    "",
    "- `no detectada`: no hay ninguna predicción con IoU ≥ 0.1 → el modelo no vio la copa.",
    "- `sub_threshold`: existe una predicción solapada pero no llega a IoU 0.5 → forma/recorte incorrecto.",
    "- **Ubicación**: la mayoría de las pérdidas está en el borde del tile (arriba/abajo/derecha), "
    "no en el centro. El mapa de calor `fn_heatmap_nms*.png` lo muestra: franjas densas en y≈0 e y≈500.",
    "- El `best IoU` mediano es ~0 y la distancia a la predicción más cercana es ~180 px: "
    "no son copas mal recortadas ni fusionadas con vecinas, **son copas que StarDist no detecta**.",
    "- El área mediana de las perdidas (~1500 px) descarta que sean solo copas chicas.",
    "- Los overlays `fn_nms*_*.png` marcan en rojo cada FN con su `ann_id`; verde = oro, "
    "amarillo = TP, azul = predicción no match.",
    "",
]

with open(DOC, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("escrito", DOC)
print("heatmaps", len([f for f in os.listdir(OUTDIR) if f.startswith("fn_heatmap")]))
