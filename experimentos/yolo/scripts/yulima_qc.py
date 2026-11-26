"""Fase 3 QC Yulima: splitB predictions vs CVAT auto annotations on 40 frames.

Read-only vs CVAT (frames + annotations already downloaded). No uploads.
Verdict thresholds: median matched IoU > 0.85 AND >=80% frames count ±10%.
"""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = "/home/biohacker24/annotacion-arboles-auto"
IMGDIR = f"{ROOT}/data/gold_v1/yulima_qc"


def rasterize(points, w=512, h=512):
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    sample = json.load(open(f"{ROOT}/data/gold_v1/yulima_qc40.json"))
    ann = json.load(open(f"{ROOT}/data/raw/yulima/annotations.json"))
    auto_by_frame = {}
    for s in ann.get("shapes", []):
        if s.get("type") == "polygon":
            auto_by_frame.setdefault(s["frame"], []).append(s["points"])

    model = YOLO(f"{ROOT}/data/kaggle-out-fase2/runs/splitB/weights/best.pt")
    paths = [os.path.join(IMGDIR, s["file"]) for s in sample]
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, f"missing frames (download running?): {missing[:3]}"
    res = model.predict(paths, imgsz=640, conf=0.25, verbose=False)

    per_frame, all_ious = [], []
    for r, s in zip(res, sample):
        auto_masks = [rasterize(p) for p in auto_by_frame.get(s["frame"], [])]
        pred_masks = []
        if r.masks is not None and len(r.masks):
            import torch.nn.functional as F
            import torch

            md = r.masks.data.float()
            if md.shape[1] != 512 or md.shape[2] != 512:
                md = F.interpolate(md.unsqueeze(0), size=(512, 512),
                                   mode="nearest").squeeze(0)
            pred_masks = [(md[k] > 0.5).cpu().numpy().astype("uint8")
                          for k in range(md.shape[0])]
        # greedy matching by best IoU
        used, ious, tp = set(), [], 0
        for pm in pred_masks:
            best, bj = 0.0, -1
            for j, am in enumerate(auto_masks):
                if j in used:
                    continue
                v = mask_iou(pm, am)
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                tp += 1
                used.add(bj)
                ious.append(best)
                all_ious.extend([best])
        prec = tp / max(len(pred_masks), 1)
        rec = tp / max(len(auto_masks), 1)
        per_frame.append({"file": s["file"], "band": s["band"],
                          "n_auto": len(auto_masks), "n_pred": len(pred_masks),
                          "tp": tp, "precision": round(prec, 3),
                          "recall": round(rec, 3),
                          "mean_iou": round(float(np.mean(ious)), 3) if ious else 0.0})

    med_iou = float(np.median(all_ious)) if all_ious else 0.0
    in_tol = sum(1 for p in per_frame
                 if abs(p["n_pred"] - p["n_auto"]) <= max(1, 0.1 * p["n_auto"]))
    verdict = {"median_matched_iou": round(med_iou, 3),
               "frames_count_in_tol": f"{in_tol}/{len(per_frame)}",
               "mean_precision": round(float(np.mean([p['precision'] for p in per_frame])), 3),
               "mean_recall": round(float(np.mean([p['recall'] for p in per_frame])), 3),
               "silver": bool(med_iou > 0.85 and in_tol / len(per_frame) >= 0.8)}
    out = {"verdict": verdict, "per_frame": per_frame}
    json.dump(out, open(f"{ROOT}/data/gold_v1/yulima_qc_report.json", "w"), indent=1)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
