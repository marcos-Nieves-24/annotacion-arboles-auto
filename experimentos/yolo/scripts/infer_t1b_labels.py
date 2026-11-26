"""Inference with YOLO T1b -> YOLO-seg labels + overlays + contact sheets.

Runs on a configurable frame range of a farm (default Villas 270-400).
Writes local files ONLY (labels/, overlays/, contact sheets, report.json).
No CVAT writes. Review output before any upload.
"""

import json, os, argparse
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
W = '/tmp/user/1000/opencode/wm/model_final.pt'
CONF, IOU = 0.25, 0.35
BATCH_CONTACT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--farm', default='villas', choices=['villas', 'magdala'])
    ap.add_argument('--start', type=int, default=270)
    ap.add_argument('--end', type=int, default=400)
    ap.add_argument('--out', default='data/fase4/infer_t1b_villas270_400')
    a = ap.parse_args()

    OUT = os.path.join(ROOT, a.out)
    os.makedirs(f'{OUT}/labels', exist_ok=True)
    os.makedirs(f'{OUT}/overlays', exist_ok=True)

    meta = json.load(open(f'{ROOT}/data/raw/{a.farm}/meta.json'))
    names = [f['name'] for f in meta['frames']]
    IMG = f'{ROOT}/data/r1/images/{a.farm}'
    frames = [(i, names[i]) for i in range(a.start, a.end + 1) if i < len(names)]

    model = YOLO(W)
    report = []
    for idx, fn in frames:
        p = os.path.join(IMG, fn)
        if not os.path.exists(p):
            continue
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        r = model.predict(img, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
        nb = len(r.boxes) if r.boxes is not None else 0

        # YOLO-seg label: "0 x1 y1 x2 y2 ..." normalized to 512
        lines = []
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if r.masks is not None and len(r.masks):
            md = r.masks.data.float().cpu().numpy()
            for k in range(md.shape[0]):
                m = md[k] > 0.5
                if m.shape != (512, 512):
                    m = cv2.resize(m.astype('uint8'), (512, 512)) > 0.5
                cnts, _ = cv2.findContours(m.astype('uint8'), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea)
                ap_e = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, ap_e, True).ravel()
                if len(approx) < 6:
                    continue
                coords = [f'{v/512:.6f}' for v in approx]
                lines.append('0 ' + ' '.join(coords))
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        # write label even if empty (so every frame has a file)
        stem = fn.rsplit('.', 1)[0]
        open(f'{OUT}/labels/{stem}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        cv2.imwrite(f'{OUT}/overlays/{fn}', vis)
        report.append({'frame_idx': idx, 'file': fn, 'n_det': nb, 'n_poly': len(lines)})

    # contact sheets in batches
    for bi in range(0, len(report), BATCH_CONTACT):
        chunk = report[bi:bi + BATCH_CONTACT]
        tw = th = 200
        cols = 6
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * th, cols * tw, 3), 40, np.uint8)
        for k, r in enumerate(chunk):
            im = cv2.imread(f"{OUT}/overlays/{r['file']}")
            if im is None:
                continue
            im = cv2.resize(im, (tw, th))
            cv2.putText(im, f"{r['file'][-9:-4]} n{r['n_det']}", (2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            sheet[(k//cols)*th:(k//cols+1)*th, (k%cols)*tw:(k%cols+1)*tw] = im
        cv2.imwrite(f'{OUT}/contact_{bi//BATCH_CONTACT+1:02d}.jpg', sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    counts = [r['n_det'] for r in report]
    summary = {'config': {'model': 'yolo11n-seg T1b', 'conf': CONF, 'iou': IOU,
                          'imgsz': 640, 'farm': a.farm, 'range': [a.start, a.end]},
               'n_frames': len(report), 'n_det': int(sum(counts)),
               'mean': round(float(np.mean(counts)), 2) if counts else 0,
               'min': min(counts) if counts else 0, 'max': max(counts) if counts else 0,
               'zero_det_frames': sum(1 for c in counts if c == 0),
               'per_frame': report}
    json.dump(summary, open(f'{OUT}/report.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=1))
    print('labels:', len(os.listdir(f'{OUT}/labels')), 'overlays:', len(os.listdir(f'{OUT}/overlays')),
          'contacts:', len([f for f in os.listdir(OUT) if f.startswith('contact_')]))


if __name__ == '__main__':
    main()
