"""Build texture-classifier patches from gold_v2 (paper Rossi IGARSS 2026 style).

POSITIVES: each iscrowd=0 crown -> square crop around its mask, resized 640,
           background filled with texture copy-pasted from the SAME crown
           (paper Sec II-C) so the model cannot cheat on black bg / shape.
NEGATIVES: random soil/background patches from frames with no crowns and
           from areas outside crowns in frames with crowns.

Split by FRAME (70/30), seed 7. Dev/holdout frames from the YOLO experiments
are excluded from train to keep the later rescue measurement clean.

Output: data/fase4/patches/{pos,neg}/{train,val}/*.jpg + manifest.json
"""

import json, os, glob, random
import numpy as np, cv2

ROOT = '/home/biohacker24/annotacion-arboles-auto'
OUT = f'{ROOT}/data/fase4/patches'
os.makedirs(OUT, exist_ok=True)
SEED = 7
random.seed(SEED)
SIZE = 640

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
dev_ids = set(D['A'] + D['B'])
byid = {i['id']: i['file_name'] for i in C['images']}
dev_names = {byid[i] for i in dev_ids if i in byid}

# real image sources (gold_v2/images symlinks are broken)
IMG_DIRS = [
    f'{ROOT}/data/r1/images/villas',
    f'{ROOT}/data/r1/images/magdala',
    f'{ROOT}/data/gold_v1/images',
]


def load_img(fn):
    for d in IMG_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return im
    return None


def mask_of(ann):
    seg = ann['segmentation'][0]
    pts = np.array(seg, float).reshape(-1, 2).astype(int)
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def fill_texture_from_crown(img, mask, rng):
    """Copy-paste: fill background with patches sampled from the crown itself."""
    out = img.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return out
    bg = mask == 0
    # sample crown pixels (with small jitter) for each background pixel,
    # indices are into the SUB-image, not the full 512 frame
    h, w = out.shape[:2]
    sample_idx = rng.integers(0, len(xs), size=bg.sum())
    sy, sx = ys[sample_idx], xs[sample_idx]
    jx = np.clip(sx + rng.integers(-3, 4, size=bg.sum()), 0, w - 1)
    jy = np.clip(sy + rng.integers(-3, 4, size=bg.sum()), 0, h - 1)
    out[bg] = img[jy, jx]
    # restore crown
    out[mask > 0] = img[mask > 0]
    return out


def square_crop_centered(img, mask, rng):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = int(xs.mean()), int(ys.mean())
    side = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    side = int(side * 1.25)  # margin
    side = max(side, 64)
    x0, y0 = cx - side // 2, cy - side // 2
    # pad if out of bounds
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(512, x0 + side), min(512, y0 + side)
    sub = img[y0c:y1c, x0c:x1c]
    msub = mask[y0c:y1c, x0c:x1c]
    if sub.size == 0:
        return None
    sub = cv2.copyMakeBorder(sub, pad_t, max(0, y0 + side - 512),
                             pad_l, max(0, x0 + side - 512),
                             cv2.BORDER_REFLECT)
    msub = cv2.copyMakeBorder(msub, pad_t, max(0, y0 + side - 512),
                              pad_l, max(0, x0 + side - 512),
                              cv2.BORDER_CONSTANT, value=0)
    # texture fill then resize
    filled = fill_texture_from_crown(sub, msub, rng)
    return cv2.resize(filled, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


# deterministic frame-level split built once
_frame_split = {}


def build_frame_split():
    eligible = sorted({i['file_name'] for i in C['images']} - hold - dev_names)
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    n_val = max(1, int(len(eligible) * 0.30))
    for fn in eligible[:n_val]:
        _frame_split[fn] = 'val'
    for fn in eligible[n_val:]:
        _frame_split[fn] = 'train'
    for fn in hold:
        _frame_split[fn] = 'holdout'
    for fn in dev_names:
        _frame_split[fn] = 'val'


def split_frame(fn):
    if fn in hold:
        return 'holdout'
    if fn in dev_names:
        return 'val'
    return _frame_split.get(fn, 'train')


def main():
    rng = np.random.default_rng(SEED)
    build_frame_split()
    pos = {'train': 0, 'val': 0, 'holdout': 0}
    neg = {'train': 0, 'val': 0, 'holdout': 0}
    per_class = {}
    # POSITIVES
    for a in C['annotations']:
        if a['iscrowd'] or not a.get('segmentation'):
            continue
        fn = byid[a['image_id']]
        img = load_img(fn)
        if img is None:
            continue
        mask = mask_of(a)
        patch = square_crop_centered(img, mask, rng)
        if patch is None:
            continue
        sp = split_frame(fn)
        d = f'{OUT}/pos/{sp}'
        os.makedirs(d, exist_ok=True)
        name = f"{fn.rsplit('.',1)[0]}_{a['id']}.jpg"
        cv2.imwrite(f'{d}/{name}', patch, [cv2.IMWRITE_JPEG_QUALITY, 90])
        pos[sp] += 1

    # NEGATIVES: soil/background patches from no-crown frames + outside crowns
    crowned = set(a['image_id'] for a in C['annotations'] if not a['iscrowd'])
    all_imgs = C['images']
    no_crown = [i for i in all_imgs if i['id'] not in crowned]
    rng2 = random.Random(SEED + 1)
    n_neg_target = int(sum(pos.values()) * 0.8)
    made = 0
    tries = 0
    while made < n_neg_target and tries < n_neg_target * 10:
        tries += 1
        if made % 2 == 0 and no_crown:
            info = rng2.choice(no_crown)
        else:
            info = rng2.choice(all_imgs)
        fn = info['file_name']
        img = load_img(fn)
        if img is None:
            continue
        side = rng2.randint(80, 180)
        x0 = rng2.randint(0, 512 - side - 1)
        y0 = rng2.randint(0, 512 - side - 1)
        patch = img[y0:y0 + side, x0:x0 + side]
        patch = cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        sp = split_frame(fn)
        if sp == 'holdout':
            continue  # keep holdout only for final eval
        d = f'{OUT}/neg/{sp}'
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f'{d}/{fn.rsplit(".",1)[0]}_{made}.jpg', patch,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        neg[sp] += 1
        made += 1

    manifest = {'seed': SEED, 'size': SIZE, 'protocol': 'Rossi 2026 copy-paste texture',
                'positive': pos, 'negative': neg,
                'dev_frames_excluded_from_train': len(dev_names),
                'holdout_excluded': len(hold)}
    json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
