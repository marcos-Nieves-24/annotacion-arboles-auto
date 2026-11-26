"""Build texture-classifier patches v2 — RAW images, no texture fill.

OPTION 2: positives = square crop centered on the crown WITH its real
surroundings (soil, shadows, neighbor crowns). Negatives = real soil patches.
Both classes come from real imagery, so there is no color shortcut.

Same frame-level split (seed 7), dev/holdout excluded from train.
Output: data/fase4/patches_v2/{pos,neg}/{train,val}/*.jpg + manifest.json
"""

import json, os, glob, random, re
import numpy as np, cv2

ROOT = '/home/biohacker24/annotacion-arboles-auto'
OUT = f'{ROOT}/data/fase4/patches_v2'
os.makedirs(OUT, exist_ok=True)
SEED = 7
SIZE = 640

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
dev_ids = set(D['A'] + D['B'])
byid = {i['id']: i['file_name'] for i in C['images']}
dev_names = {byid[i] for i in dev_ids if i in byid}

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
    pts = np.array(ann['segmentation'][0], float).reshape(-1, 2).astype(int)
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def raw_crop_centered(img, mask):
    """Square crop around crown, WITH real surroundings, no fill."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = int(xs.mean()), int(ys.mean())
    side = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    side = int(side * 1.8)  # more context than v1 so real soil is included
    side = max(side, 128)
    x0, y0 = cx - side // 2, cy - side // 2
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(512, x0 + side), min(512, y0 + side)
    sub = img[y0c:y1c, x0c:x1c]
    if sub.size == 0:
        return None
    sub = cv2.copyMakeBorder(sub, pad_t, max(0, y0 + side - 512),
                             pad_l, max(0, x0 + side - 512), cv2.BORDER_REFLECT)
    return cv2.resize(sub, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


# deterministic frame-level split
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
    build_frame_split()
    pos = {'train': 0, 'val': 0, 'holdout': 0}
    neg = {'train': 0, 'val': 0, 'holdout': 0}
    # POSITIVES: raw crop with real context
    for a in C['annotations']:
        if a['iscrowd'] or not a.get('segmentation'):
            continue
        fn = byid[a['image_id']]
        img = load_img(fn)
        if img is None:
            continue
        patch = raw_crop_centered(img, mask_of(a))
        if patch is None:
            continue
        sp = split_frame(fn)
        d = f'{OUT}/pos/{sp}'
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f'{d}/{fn.rsplit(".",1)[0]}_{a["id"]}.jpg', patch,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        pos[sp] += 1

    # NEGATIVES: real soil/background patches (same as v1)
    crowned = set(a['image_id'] for a in C['annotations'] if not a['iscrowd'])
    no_crown = [i for i in C['images'] if i['id'] not in crowned]
    rng = random.Random(SEED + 1)
    target = int(sum(pos.values()) * 0.8)
    made = tries = 0
    while made < target and tries < target * 10:
        tries += 1
        info = rng.choice(no_crown) if (made % 2 == 0 and no_crown) else rng.choice(C['images'])
        fn = info['file_name']
        img = load_img(fn)
        if img is None:
            continue
        side = rng.randint(80, 180)
        x0 = rng.randint(0, 512 - side - 1)
        y0 = rng.randint(0, 512 - side - 1)
        patch = cv2.resize(img[y0:y0+side, x0:x0+side], (SIZE, SIZE),
                           interpolation=cv2.INTER_LINEAR)
        sp = split_frame(fn)
        if sp == 'holdout':
            continue
        d = f'{OUT}/neg/{sp}'
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f'{d}/{fn.rsplit(".",1)[0]}_{made}.jpg', patch,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        neg[sp] += 1
        made += 1

    manifest = {'seed': SEED, 'size': SIZE, 'protocol': 'raw crops (no fill) - option 2',
                'positive': pos, 'negative': neg,
                'dev_frames_excluded_from_train': len(dev_names),
                'holdout_excluded': len(hold)}
    json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
