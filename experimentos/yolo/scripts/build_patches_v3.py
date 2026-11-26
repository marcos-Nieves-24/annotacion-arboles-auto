"""Build texture-classifier patches v3 — RAW + hard negatives (non-citrus vegetation).

Same positives and soil negatives as v2, PLUS:
  NEG-VEG: green patches from frames with NO citrus crowns (shrubs/weeds/brush)
           sampled where ExG>threshold, so the model learns "green != always citrus".
  NEG-PARTIAL: partial/occluded citrus crowns at tile borders (hard).

Split by frame (seed 7). Output: data/fase4/patches_v3/
"""

import json, os, random
import numpy as np, cv2

ROOT = '/home/biohacker24/annotacion-arboles-auto'
OUT = f'{ROOT}/data/fase4/patches_v3'
SEED = 7
SIZE = 640

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
dev_names = {C['images'][i]['file_name'] for i in D['A'] + D['B'] if i < len(C['images'])}
byid = {i['id']: i['file_name'] for i in C['images']}
IMG_DIRS = [f'{ROOT}/data/r1/images/villas', f'{ROOT}/data/r1/images/magdala',
            f'{ROOT}/data/gold_v1/images']


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
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    side = int(max(xs.max()-xs.min(), ys.max()-ys.min()) * 1.8)
    side = max(side, 128)
    x0, y0 = int(xs.mean()) - side//2, int(ys.mean()) - side//2
    pl, pt = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(512, x0+side), min(512, y0+side)
    sub = img[y0c:y1c, x0c:x1c]
    if sub.size == 0:
        return None
    sub = cv2.copyMakeBorder(sub, pt, max(0, y0+side-512), pl,
                             max(0, x0+side-512), cv2.BORDER_REFLECT)
    return cv2.resize(sub, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


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


def save_patch(kind, sp, fn, uid, patch):
    d = f'{OUT}/{kind}/{sp}'
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f'{d}/{fn.rsplit(".",1)[0]}_{uid}.jpg', patch,
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    build_frame_split()
    counts = {k: {'train': 0, 'val': 0} for k in ('pos', 'neg_soil', 'neg_veg', 'neg_partial')}

    # POSITIVES
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
        if sp == 'holdout':
            continue
        save_patch('pos', sp, fn, a['id'], patch)
        counts['pos'][sp] += 1

    # frames with no citrus crowns, sorted by green fraction
    crowned = set(a['image_id'] for a in C['annotations'] if not a['iscrowd'])
    no_crown = [i['file_name'] for i in C['images'] if i['id'] not in crowned]
    green_frames = []
    for fn in no_crown:
        img = load_img(fn)
        if img is None:
            continue
        f = img.astype(np.int16)
        exg = 2*f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
        green_frames.append((fn, float((exg > 15).mean())))
    green_frames.sort(key=lambda x: -x[1])

    rng = random.Random(SEED + 1)
    n_pos = sum(counts['pos'].values())
    target_each = int(n_pos * 0.45)

    # NEG SOIL (as v2): random patches incl. low-green frames
    low_green = [fn for fn, g in green_frames if g < 0.3] or no_crown
    made = 0
    while made < target_each:
        fn = rng.choice(low_green)
        img = load_img(fn)
        if img is None:
            continue
        side = rng.randint(80, 180)
        x0, y0 = rng.randint(0, 512-side-1), rng.randint(0, 512-side-1)
        patch = cv2.resize(img[y0:y0+side, x0:x0+side], (SIZE, SIZE))
        sp = split_frame(fn)
        if sp == 'holdout':
            continue
        save_patch('neg_soil', sp, fn, made, patch)
        counts['neg_soil'][sp] += 1
        made += 1

    # NEG VEG: green patches from high-green no-crown frames (the key new negatives)
    EXCLUDE_VEG = {'Limon_Magdala_00700.png', 'Limon_Magdala_00701.png',
                   'Limon_Magdala_00673.png', 'Limon_Magdala_00622.png',
                   'Limon_Magdala_00675.png', 'Limon_Magdala_00674.png',
                   'Limon_Magdala_00672.png', 'Limon_Magdala_00710.png',
                   'Limon_Magdala_00713.png'}
    high_green = [fn for fn, g in green_frames if g > 0.5 and fn not in EXCLUDE_VEG]
    print('high-green no-crown frames:', len(high_green))
    made = 0
    while made < target_each and high_green:
        fn = rng.choice(high_green)
        img = load_img(fn)
        if img is None:
            continue
        f = img.astype(np.int16)
        exg = 2*f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
        green = exg > 15
        ys, xs = np.where(green)
        if len(xs) == 0:
            continue
        idx = rng.randrange(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        side = rng.randint(80, 160)
        x0, y0 = max(0, cx-side//2), max(0, cy-side//2)
        x1, y1 = min(512, x0+side), min(512, y0+side)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = cv2.resize(img[y0:y1, x0:x1], (SIZE, SIZE))
        sp = split_frame(fn)
        if sp == 'holdout':
            continue
        save_patch('neg_veg', sp, fn, made, patch)
        counts['neg_veg'][sp] += 1
        made += 1

    # NEG PARTIAL: crowns clipped by tile border from gold (iscrowd) as hard negatives
    partial = [a for a in C['annotations'] if a['iscrowd'] and a.get('segmentation')]
    made = 0
    for a in partial:
        if made >= target_each // 3:
            break
        fn = byid[a['image_id']]
        img = load_img(fn)
        if img is None:
            continue
        patch = raw_crop_centered(img, mask_of(a))
        if patch is None:
            continue
        sp = split_frame(fn)
        if sp == 'holdout':
            continue
        save_patch('neg_partial', sp, fn, a['id'], patch)
        counts['neg_partial'][sp] += 1
        made += 1

    manifest = {'seed': SEED, 'size': SIZE,
                'protocol': 'raw crops + hard negatives (soil/veg/partial)',
                'counts': counts,
                'high_green_no_crown_frames': len(high_green),
                'dev_excluded': len(dev_names), 'holdout_excluded': len(hold)}
    json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
