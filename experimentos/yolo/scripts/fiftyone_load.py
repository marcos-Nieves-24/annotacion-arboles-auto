"""Load gold_v1 into FiftyOne for visual audit. Run once; server started
separately and kept alive in background."""

import json

import fiftyone as fo

ROOT = "/home/biohacker24/annotacion-arboles-auto"
coco = json.load(open(f"{ROOT}/data/gold_v1/coco.json"))
byimg = {i["id"]: i for i in coco["images"]}
anns = {}
for a in coco["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)

name = "limon-gold-v1"
if fo.dataset_exists(name):
    fo.delete_dataset(name)
ds = fo.Dataset(name)
ds.persistent = True
samples = []
for iid, info in byimg.items():
    fp = f"{ROOT}/data/gold_v1/images/{info['file_name']}"
    dets = []
    for a in anns.get(iid, []):
        x, y, w, h = a["bbox"]
        dets.append(fo.Detection(
            label="crop_tree", bounding_box=[x / 512, y / 512, w / 512, h / 512],
            mask=None))
    samples.append(fo.Sample(filepath=fp, tags=[info["farm"]],
                             ground_truth=fo.Detections(detections=dets)))
ds.add_samples(samples)
print(f"dataset {name}: {len(ds)} samples")
