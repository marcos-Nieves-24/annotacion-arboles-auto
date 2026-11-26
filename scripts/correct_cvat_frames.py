#!/usr/bin/env python3
"""Fix the frame-mapping error from the 263-300 upload.

Removes the 55 StarDist shapes that were placed on frames 295-300 but belong
to images 00805-00810 (real frames 280-285), then adds the corrected run's
polygons for frames 261, 280-294 and 295-300.

DEFAULT: dry-run. Nothing is written to CVAT unless --authorized is passed.
"""
import argparse
import json
import os
import re
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = 4307835
URL = f"https://app.cvat.ai/api/jobs/{JOB}/annotations"
WRONG_FRAMES = set(range(295, 301))
LAST_PKG = os.path.join(ROOT, "data", "para_subir", "villas_263_300")
NEW_DIR = os.path.join(ROOT, "data", "stardist_resto")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--authorized", action="store_true")
args = ap.parse_args()

token = re.search(
    r'TOKEN\s*=\s*"([^"]+)"',
    open(os.path.join(ROOT, "experimentos", "yolo", "scripts", "ronda1.py")).read(),
).group(1)

last = json.load(open(os.path.join(LAST_PKG, "annotations.json")))["shapes"]
wrong_points = {tuple(s["points"]) for s in last if s["frame"] in WRONG_FRAMES}

new_shapes = []
for f in (261, *range(280, 295), *range(295, 301)):
    p = os.path.join(NEW_DIR, f"cvat_{f}_stardist.json")
    new_shapes.extend(json.load(open(p))["shapes"])

req = urllib.request.Request(URL, headers={"Authorization": f"Token {token}"})
with urllib.request.urlopen(req, timeout=180) as r:
    current = json.loads(r.read().decode("utf-8"))

removed = []
kept = []
for s in current.get("shapes", []):
    if s.get("frame") in WRONG_FRAMES and tuple(s.get("points", [])) in wrong_points:
        removed.append(s)
    else:
        # CVAT rejects PUT payloads that carry shape ids; strip them.
        kept.append({k: v for k, v in s.items() if k != "id"})

clean_new = [
    {k: s[k] for k in ("type", "label_id", "frame", "group", "source",
                       "points", "occluded", "outside", "z_order")}
    for s in new_shapes
]
body = json.dumps({
    "version": current.get("version", 2.0),
    "tags": current.get("tags", []),
    "shapes": kept + clean_new,
    "tracks": current.get("tracks", []),
    "intervals": current.get("intervals", []),
}).encode("utf-8")

print(f"shapes actuales={len(current.get('shapes', []))}")
print(f"a quitar (mal puestas en 295-300)={len(removed)}  (esperado 55)")
print(f"a agregar corregidas={len(clean_new)}  frames={len({s['frame'] for s in clean_new})}")
print(f"total final={len(kept)+len(clean_new)}")
print("quitar por frame:", dict(Counter(s["frame"] for s in removed)))
print("agregar por frame:", dict(sorted(Counter(s["frame"] for s in clean_new).items())))

payload = json.loads(body.decode("utf-8"))
assert all("id" not in s for s in payload["shapes"]), "el payload lleva shape ids; CVAT lo rechaza"

if not args.authorized:
    print("DRY-RUN: no se escribio en CVAT. Requiere --authorized.")
    raise SystemExit(0)
if len(removed) != 55:
    raise SystemExit(f"ABORTO: esperaba quitar 55 shapes, encontre {len(removed)}")

json.dump(current, open(os.path.join(NEW_DIR, "backup_annotations_before_fix.json"), "w"))
rq = urllib.request.Request(URL, data=body, method="PUT",
                            headers={"Authorization": f"Token {token}",
                                     "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(rq, timeout=240) as r:
        status = r.status
        resp = json.loads(r.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", "replace")[:500]
    raise SystemExit(f"PUT fallo HTTP {e.code}: {detail}")
with urllib.request.urlopen(req, timeout=180) as r:
    after = json.loads(r.read().decode("utf-8"))
print(json.dumps({"http_status": status, "removed": len(removed),
                  "added": len(clean_new),
                  "before": len(current.get("shapes", [])),
                  "after": len(after.get("shapes", [])),
                  "response_version": resp.get("version")}, indent=1))
