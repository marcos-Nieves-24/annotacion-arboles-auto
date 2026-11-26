#!/usr/bin/env python3
"""Upload StarDist polygons for Magdala frames 240-300 to CVAT job 4307818.

DEFAULT: dry-run. Requires --authorized (explicit per-batch authorisation)
for any write. Shape ids are stripped from the existing shapes.
"""
import argparse, json, os, re, urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = 4307818
URL = f"https://app.cvat.ai/api/jobs/{JOB}/annotations"
PKG = os.path.join(ROOT, "data", "para_subir", "magdala_240_300")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--authorized", action="store_true")
args = ap.parse_args()

token = re.search(r'TOKEN\s*=\s*"([^"]+)"',
                  open(os.path.join(ROOT, "experimentos", "yolo", "scripts", "ronda1.py")).read()).group(1)
new = json.load(open(os.path.join(PKG, "annotations.json")))["shapes"]
clean_new = [{k: s[k] for k in ("type", "label_id", "frame", "group", "source",
                                "points", "occluded", "outside", "z_order")} for s in new]
req = urllib.request.Request(URL, headers={"Authorization": f"Token {token}"})
with urllib.request.urlopen(req, timeout=180) as r:
    current = json.loads(r.read().decode("utf-8"))
existing = [{k: v for k, v in s.items() if k != "id"} for s in current.get("shapes", [])]
body = json.dumps({
    "version": current.get("version", 2.0), "tags": current.get("tags", []),
    "shapes": existing + clean_new, "tracks": current.get("tracks", []),
    "intervals": current.get("intervals", []),
}).encode("utf-8")
payload = json.loads(body.decode("utf-8"))
assert all("id" not in s for s in payload["shapes"]), "payload lleva ids; CVAT lo rechaza"

print(f"job={JOB} shapes actuales={len(existing)} nuevas={len(clean_new)} total={len(existing)+len(clean_new)}")
print(f"frames nuevas={len({s['frame'] for s in clean_new})} rango={min(s['frame'] for s in clean_new)}-{max(s['frame'] for s in clean_new)}")
if not args.authorized:
    print("DRY-RUN: no se escribio en CVAT. Requiere --authorized.")
    raise SystemExit(0)

json.dump(current, open(os.path.join(PKG, "backup_annotations_before_upload.json"), "w"))
rq = urllib.request.Request(URL, data=body, method="PUT",
                            headers={"Authorization": f"Token {token}",
                                     "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(rq, timeout=240) as r:
        status = r.status; resp = json.loads(r.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    raise SystemExit(f"PUT fallo HTTP {e.code}: {e.read().decode('utf-8','replace')[:500]}")
with urllib.request.urlopen(req, timeout=180) as r:
    after = json.loads(r.read().decode("utf-8"))
m = json.load(open(os.path.join(PKG, "manifest.json")))
m["status"] = "UPLOADED"
m["upload"] = {"method": "PUT", "job_id": JOB, "http_status": status,
               "n_shapes_sent": len(clean_new),
               "n_shapes_total_before": len(existing),
               "n_shapes_total_after": len(after.get("shapes", [])),
               "response_version": resp.get("version")}
json.dump(m, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)
print(json.dumps(m["upload"], indent=1))
