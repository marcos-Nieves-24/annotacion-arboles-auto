#!/usr/bin/env python3
"""Upload the prepared frame-261 pre-annotation to CVAT with PATCH.

PATCH adds/updates objects without deleting existing annotations in the job.
The token is read from scripts/ronda1.py and is never printed or stored here.
Requires explicit user authorisation (given by 'subelo').
"""
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = 4307835
FRAME = 261
PKG = os.path.join(ROOT, "data", "para_subir", "villas_261")

token = re.search(r'TOKEN\s*=\s*"([^"]+)"',
                  open(os.path.join(ROOT, "scripts", "ronda1.py")).read()).group(1)
payload = json.load(open(os.path.join(PKG, "annotations.json")))
# keep only fields CVAT accepts for shapes
clean_shapes = []
for s in payload["shapes"]:
    clean_shapes.append({
        "type": s["type"], "label_id": s["label_id"], "frame": s["frame"],
        "group": s.get("group", 0), "source": s.get("source", "auto"),
        "points": s["points"], "occluded": s.get("occluded", False),
        "outside": s.get("outside", False), "z_order": s.get("z_order", 0),
    })
url = f"https://app.cvat.ai/api/jobs/{JOB}/annotations"
# Fetch current annotations and merge our new shapes, so PUT never deletes data.
req_get = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
with urllib.request.urlopen(req_get, timeout=180) as r:
    current = json.loads(r.read().decode("utf-8"))
existing = []
for s in current.get("shapes", []):
    s2 = {k: v for k, v in s.items() if k != "id"}
    existing.append(s2)
merged = existing + clean_shapes
body = json.dumps({"version": current.get("version", 2.0),
                   "tags": current.get("tags", []),
                   "shapes": merged,
                   "tracks": current.get("tracks", []),
                   "intervals": current.get("intervals", [])}).encode("utf-8")
req = urllib.request.Request(url, data=body, method="PUT",
                             headers={"Authorization": f"Token {token}",
                                      "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    resp = json.loads(r.read().decode("utf-8"))
    status = r.status

# verify: count shapes on frame 261 after the write
req2 = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
with urllib.request.urlopen(req2, timeout=180) as r2:
    after = json.loads(r2.read().decode("utf-8"))
n_after = sum(1 for s in after.get("shapes", []) if s.get("frame") == FRAME)

manifest_path = os.path.join(PKG, "manifest.json")
manifest = json.load(open(manifest_path))
manifest["status"] = "UPLOADED"
manifest["upload"] = {"method": "PATCH", "job_id": JOB, "http_status": status,
                      "n_shapes_sent": len(clean_shapes),
                      "n_shapes_frame_261_after": n_after,
                      "response_version": resp.get("version")}
json.dump(manifest, open(manifest_path, "w"), indent=1)
print(json.dumps(manifest["upload"], indent=1))
