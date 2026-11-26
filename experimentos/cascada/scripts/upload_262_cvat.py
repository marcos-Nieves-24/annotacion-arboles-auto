#!/usr/bin/env python3
"""Upload frame-262 StarDist pre-annotations to CVAT with PUT (merging, no deletion)."""
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = 4307835
FRAME = 262
PKG = os.path.join(ROOT, "data", "para_subir", "villas_262")
token = re.search(r'TOKEN\s*=\s*"([^"]+)"',
                  open(os.path.join(ROOT, "scripts", "ronda1.py")).read()).group(1)
url = f"https://app.cvat.ai/api/jobs/{JOB}/annotations"

# backup current state
req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
with urllib.request.urlopen(req, timeout=180) as r:
    current = json.loads(r.read().decode("utf-8"))
json.dump(current, open(os.path.join(PKG, "backup_annotations_before_upload.json"), "w"))
n_before = sum(1 for s in current.get("shapes", []) if s.get("frame") == FRAME)

new = json.load(open(os.path.join(PKG, "annotations.json")))["shapes"]
clean_new = [{k: s[k] for k in ("type", "label_id", "frame", "group", "source",
                                "points", "occluded", "outside", "z_order")}
             for s in new]
existing = [{k: v for k, v in s.items() if k != "id"}
            for s in current.get("shapes", [])]
body = json.dumps({
    "version": current.get("version", 2.0),
    "tags": current.get("tags", []),
    "shapes": existing + clean_new,
    "tracks": current.get("tracks", []),
    "intervals": current.get("intervals", []),
}).encode("utf-8")
rq = urllib.request.Request(url, data=body, method="PUT",
                            headers={"Authorization": f"Token {token}",
                                     "Content-Type": "application/json"})
with urllib.request.urlopen(rq, timeout=240) as r:
    status = r.status
    resp = json.loads(r.read().decode("utf-8"))

with urllib.request.urlopen(req, timeout=180) as r:
    after = json.loads(r.read().decode("utf-8"))
n_after = sum(1 for s in after.get("shapes", []) if s.get("frame") == FRAME)

manifest = json.load(open(os.path.join(PKG, "manifest.json")))
manifest["status"] = "UPLOADED"
manifest["upload"] = {"method": "PUT", "job_id": JOB, "http_status": status,
                      "n_shapes_sent": len(clean_new),
                      "n_shapes_frame_262_before": n_before,
                      "n_shapes_frame_262_after": n_after,
                      "response_version": resp.get("version")}
json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)
print(json.dumps(manifest["upload"], indent=1))
