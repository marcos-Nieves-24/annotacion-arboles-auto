#!/usr/bin/env python3
"""Upload one prepared diff package to CVAT job 4307835.

Each invocation performs a fresh GET, so concurrent user edits are preserved.
Default: dry-run. Requires --authorized to write.
"""
import argparse, json, os, re, urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pkg", required=True, help="nombre de la carpeta en data/para_subir/")
ap.add_argument("--authorized", action="store_true")
a = ap.parse_args()

PKG = os.path.join(ROOT, "data", "para_subir", a.pkg)
# El job se toma del manifest del paquete: nunca hardcodear el ID del job.
manifest_path = os.path.join(PKG, "manifest.json")
_m = json.load(open(manifest_path))
JOB = int(_m.get("job") or _m.get("cvat", {}).get("job_id"))
assert JOB, f"el manifest {manifest_path} no declara job"
URL = f"https://app.cvat.ai/api/jobs/{JOB}/annotations"
token = re.search(r'TOKEN\s*=\s*"([^"]+)"',
                  open(os.path.join(ROOT, "experimentos", "yolo", "scripts", "ronda1.py")).read()).group(1)
H = {"Authorization": f"Token {token}"}
req = urllib.request.Request(URL, headers=H)

new = json.load(open(os.path.join(PKG, "annotations.json")))["shapes"]
clean_new = [{k: s[k] for k in ("type", "label_id", "frame", "group", "source",
                                "points", "occluded", "outside", "z_order")} for s in new]
with urllib.request.urlopen(req, timeout=240) as r:
    current = json.loads(r.read().decode("utf-8"))
existing = [{k: v for k, v in s.items() if k != "id"} for s in current.get("shapes", [])]
body = json.dumps({
    "version": current.get("version", 2.0), "tags": current.get("tags", []),
    "shapes": existing + clean_new, "tracks": current.get("tracks", []),
    "intervals": current.get("intervals", []),
}).encode("utf-8")
payload = json.loads(body.decode("utf-8"))
assert all("id" not in s for s in payload["shapes"]), "payload lleva ids; CVAT lo rechaza"

nframe = Counter(s["frame"] for s in clean_new)
print(f"{a.pkg}: actuales={len(existing)} nuevas={len(clean_new)} total={len(existing)+len(clean_new)} frames={len(nframe)}")
if not a.authorized:
    print("DRY-RUN: sin escritura.")
    raise SystemExit(0)

json.dump(current, open(os.path.join(PKG, "backup_at_upload.json"), "w"))
rq = urllib.request.Request(URL, data=body, method="PUT",
                            headers={**H, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(rq, timeout=300) as r:
        status = r.status; resp = json.loads(r.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    raise SystemExit(f"PUT fallo HTTP {e.code}: {e.read().decode('utf-8','replace')[:400]}")

with urllib.request.urlopen(req, timeout=240) as r:
    after = json.loads(r.read().decode("utf-8"))
ca = Counter(s.get("frame") for s in after.get("shapes", []))
cb = Counter(s.get("frame") for s in current.get("shapes", []))
bad = [(f, cb[f] + nframe[f], ca[f]) for f in nframe if ca[f] != cb[f] + nframe[f]]
m = json.load(open(os.path.join(PKG, "manifest.json")))
m["status"] = "UPLOADED"
m["upload"] = {"method": "PUT", "job_id": JOB, "http_status": status,
               "n_shapes_sent": len(clean_new), "before": len(existing),
               "after": len(after.get("shapes", [])),
               "frames_verified": len(nframe) - len(bad), "mismatches": bad[:10]}
json.dump(m, open(os.path.join(PKG, "manifest.json"), "w"), indent=1)
print(f"OK http={status} before={len(existing)} after={len(after.get('shapes', []))}")
print(f"frames verificados: {len(nframe)-len(bad)}/{len(nframe)}  desajustes: {len(bad)}")
if bad:
    print("DETALLE:", bad[:10])
