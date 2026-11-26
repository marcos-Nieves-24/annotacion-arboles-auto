"""Export raw CVAT annotations for gold_v2 ranges (READ ONLY, no writes to CVAT).

Downloads via GET the annotations of jobs 4307818 (Magdala) and 4307835 (Villas)
and stores only the relevant frames as compact JSON, plus a manifest.
Output: data/gold_v2_raw/{magdala_annotations.json,villas_annotations.json,manifest.json}

This is the ONLY step that touches CVAT. It does not write to the platform.
"""

import json
import os
import re
import urllib.request

ROOT = "/home/biohacker24/annotacion-arboles-auto"
OUT = f"{ROOT}/data/gold_v2_raw"
os.makedirs(OUT, exist_ok=True)

TOKEN = re.search(r'TOKEN\s*=\s*"([^"]+)"',
                  open(f"{ROOT}/scripts/ronda1.py").read()).group(1)

JOBS = {
    "magdala": {"job": 4307818, "range": range(200, 241)},
    "villas": {"job": 4307835, "range": range(200, 233)},
}


def main():
    manifest = {"token_source": "scripts/ronda1.py (read only)", "jobs": {}}
    for farm, cfg in JOBS.items():
        url = f"https://app.cvat.ai/api/jobs/{cfg['job']}/annotations"
        req = urllib.request.Request(url,
                                     headers={"Authorization": f"Token {TOKEN}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            ann = json.loads(r.read())
        rng = set(cfg["range"])
        shapes = [s for s in ann.get("shapes", []) if s.get("frame") in rng]
        tracks = []
        for t in ann.get("tracks", []):
            ts = [s for s in t.get("shapes", []) if s.get("frame") in rng]
            if ts:
                tracks.append({**{k: v for k, v in t.items() if k != "shapes"},
                               "shapes": ts})
        out = {"version": ann.get("version"), "tags": ann.get("tags", []),
               "shapes": shapes, "tracks": tracks, "intervals": ann.get("intervals", [])}
        path = f"{OUT}/{farm}_annotations.json"
        json.dump(out, open(path, "w"))
        manifest["jobs"][farm] = {"job_id": cfg["job"],
                                  "frames": [min(rng), max(rng)],
                                  "n_shapes": len(shapes),
                                  "n_tracks": len(tracks),
                                  "source_counts": {
                                      k: sum(1 for s in shapes if s.get("source") == k)
                                      for k in set(s.get("source") for s in shapes)},
                                  "file": os.path.basename(path)}
        print(farm, manifest["jobs"][farm], flush=True)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    print("saved manifest", flush=True)


if __name__ == "__main__":
    main()
