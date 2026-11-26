#!/usr/bin/env python3
"""Merge the 0.20/0.25 run into the main Villas 273-300 sweep report."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "data", "stardist_villas_273_300_sweep",
                    "stardist_villas_273_300_sweep.json")
LOW = os.path.join(ROOT, "data", "stardist_villas_273_300_low",
                   "stardist_villas_273_300_sweep.json")

main = json.load(open(MAIN))
low = json.load(open(LOW))
assert low["nms"] == main["nms"]

for tag in ("0.2000", "0.2500"):
    assert tag in low["per_prob"], f"falta {tag} en el reporte bajo"
    main["per_prob"][tag] = low["per_prob"][tag]
main["probs"] = sorted(set([0.20, 0.25] + main["probs"]))
json.dump(main, open(MAIN, "w"), indent=1)
print("probs:", main["probs"])
print("per_prob claves:", sorted(main["per_prob"]))
