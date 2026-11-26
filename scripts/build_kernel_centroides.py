#!/usr/bin/env python3
"""Convert a # %% kernel source into a Kaggle notebook + metadata."""
import argparse
import json
import uuid
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source", default="kernels/centroides/kernel_source.py")
ap.add_argument("--out", default="kernels/centroides")
ap.add_argument("--id", default="marcosalvarez234/limon-centroides-v1")
ap.add_argument("--title", default="Limon Centroides V1")
ap.add_argument("--code-file", default="centroides.ipynb")
ap.add_argument("--dataset", default="marcosalvarez234/limon-gold-v2d")
ap.add_argument("--kernel-source", default="", help="comma-separated kernel ids")
args = ap.parse_args()

KDIR = Path(args.out)
src = Path(args.source).read_text()
parts = src.split("# %%")
cells = []
for part in parts[1:]:
    first, sep, rest = part.partition("\n")
    assert sep, "cell marker must be followed by newline"
    kind = "markdown" if first.strip() == "[markdown]" else "code"
    cell = {
        "cell_type": kind,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": (rest.strip("\n") + "\n").splitlines(True),
    }
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cells.append(cell)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}
(KDIR / args.code_file).write_text(json.dumps(nb, indent=1))

meta = {
    "id": args.id,
    "title": args.title,
    "code_file": args.code_file,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [args.dataset],
    "competition_sources": [],
    "kernel_sources": [s for s in args.kernel_source.split(",") if s],
}
(KDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(f"{args.id}: {len(cells)} cells -> {KDIR}")
