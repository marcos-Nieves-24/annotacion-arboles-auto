#!/usr/bin/env python3
"""Remove specific contaminating StarDist polygons from CVAT job 4307818.

Default: dry-run. --frame N limits to that frame only. --authorized writes.
Auth: Bearer PAT. Builds PUT from a FRESH GET, stripping ids, preserving edits.
"""
import argparse, json, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = 4307818
URL = f'https://app.cvat.ai/api/jobs/{JOB}/annotations'
TOKEN = os.environ.get("CVAT_TOKEN", "")  # PAT (Bearer) - set env var, never commit

ap = argparse.ArgumentParser()
ap.add_argument('--authorized', action='store_true')
ap.add_argument('--frame', type=int, default=None, help='limitar a un frame')
a = ap.parse_args()

def k(s):
    return (s.get('frame'), tuple(round(float(x), 2) for x in s.get('points', [])))

req = urllib.request.Request(URL, headers={'Authorization': f'Bearer {TOKEN}'})
with urllib.request.urlopen(req, timeout=240) as r:
    current = json.loads(r.read().decode())

base = json.load(open('data/para_subir/magdala_0_99/backup_at_upload.json'))
base_keys = {k(s) for s in base['shapes']}
base_frames = {}
for s in base['shapes']:
    base_frames.setdefault(s['frame'], 0)
    base_frames[s['frame']] += 1

mine = []
for n in ['magdala_0_99', 'magdala_100_199', 'magdala_200_239']:
    mine += json.load(open(f'data/para_subir/{n}/annotations.json'))['shapes']
live_keys = {k(s) for s in current['shapes']}

contaminantes = [s for s in mine
                 if 0 <= s['frame'] <= 220
                 and k(s) in live_keys
                 and k(s) not in base_keys
                 and base_frames.get(s['frame'], 0) > 0]
if a.frame is not None:
    contaminantes = [s for s in contaminantes if s['frame'] == a.frame]
assert contaminantes, 'no hay contaminantes para quitar'

remove = {}
for s in contaminantes:
    remove[k(s)] = remove.get(k(s), 0) + 1

keep = []
for s in current['shapes']:
    kk = k(s)
    if remove.get(kk, 0) > 0:
        remove[kk] -= 1
        continue
    keep.append({x: y for x, y in s.items() if x != 'id'})
assert all('id' not in s for s in keep)

print(f'frame={a.frame if a.frame is not None else "0-220"} shapes actuales={len(current["shapes"])} '
      f'a quitar={len(contaminantes)} resultado={len(keep)}')
if not a.authorized:
    print('DRY-RUN: sin escritura en CVAT.')
    raise SystemExit(0)

body = json.dumps({'version': current.get('version', 2.0), 'tags': current.get('tags', []),
                   'shapes': keep, 'tracks': current.get('tracks', []),
                   'intervals': current.get('intervals', [])}).encode()
rq = urllib.request.Request(URL, data=body, method='PUT',
                            headers={'Authorization': f'Bearer {TOKEN}',
                                     'Content-Type': 'application/json'})
with urllib.request.urlopen(rq, timeout=300) as r:
    status = r.status
with urllib.request.urlopen(req, timeout=240) as r:
    after = json.loads(r.read().decode())

print(json.dumps({'http': status, 'before': len(current['shapes']),
                  'removed': len(contaminantes), 'after': len(after['shapes']),
                  'frame': a.frame}, indent=1))
