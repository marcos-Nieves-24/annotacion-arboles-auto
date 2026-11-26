#!/usr/bin/env python3
"""Prepare a CVAT removal package for StarDist polygons added to already-annotated
Magdala frames 0-220. Read-only preparation; NO CVAT write.

Removal = PUT with all current shapes EXCEPT these 426 (matched by frame+points).
Only job 4307818 is touched. Only frames 0-220 of that job are affected.
"""
import json, os, re, urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = 4307818
TOKEN = os.environ.get("CVAT_TOKEN", "")  # PAT (Bearer) - set env var, never commit
URL = f'https://app.cvat.ai/api/jobs/{JOB}/annotations'
PKG = os.path.join(ROOT, 'data', 'para_subir', 'magdala_remocion_0_220')

def k(s):
    return (s.get('frame'), tuple(round(float(x), 2) for x in s.get('points', [])))

ap = argparse = None
os.makedirs(PKG, exist_ok=True)

# Fuente: estado vivo leido en la auditoria (mag_live2.json), GET con Bearer
live = json.load(open('/tmp/user/1000/opencode/mag_live2.json'))
# baseline antes de nuestras subidas
base = json.load(open('data/para_subir/magdala_0_99/backup_at_upload.json'))
base_keys = {k(s) for s in base['shapes']}
base_frames = Counter(s['frame'] for s in base['shapes'])
live_keys = {k(s) for s in live['shapes']}

mine = []
for n in ['magdala_0_99', 'magdala_100_199', 'magdala_200_239']:
    mine += json.load(open(f'data/para_subir/{n}/annotations.json'))['shapes']

# Contaminantes = mios, presentes hoy, no existian antes, en frames 0-220
# que YA tenian anotaciones en el baseline.
remove = [s for s in mine
          if 0 <= s['frame'] <= 220
          and k(s) in live_keys
          and k(s) not in base_keys
          and base_frames[s['frame']] > 0]
assert len(remove) == 426, f'esperaba 426, encontre {len(remove)}'
assert all(0 <= s['frame'] <= 220 for s in remove)

# Construir el PUT: todas las formas vivas salvo las 426, sin id
remove_keys = Counter(k(s) for s in remove)
keep = []
for s in live['shapes']:
    kk = k(s)
    if remove_keys[kk] > 0:
        remove_keys[kk] -= 1
        continue
    keep.append({a: b for a, b in s.items() if a != 'id'})

assert all('id' not in s for s in keep)
assert len(keep) == len(live['shapes']) - len(remove)

json.dump({'version': live.get('version', 2.0), 'tags': live.get('tags', []),
           'shapes': keep, 'tracks': live.get('tracks', []),
           'intervals': live.get('intervals', [])},
          open(os.path.join(PKG, 'annotations_after_removal.json'), 'w'))
json.dump(remove, open(os.path.join(PKG, 'a_remover.json'), 'w'), indent=1)

manifest = {
    'status': 'PREPARED — no CVAT write performed',
    'job_id': JOB, 'auth': 'Bearer PAT',
    'scope': 'Magdala frames 0-220 ONLY',
    'criterion': 'poligonos StarDist mios, presentes en CVAT, que no existian en el baseline, en frames 0-220 que ya tenian anotaciones',
    'n_a_remover': len(remove),
    'n_frames_afectados': len(set(s['frame'] for s in remove)),
    'n_shapes_antes': len(live['shapes']),
    'n_shapes_despues': len(keep),
    'garantia': 'job 4307818 unicamente; frames fuera de 0-220 intactos; ninguna otra finca tocada; las 426 shapes se identifican por (frame, points) exacto',
    'upload_auth': 'REQUIRES explicit user authorisation',
}
json.dump(manifest, open(os.path.join(PKG, 'manifest.json'), 'w'), indent=1)

print('job', JOB, '| frames', min(s['frame'] for s in remove), '-', max(s['frame'] for s in remove))
print('a remover', len(remove), '| shapes antes', len(live['shapes']), '| despues', len(keep))
print('por frame (top):', Counter(s['frame'] for s in remove).most_common(5))
