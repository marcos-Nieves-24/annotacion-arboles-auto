#!/bin/bash
# Parallel CVAT frame downloader (read-only GETs). Usage: dl_parallel.sh
TOKEN="${CVAT_TOKEN:-}"
ROOT=/home/biohacker24/annotacion-arboles-auto
dl_one() {
  job=$1; idx=$2; dest=$3
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" -gt 10000 ]; then return 0; fi
  for att in 1 2 3; do
    code=$(curl -s -H "Authorization: Token $TOKEN" -o "$dest" -w "%{http_code}" \
      "https://app.cvat.ai/api/jobs/$job/data?number=$idx&type=frame&quality=original")
    if [ "$code" = "200" ] && [ "$(stat -c%s "$dest")" -gt 10000 ]; then return 0; fi
    sleep 2
  done
  echo "FAIL $dest"
  return 1
}
export TOKEN
export -f dl_one 2>/dev/null || true
lista() { # farm job start end: prints "job idx dest" lines
  farm=$1; job=$2; s=$3; e=$4
  python3 - "$farm" "$job" "$s" "$e" <<'EOF'
import json, sys
farm, job, s, e = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
root = '/home/biohacker24/annotacion-arboles-auto'
meta = json.load(open(f'{root}/data/raw/{farm}/meta.json'))
names = [f['name'] for f in meta['frames']]
deleted = {'yulima': {700, 701, 704, 725}}.get(farm, set())
for i in range(s, e):
    if i in deleted:
        continue
    print(job, i, f'{root}/data/r1/images/{farm}/{names[i]}')
EOF
}
mkdir -p "$ROOT/data/r1/images/magdala" "$ROOT/data/r1/images/yulima"
lista magdala 4307818 131 1302 > /tmp/user/1000/opencode/dl_magdala.txt
lista yulima 4307788 0 756 > /tmp/user/1000/opencode/dl_yulima.txt
wc -l /tmp/user/1000/opencode/dl_magdala.txt /tmp/user/1000/opencode/dl_yulima.txt
export SHELL=/bin/bash
cat /tmp/user/1000/opencode/dl_magdala.txt /tmp/user/1000/opencode/dl_yulima.txt \
  | xargs -P 6 -n 3 bash -c 'dl_one "$@"' _
echo "parallel done"
echo -n "magdala: "; ls "$ROOT/data/r1/images/magdala" | wc -l
echo -n "yulima: "; ls "$ROOT/data/r1/images/yulima" | wc -l
