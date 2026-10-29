#!/bin/bash
set -euo pipefail

SRC="/mnt/d/dataset/hmdb51/videos"
DST="/mnt/d/dataset/hmdb51/videos_coviar"

mkdir -p "$DST"

while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    cls="$(dirname "$rel")"
    base="$(basename "$f")"
    stem="${base%.*}"

    mkdir -p "$DST/$cls"
    out="$DST/$cls/$stem.mp4"

    if [ -f "$out" ]; then
        echo "[skip] $out"
        continue
    fi

    echo "[encode] $f -> $out"

    ffmpeg -nostdin -y -i "$f" \
      -c:v libx264 -preset medium -crf 23 \
      -pix_fmt yuv420p \
      -g 12 -bf 0 \
      -an \
      "$out" </dev/null

done < <(find "$SRC" -type f \( -iname "*.avi" -o -iname "*.mp4" \) -print0)
