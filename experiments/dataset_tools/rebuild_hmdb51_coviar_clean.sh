#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/mnt/d/dataset/hmdb51/videos"
DST_ROOT="/mnt/d/dataset/hmdb51/videos_coviar"
RESULTS_DIR="./results"

RUN_LOG="${RESULTS_DIR}/hmdb51_rebuild_clean_run.log"
OK_LOG="${RESULTS_DIR}/hmdb51_rebuild_clean_ok.log"
FAIL_LOG="${RESULTS_DIR}/hmdb51_rebuild_clean_fail.log"
BAD_LIST="${RESULTS_DIR}/hmdb51_bad_coviar_all.txt"

mkdir -p "${RESULTS_DIR}"
: > "${RUN_LOG}"
: > "${OK_LOG}"
: > "${FAIL_LOG}"
: > "${BAD_LIST}"

echo "===== REBUILD START =====" | tee -a "${RUN_LOG}"

count=0

find "${SRC_ROOT}" -type f \( -iname "*.avi" -o -iname "*.mp4" -o -iname "*.mpg" -o -iname "*.mpeg" \) -print0 | \
while IFS= read -r -d '' src; do
    count=$((count+1))
    rel="${src#${SRC_ROOT}/}"
    out="${DST_ROOT}/${rel%.*}.mp4"

    mkdir -p "$(dirname "$out")"

    echo "[BUILD ${count}] ${src} -> ${out}" >> "${RUN_LOG}"

    # 用 mpeg4，和 coviar/ffmpeg 解码兼容性更稳
    if ffmpeg -y -i "$src" \
        -c:v mpeg4 \
        -q:v 2 \
        -pix_fmt yuv420p \
        -an \
        "$out" >> "${RUN_LOG}" 2>&1; then
        echo "$out" >> "${OK_LOG}"
    else
        echo "[BUILD FAIL] $src -> $out" >> "${FAIL_LOG}"
    fi
done

echo "===== REBUILD FINISHED, START FULL RESCAN =====" | tee -a "${RUN_LOG}"

scan_count=0
bad_count=0

find "${DST_ROOT}" -type f -name "*.mp4" -print0 | \
while IFS= read -r -d '' f; do
    scan_count=$((scan_count+1))
    echo "[SCAN ${scan_count}] $f" >> "${RUN_LOG}"

    if ! ffmpeg -v error -xerror -i "$f" -f null - >/dev/null 2>&1; then
        echo "$f" >> "${BAD_LIST}"
        echo "[BAD] $f" >> "${FAIL_LOG}"
        bad_count=$((bad_count+1))
    fi
done

echo "===== DONE =====" | tee -a "${RUN_LOG}"
echo "OK   -> ${OK_LOG}" | tee -a "${RUN_LOG}"
echo "FAIL -> ${FAIL_LOG}" | tee -a "${RUN_LOG}"
echo "BAD  -> ${BAD_LIST}" | tee -a "${RUN_LOG}"
