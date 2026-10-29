import os
import subprocess

LISTS = [
    "./data/hmdb51/list_cvt/hmdb51_split1_test_coviar.txt",
    "./data/hmdb51/list_cvt/hmdb51_split2_test_coviar.txt",
    "./data/hmdb51/list_cvt/hmdb51_split3_test_coviar.txt",
]

OUT = "hmdb51_testlist_bad_videos.txt"

videos = []
seen = set()

for path in LISTS:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            video = " ".join(parts[:-2])
            if video not in seen:
                seen.add(video)
                videos.append(video)

print(f"total unique test videos = {len(videos)}")

bad = []

for i, path in enumerate(videos, 1):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height",
        "-of", "default=noprint_wrappers=1",
        path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = (r.stdout + "\n" + r.stderr).strip()

        if r.returncode != 0:
            bad.append((path, "ffprobe_failed"))
            print(f"[BAD {i}/{len(videos)}] {path} :: ffprobe_failed")
            continue

        codec_ok = "codec_name=h264" in out
        pix_ok = "pix_fmt=yuv420p" in out

        if not (codec_ok and pix_ok):
            reason = []
            if not codec_ok:
                reason.append("codec_not_h264")
            if not pix_ok:
                reason.append("pix_not_yuv420p")
            reason = ",".join(reason)
            bad.append((path, reason))
            print(f"[BAD {i}/{len(videos)}] {path} :: {reason}")

        if i % 500 == 0:
            print(f"[PROGRESS] checked {i}/{len(videos)}")

    except Exception as e:
        bad.append((path, f"exception:{e}"))
        print(f"[BAD {i}/{len(videos)}] {path} :: exception:{e}")

with open(OUT, "w", encoding="utf-8") as f:
    for p, reason in bad:
        f.write(f"{p}\t{reason}\n")

print(f"\nDone. bad_count={len(bad)} saved_to={OUT}")
