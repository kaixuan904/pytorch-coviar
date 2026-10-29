import os
import subprocess

ROOT = "/mnt/d/dataset/hmdb51/videos_coviar"
OUT = "hmdb51_bad_videos.txt"

exts = (".mp4", ".avi")
bad = []

for dirpath, _, filenames in os.walk(ROOT):
    for fn in filenames:
        if not fn.lower().endswith(exts):
            continue
        path = os.path.join(dirpath, fn)
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
                print("[BAD]", path, "ffprobe_failed")
                continue

            codec_ok = "codec_name=h264" in out
            pix_ok = "pix_fmt=yuv420p" in out

            if not (codec_ok and pix_ok):
                reason = []
                if not codec_ok:
                    reason.append("codec_not_h264")
                if not pix_ok:
                    reason.append("pix_not_yuv420p")
                bad.append((path, ",".join(reason)))
                print("[BAD]", path, ",".join(reason))

        except Exception as e:
            bad.append((path, f"exception:{e}"))
            print("[BAD]", path, f"exception:{e}")

with open(OUT, "w", encoding="utf-8") as f:
    for p, reason in bad:
        f.write(f"{p}\t{reason}\n")

print(f"\nDone. bad_count={len(bad)} saved_to={OUT}")
