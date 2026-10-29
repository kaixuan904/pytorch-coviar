import os
import cv2

SRC_DIR = "/mnt/d/dataset/hmdb51/videos_coviar"
LIST_DIR = "./data/hmdb51/list_cvt"

def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

for s in [1, 2, 3]:
    src = os.path.join(LIST_DIR, f"hmdb51_split{s}_test.txt")
    dst = os.path.join(LIST_DIR, f"hmdb51_split{s}_test_coviar.txt")

    keep = 0
    drop = 0

    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            label = parts[-1]
            old_video = " ".join(parts[:-2])

            cls = os.path.basename(os.path.dirname(old_video))
            stem = os.path.splitext(os.path.basename(old_video))[0]
            new_video = os.path.join(SRC_DIR, cls, stem + ".mp4")

            if os.path.exists(new_video):
                n = get_num_frames(new_video)
                if n > 0:
                    fout.write(f"{new_video} {n} {label}\n")
                    keep += 1
                else:
                    drop += 1
            else:
                drop += 1

    print(f"split{s}: keep={keep}, drop={drop}, saved={dst}")
