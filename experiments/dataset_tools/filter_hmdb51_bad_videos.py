import os

# 这里填你已经确认会导致 coviar 解码崩溃的视频关键词
bad_keywords = [
    "I_ll_Just_Wave_My_Hand",
]

for s in [1, 2, 3]:
    src = f"./data/hmdb51/list_cvt/hmdb51_split{s}_test_coviar.txt"
    dst = f"./data/hmdb51/list_cvt/hmdb51_split{s}_test_coviar_filtered.txt"

    keep = 0
    drop = 0

    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if any(k in line for k in bad_keywords):
                drop += 1
                continue
            fout.write(line)
            keep += 1

    print(f"split{s}: keep={keep}, drop={drop}, saved={dst}")
