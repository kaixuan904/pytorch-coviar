import os

video_root = "/mnt/d/dataset/hmdb51/videos"
output_dir = "/home/srtp/pytorch-coviar/data/hmdb51/list_cvt"
os.makedirs(output_dir, exist_ok=True)

class_to_idx = {name: idx for idx, name in enumerate(sorted(os.listdir(video_root)))}

splits = [1, 2, 3]

for split in splits:
    test_list = []
    for class_name in os.listdir(video_root):
        class_path = os.path.join(video_root, class_name)
        if not os.path.isdir(class_path):
            continue
        label = class_to_idx[class_name]
        for video in os.listdir(class_path):
            path = os.path.join(class_path, video)
            frame_count = 25
            test_list.append(f"{path} {frame_count} {label}\n")
    out_file = os.path.join(output_dir, f"hmdb51_split{split}_test.txt")
    with open(out_file, "w") as f:
        f.writelines(test_list)

print("All split test lists generated at:", output_dir)
