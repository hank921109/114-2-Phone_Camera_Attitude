import os
import shutil
import cv2
import numpy as np

src_imgs = "/home/nvidia/Documents/VanishingPointCameraCalibration/data/kitti/images"
dst_root = "/home/nvidia/Documents/VanishingPointCameraCalibration/visual_odometry/KITTI/dataset"

seq_dir = os.path.join(dst_root, "sequences", "00")
os.makedirs(os.path.join(seq_dir, "image_0"), exist_ok=True)
os.makedirs(os.path.join(seq_dir, "image_1"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "poses"), exist_ok=True)

# Copy images
for i in range(10):
    img_name = f"{i:06d}.png"
    src = os.path.join(src_imgs, img_name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(seq_dir, "image_0", img_name))
        shutil.copy(src, os.path.join(seq_dir, "image_1", img_name))

# Create calib.txt
with open(os.path.join(seq_dir, "calib.txt"), "w") as f:
    f.write("P0: 718.856 0 607.1928 0 0 718.856 185.2157 0 0 0 1 0\n")
    f.write("P1: 718.856 0 607.1928 -386.1448 0 718.856 185.2157 0 0 0 1 0\n")
    f.write("P2: 718.856 0 607.1928 45.38225 0 718.856 185.2157 -0.1130887 0 0 1 0.003779761\n")
    f.write("P3: 718.856 0 607.1928 -337.2877 0 718.856 185.2157 2.369057 0 0 1 0.004915215\n")

# Create times.txt
with open(os.path.join(seq_dir, "times.txt"), "w") as f:
    for i in range(10):
        f.write(f"{i*0.1}\n")

# Create poses
with open(os.path.join(dst_root, "poses", "00.txt"), "w") as f:
    for i in range(10):
        # identity matrix flattened 3x4
        f.write("1 0 0 0 0 1 0 0 0 0 1 0\n")

print("Dataset generated")
