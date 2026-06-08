import os
import shutil

src_dir = "/home/nvidia/Documents/VanishingPointCameraCalibration/visual_odometry/temp_kitti"
dst_root = "/home/nvidia/Documents/VanishingPointCameraCalibration/visual_odometry/KITTI/dataset"

# Unzip
os.system(f"unzip -q {os.path.join(src_dir, '2011_09_26_drive_0001_sync.zip')} -d {src_dir}")
os.system(f"unzip -q {os.path.join(src_dir, '2011_09_26_calib.zip')} -d {src_dir}")

raw_drive_dir = os.path.join(src_dir, "2011_09_26", "2011_09_26_drive_0001_sync")
raw_calib_dir = os.path.join(src_dir, "2011_09_26")

seq_dir = os.path.join(dst_root, "sequences", "00")
os.makedirs(os.path.join(seq_dir, "image_0"), exist_ok=True)
os.makedirs(os.path.join(seq_dir, "image_1"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "poses"), exist_ok=True)

# Copy and rename images
print("Copying image_0...")
imgs_00 = sorted(os.listdir(os.path.join(raw_drive_dir, "image_00", "data")))
for i, img in enumerate(imgs_00):
    src = os.path.join(raw_drive_dir, "image_00", "data", img)
    dst = os.path.join(seq_dir, "image_0", f"{i:06d}.png")
    shutil.copy(src, dst)

print("Copying image_1...")
imgs_01 = sorted(os.listdir(os.path.join(raw_drive_dir, "image_01", "data")))
for i, img in enumerate(imgs_01):
    src = os.path.join(raw_drive_dir, "image_01", "data", img)
    dst = os.path.join(seq_dir, "image_1", f"{i:06d}.png")
    shutil.copy(src, dst)

# Create calib.txt
print("Creating calib.txt...")
calib_lines = []
with open(os.path.join(raw_calib_dir, "calib_cam_to_cam.txt"), "r") as f:
    for line in f:
        if line.startswith("P_rect_00:"): calib_lines.append("P0: " + line.split(":", 1)[1].strip())
        elif line.startswith("P_rect_01:"): calib_lines.append("P1: " + line.split(":", 1)[1].strip())
        elif line.startswith("P_rect_02:"): calib_lines.append("P2: " + line.split(":", 1)[1].strip())
        elif line.startswith("P_rect_03:"): calib_lines.append("P3: " + line.split(":", 1)[1].strip())

with open(os.path.join(raw_calib_dir, "calib_velo_to_cam.txt"), "r") as f:
    r_lines = []
    t_lines = []
    for line in f:
        if line.startswith("R:"): r_lines = line.split(":", 1)[1].strip().split()
        elif line.startswith("T:"): t_lines = line.split(":", 1)[1].strip().split()
    if r_lines and t_lines:
        tr = f"{r_lines[0]} {r_lines[1]} {r_lines[2]} {t_lines[0]} " \
             f"{r_lines[3]} {r_lines[4]} {r_lines[5]} {t_lines[1]} " \
             f"{r_lines[6]} {r_lines[7]} {r_lines[8]} {t_lines[2]}"
        calib_lines.append("Tr: " + tr)

with open(os.path.join(seq_dir, "calib.txt"), "w") as f:
    f.write("\n".join(calib_lines) + "\n")

# Create times.txt
print("Creating times.txt...")
num_imgs = len(imgs_00)
with open(os.path.join(seq_dir, "times.txt"), "w") as f:
    for i in range(num_imgs):
        f.write(f"{i*0.1}\n")

# Create poses/00.txt
print("Creating poses/00.txt...")
with open(os.path.join(dst_root, "poses", "00.txt"), "w") as f:
    for i in range(num_imgs):
        f.write("1 0 0 0 0 1 0 0 0 0 1 0\n")

print("Format conversion complete.")
