import os
import sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt

# 加入 src 到路徑
sys.path.append(os.path.abspath('src'))
from vp_calib.engine import adapt_to_kitti_frame

def parse_oxts_gt(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        data = f.read().split()
        if len(data) < 6: return None
        roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])
        return np.rad2deg([yaw, pitch, roll])

def parse_system_output(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        for line in f:
            if "Attitude (Y,P,R):" in line:
                val_str = line.split(":")[1].strip().strip('[]')
                return np.fromstring(val_str, sep=',')
    return None

images = sorted([f for f in os.listdir('data/kitti/images') if f.endswith('.png')])
gt_poses_dir = 'data/kitti/gt'

frame_indices = []
y_errs, p_errs, r_errs = [], [], []
first_sys, first_gt = None, None

print("Analyzing frames for visualization...")

for i, img in enumerate(images):
    img_name = os.path.splitext(img)[0]
    gt_path = os.path.join(gt_poses_dir, f"{img_name}.txt")
    gt_att = parse_oxts_gt(gt_path)
    if gt_att is None: continue

    # 執行一次 main.py (若 outputs 已存在可跳過，但為了數據準確建議重新執行)
    img_path = os.path.join('data/kitti/images', img)
    subprocess.run(['python3', 'main.py', img_path], capture_output=True)
    
    res_path = os.path.join('outputs', f"{img_name}CalibrationResult.txt")
    raw_sys = parse_system_output(res_path)
    
    if raw_sys is not None:
        curr_sys = adapt_to_kitti_frame(raw_sys)
        curr_gt = gt_att
        
        if first_sys is None:
            first_sys, first_gt = curr_sys, curr_gt
            continue
            
        rel_sys = (curr_sys - first_sys + 180) % 360 - 180
        rel_gt = (curr_gt - first_gt + 180) % 360 - 180
        
        err = np.abs(rel_sys - rel_gt)
        # Yaw wrapping
        err[0] = min(err[0], 360 - err[0])
        
        frame_indices.append(i)
        y_errs.append(err[0])
        p_errs.append(err[1])
        r_errs.append(err[2])

# 繪圖
if frame_indices:
    plt.figure(figsize=(12, 8))
    
    # 子圖 1: 誤差隨時間變化
    plt.subplot(2, 1, 1)
    plt.plot(frame_indices, y_errs, 'r-o', label='Yaw Error', alpha=0.7)
    plt.plot(frame_indices, p_errs, 'g-o', label='Pitch Error', alpha=0.7)
    plt.plot(frame_indices, r_errs, 'b-o', label='Roll Error', alpha=0.7)
    plt.title('KITTI Relative Tracking Error (Frame by Frame)')
    plt.xlabel('Frame Index')
    plt.ylabel('MAE (Degrees)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 子圖 2: 誤差分佈箱型圖
    plt.subplot(2, 1, 2)
    plt.boxplot([y_errs, p_errs, r_errs], labels=['Yaw', 'Pitch', 'Roll'])
    plt.title('Error Distribution across Sequence')
    plt.ylabel('Error (Degrees)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = 'outputs/kitti_accuracy_report.png'
    plt.savefig(plot_path)
    print(f"Visualization report saved to: {plot_path}")
else:
    print("No data found to plot.")
