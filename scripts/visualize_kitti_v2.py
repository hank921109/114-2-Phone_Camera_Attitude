import os
import sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

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
y_errs, p_errs, r_errs = [], [], []
first_sys, first_gt = None, None

for i, img in enumerate(images):
    img_name = os.path.splitext(img)[0]
    res_path = os.path.join('outputs', f"{img_name}CalibrationResult.txt")
    gt_path = os.path.join('data/kitti/gt', f"{img_name}.txt")
    
    raw_sys = parse_system_output(res_path)
    gt_att = parse_oxts_gt(gt_path)
    
    if raw_sys is not None and gt_att is not None:
        curr_sys = adapt_to_kitti_frame(raw_sys)
        if first_sys is None:
            first_sys, first_gt = curr_sys, gt_att
            continue
        rel_sys = (curr_sys - first_sys + 180) % 360 - 180
        rel_gt = (gt_att - first_gt + 180) % 360 - 180
        err = np.abs(rel_sys - rel_gt)
        err[0] = min(err[0], 360 - err[0])
        y_errs.append(err[0]); p_errs.append(err[1]); r_errs.append(err[2])

# 繪圖美化設定
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. CDF (Cumulative Distribution Function) - 期刊最愛
for data, label, color in zip([y_errs, p_errs, r_errs], ['Yaw', 'Pitch', 'Roll'], ['r', 'g', 'b']):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    axes[0].plot(sorted_data, yvals, color=color, label=f'{label} (MAE:{np.mean(data):.1f}°) ', linewidth=2)
axes[0].set_title('Cumulative Distribution Function (CDF)', fontsize=14)
axes[0].set_xlabel('Error (Degrees)', fontsize=12)
axes[0].set_ylabel('Percentage of Samples (%)', fontsize=12)
axes[0].legend()

# 2. Violin Plot - 顯示誤差分佈密度
sns.violinplot(data=[y_errs, p_errs, r_errs], ax=axes[1], inner="quartile")
axes[1].set_xticklabels(['Yaw', 'Pitch', 'Roll'])
axes[1].set_title('Error Density Distribution (Violin)', fontsize=14)
axes[1].set_ylabel('Degrees', fontsize=12)

# 3. Temporal Tracking - 相對誤差變化
axes[2].plot(y_errs, 'r--', alpha=0.5)
axes[2].plot(p_errs, 'g-', linewidth=2)
axes[2].plot(r_errs, 'b--', alpha=0.5)
axes[2].set_title('Temporal Error Persistence', fontsize=14)
axes[2].set_xlabel('Frame Sequence', fontsize=12)
axes[2].set_ylabel('Error (Degrees)', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/kitti_accuracy_report_v2.png', dpi=300)
print("Journal-style visualization saved to: outputs/kitti_accuracy_report_v2.png")
