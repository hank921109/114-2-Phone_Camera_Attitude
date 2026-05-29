import os
import sys
import numpy as np
import subprocess
import time

# 加入 src 到路徑
sys.path.append(os.path.abspath('src'))
from vp_calib.engine import adapt_to_kitti_frame

def parse_oxts_gt(file_path):
    """解析 KITTI 原始 OXTS 文字檔"""
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        data = f.read().split()
        if len(data) < 6: return None
        # KITTI: 3:roll, 4:pitch, 5:yaw (弧度)
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
errors_rel = []
first_sys, first_gt = None, None

print(f"{'Image':<12} | {'Rel Sys (Y,P,R)':<25} | {'Rel GT (Y,P,R)':<25} | {'Error'}")
print("-" * 85)

for i, img in enumerate(images):
    img_path = os.path.join('data/kitti/images', img)
    gt_path = os.path.join('data/kitti/gt', f"{os.path.splitext(img)[0]}.txt")
    
    gt_att = parse_oxts_gt(gt_path)
    if gt_att is None: continue
    
    # 執行主程式
    subprocess.run(['python3', 'main.py', img_path], capture_output=True)
    
    res_path = os.path.join('outputs', f"{os.path.splitext(img)[0]}CalibrationResult.txt")
    raw_sys = parse_system_output(res_path)
    
    if raw_sys is not None:
        # 轉換座標系
        curr_sys = adapt_to_kitti_frame(raw_sys)
        curr_gt = gt_att
        
        if first_sys is None:
            first_sys, first_gt = curr_sys, curr_gt
            print(f"{img:<12} | [First Frame - Bias Set]  | [First Frame]           | 0.00")
            continue
            
        # 計算相對量 (Delta)
        rel_sys = curr_sys - first_sys
        rel_gt = curr_gt - first_gt
        
        # 修正角度環繞
        rel_sys = (rel_sys + 180) % 360 - 180
        rel_gt = (rel_gt + 180) % 360 - 180
        
        err = np.abs(rel_sys - rel_gt)
        errors_rel.append(err)
        print(f"{img:<12} | {str(np.round(rel_sys, 2)):<25} | {str(np.round(rel_gt, 2)):<25} | {np.mean(err):.2f}")

if errors_rel:
    mae = np.mean(errors_rel, axis=0)
    print("\n" + "="*40 + "\nKITTI RELATIVE TRACKING FINAL REPORT\n" + "="*40)
    print(f"Yaw MAE (Relative):   {mae[0]:.3f} deg")
    print(f"Pitch MAE (Relative): {mae[1]:.3f} deg")
    print(f"Roll MAE (Relative):  {mae[2]:.3f} deg")
    print("="*40)
