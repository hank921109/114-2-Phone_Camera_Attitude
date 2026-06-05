import os
import numpy as np
import subprocess
import time

# 針對本地 samples 的預期值 (Pitch 應趨近於 0)
# 格式: {文件名: [Yaw, Pitch, Roll]}
EXPECTED_ATT = {f"rt{i}.jpg": [0, 0, 0] for i in range(8)}

def parse_system_output(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        for line in f:
            if "Attitude (Y,P,R):" in line:
                val_str = line.split(":")[1].strip().strip('[]')
                return np.fromstring(val_str, sep=',')
    return None

samples = [f"rt{i}.jpg" for i in range(8)]
errors = []
latencies = []

print(f"{'Sample':<12} | {'Estimated (Y,P,R)':<25} | {'Latency':<10}")
print("-" * 55)

for img in samples:
    img_path = os.path.join('data/samples', img)
    start_t = time.time()
    # 執行 main.py
    subprocess.run(['python3', 'main.py', img_path], capture_output=True)
    latency = time.time() - start_t
    
    res_path = os.path.join('outputs', f"{os.path.splitext(img)[0]}CalibrationResult.txt")
    sys_att = parse_system_output(res_path)
    
    if sys_att is not None:
        # 在水平拍攝下，我們主要關注 Pitch 是否接近 0
        pitch_err = abs(sys_att[1]) 
        errors.append(pitch_err)
        latencies.append(latency)
        print(f"{img:<12} | {str(np.round(sys_att, 2)):<25} | {latency:.3f} s")

if errors:
    print("\n" + "="*30)
    print("SYSTEM PERFORMANCE REPORT")
    print("="*30)
    print(f"Mean Pitch Error:    {np.mean(errors):.3f} deg")
    print(f"Average Latency:     {np.mean(latencies):.3f} s")
    print(f"Estimated FPS:       {1/np.mean(latencies):.1f}")
    print("="*30)
