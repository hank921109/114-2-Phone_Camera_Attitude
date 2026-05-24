import sys
import os
import cv2
import numpy as np
import time
from typing import Optional, List

# 將 src 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from vp_calib.engine import (
        get_vp_inliers, draw_axes_on_image, calculate_camera_attitude, 
        choose_vanishing_points, determine_focal_length, read_image,
        calculate_rotation_matrix, AttitudeSmoother, draw_inliers,
        estimate_origin_from_inliers
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def save_results(img_path: str, vps: List[np.ndarray], focal: float, attitude: np.ndarray):
    """將校正結果儲存至文字檔"""
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join("outputs", f"{img_name}CalibrationResult.txt")
    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"VPs: {[v[:2].tolist() for v in vps]}\n")
        f.write(f"Focal: {focal:.2f}\n")
        f.write(f"Attitude (Y,P,R): {attitude.tolist()}\n")
    print(f"Results saved to {output_path}")

def process_single_image(img_path: str, output_path: Optional[str] = None):
    """處理單張影像並儲存/顯示結果"""
    print(f"Processing image: {img_path}")
    frame = read_image(img_path)
    if frame is None:
        print(f"Error: Could not read image {img_path}")
        return

    height, width = frame.shape[:2]
    try:
        # 使用與準確版本一致的參數
        inliers, hypothesis_list, viz_stuff = get_vp_inliers(
            frame, contrast=1.5, sharpness=2.0, sigma=3,
            iterations=3000, line_len=60, line_gap=15, threshold=2.0,
            processing_width=960
        )        
        pp = np.array([width/2, height/2])
        selected_vps = choose_vanishing_points(hypothesis_list, frame)
        
        if len(selected_vps) < 2:
            print(f"Error: Not enough vanishing points detected for {img_path}")
            return
            
        focal_results = determine_focal_length(selected_vps, frame)
        focal = focal_results[0]
        rot_matrix = calculate_rotation_matrix(selected_vps, focal, pp)
        attitude = calculate_camera_attitude(rot_matrix)

        save_results(img_path, selected_vps, focal, attitude)

        # 1. 先繪製物理線段標記
        processed_frame = draw_inliers(frame, inliers, viz_stuff[3])

        # 2. 自動估計繪圖原點 (地平線起點)
        origin = estimate_origin_from_inliers(frame.shape, inliers, viz_stuff[3])
            
        processed_frame = draw_axes_on_image(processed_frame, selected_vps, origin, length=height//4, attitude=attitude)
        
        if focal:
            cv2.putText(processed_frame, f"Focal: {focal:.1f}", (width - 300, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            print(f"Result saved to {output_path}")
        else:
            cv2.imshow("Calibration Result", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing image: {e}")

def process_video(video_path: str, output_path: str, stride: int = 2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    smoother = AttitudeSmoother(alpha=0.3)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / stride, (width, height))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % stride == 0:
            start_time = time.time()
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                in_masks, hyp_list, viz = get_vp_inliers(frame_rgb, 1.5, 2.0, 3, 1000, 30, 10, 2.0, 800)
                vps = choose_vanishing_points(hyp_list, frame)
                if len(vps) >= 2:
                    f = determine_focal_length(vps, frame)[0]
                    att = calculate_camera_attitude(calculate_rotation_matrix(vps, f, [width/2, height/2]))
                    att = smoother.smooth(att)
                    # 影片繪製原點設在中心
                    processed_frame = draw_inliers(frame, in_masks, viz[3])
                    processed_frame = draw_axes_on_image(processed_frame, vps, [width//2, height//2], length=height//4, attitude=att)
                else:
                    processed_frame = frame
                proc_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                cv2.putText(processed_frame, f"Proc FPS: {proc_fps:.1f}", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                out.write(processed_frame)
            except: out.write(frame)
        count += 1
        if count > 500: break
    cap.release(); out.release()

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.exists(input_path): return
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"result_{os.path.basename(input_path)}")
        if os.path.splitext(input_path)[1].lower() in ['.mp4', '.avi', '.mov']:
            process_video(input_path, output_path, stride=5)
        else:
            process_single_image(input_path, output_path)
    else:
        try:
            from vp_calib.gui import CalibrationApp
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication(sys.argv)
            window = CalibrationApp(); window.show(); sys.exit(app.exec_())
        except ImportError: pass

if __name__ == "__main__": main()
