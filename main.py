import sys
import os
import cv2
import numpy as np
import time
import json
from typing import Optional, List

# 將 src 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from vp_calib.engine import (
        get_vp_inliers, draw_axes_on_image, calculate_camera_attitude, 
        determine_focal_length, read_image,
        calculate_rotation_matrix, draw_inliers,
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
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)['calibration']
    except Exception as e:
        print(f"Warning: Could not load config.json, using defaults. {e}")
        cfg = {"contrast": 1.5, "sharpness": 2.0, "sigma": 3, "iterations": 3000, "line_length": 60, "line_gap": 15, "threshold": 2.0, "processing_width": 960}

    try:
        inliers, selected_vps, viz_stuff = get_vp_inliers(
            frame, contrast=cfg.get("contrast", 1.5), sharpness=cfg.get("sharpness", 2.0), sigma=cfg.get("sigma", 3.0),
            iterations=cfg.get("iterations", 3000), line_len=cfg.get("line_length", 60), line_gap=cfg.get("line_gap", 15), threshold=cfg.get("threshold", 2.0),
            processing_width=cfg.get("processing_width", 960)
        )        
        pp = np.array([width/2, height/2])
        
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
            
        # 自動開啟對應的結果圖片 (Auto-open the result image)
        cv2.imshow("Calibration Result", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing image: {e}")

def process_video(video_path: str, output_path: str, stride: int = 2, max_frames: int = 200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / stride, (width, height))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % stride == 0:
            start_time = time.time()
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                in_masks, vps, viz = get_vp_inliers(frame_rgb, 1.2, 1.5, 2.0, 1000, 30, 10, 2.0, 800)
                if len(vps) >= 2:
                    f = determine_focal_length(vps, frame)[0]
                    att = calculate_camera_attitude(calculate_rotation_matrix(vps, f, [width/2, height/2]))
                    # 影片繪製原點設在中心
                    processed_frame = draw_axes_on_image(frame, vps, [width//2, height//2], length=height//4, attitude=att)
                else:
                    processed_frame = frame
                proc_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                cv2.putText(processed_frame, f"Proc FPS: {proc_fps:.1f}", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                out.write(processed_frame)
                
                # 跳出即時處理畫面 (Live processing window)
                cv2.imshow("Live Video Processing", processed_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to stop
                    break
            except Exception as e:
                print(f"Error processing video frame: {e}")
                out.write(frame)
        count += 1
        if count >= max_frames: break
    cap.release(); out.release()
    cv2.destroyAllWindows()


def launch_gui_menu():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("System Launcher")
    root.geometry("380x250")
    
    kitti_base = os.path.abspath(os.path.join(os.path.dirname(__file__), 'visual_odometry', 'KITTI', 'dataset'))
    kitti_img_dir = os.path.join(kitti_base, 'sequences', '00', 'image_0')
    if not os.path.exists(kitti_img_dir):
        kitti_img_dir = os.path.dirname(__file__)
        
    lbl_frames = tk.Label(root, text="影片處理影格數 (預設 200):")
    lbl_frames.pack(pady=(10, 0))
    entry_frames = tk.Entry(root)
    entry_frames.insert(0, "200")
    entry_frames.pack(pady=(0, 10))

    def on_single_cam():
        filepath = filedialog.askopenfilename(
            title="Select Image or Video (Single Camera)",
            initialdir=kitti_img_dir,
            filetypes=[("Media Files", "*.png *.jpg *.jpeg *.mp4 *.avi *.mov")]
        )
        if filepath:
            try:
                max_frames = int(entry_frames.get())
            except ValueError:
                max_frames = 200
                
            root.destroy()
            if os.path.splitext(filepath)[1].lower() in ['.mp4', '.avi', '.mov']:
                output_path = os.path.join("outputs", f"result_{os.path.basename(filepath)}")
                os.makedirs("outputs", exist_ok=True)
                process_video(filepath, output_path, stride=5, max_frames=max_frames)
            else:
                output_path = os.path.join("outputs", f"result_{os.path.basename(filepath)}")
                os.makedirs("outputs", exist_ok=True)
                process_single_image(filepath, output_path)
            
    def on_stereo():
        dirpath = filedialog.askdirectory(
            title="Select KITTI Dataset Directory (Stereo)",
            initialdir=kitti_base
        )
        if dirpath:
            try:
                max_frames = int(entry_frames.get())
            except ValueError:
                max_frames = 200
                
            root.destroy()
            
            # Setup path for visual_odometry
            vo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'visual_odometry'))
            vo_src_path = os.path.join(vo_path, 'src')
            vo_script_path = os.path.join(vo_path, 'scripts', 'main.py')
            
            try:
                import subprocess
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{vo_src_path}:{env.get('PYTHONPATH', '')}"
                subprocess.run([sys.executable, vo_script_path, dirpath, str(max_frames)], env=env)
            except Exception as e:
                print(f"Error launching Visual Odometry: {e}")

    lbl = tk.Label(root, text="Select Pipeline Mode", font=("Helvetica", 14))
    lbl.pack(pady=15)
    
    btn1 = tk.Button(root, text="單鏡頭消失點管線 (選擇檔案)", command=on_single_cam, width=30, height=2)
    btn1.pack(pady=5)
    
    btn2 = tk.Button(root, text="雙鏡頭視覺里程計 (選擇資料夾)", command=on_stereo, width=30, height=2)
    btn2.pack(pady=5)
    
    root.mainloop()

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
        launch_gui_menu()

if __name__ == "__main__": main()
