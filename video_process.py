import cv2
import os
import sys
import numpy as np
from PIL import Image

# 將 src 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from vp_calib.engine import get_vp_inliers, draw_axes_on_image, calculate_camera_attitude, choose_vanishing_points, determine_focal_lenth

def process_video(video_path, output_path, stride=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用 XVID 編碼器，輸出為 .mp4 (或 .avi)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / stride, (width, height))

    count = 0
    print(f"Processing {video_path} ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % stride == 0:
            try:
                # 實作 Memory Pipe：直接傳入 frame 陣列，不再透過 temp_frame.jpg
                inlier_lines_list, hypothesis_list, viz_stuff = get_vp_inliers(
                    frame, contrast=1.5, sharpness=2.0, sigma=3, 
                    iterations=800, line_len=30, line_gap=10, threshold=2,
                    processing_width=640
                )
                
                # 計算姿態角以顯示在畫面上
                image = viz_stuff[0]
                vps = choose_vanishing_points(hypothesis_list[0], hypothesis_list[1], hypothesis_list[2], image)
                focal = determine_focal_lenth(vps, image)
                
                # 在畫面上畫軸 (原點設在中心)
                origin = [width // 2, height // 2]
                processed_frame = draw_axes_on_image(frame, hypothesis_list, origin, length=height//4)
                
                # 加上資訊文字
                cv2.putText(processed_frame, f"Frame: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if focal:
                    cv2.putText(processed_frame, f"Focal: {focal[0]:.1f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                out.write(processed_frame)
                print(f"Processed frame {count}/{total_frames}", end='\r')
            except Exception as e:
                print(f"\nError processing frame {count}: {e}")
                out.write(frame)
        
        count += 1
        # 測試用，只處理前 100 幀
        if count > 100:
            break

    cap.release()
    out.release()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    print(f"\nFinished. Output saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    # 處理 user 提供的影片
    process_video("data/videos/test_v1.mp4", "outputs/result_v1.mp4", stride=10)
    process_video("data/videos/test_v2.mp4", "outputs/result_v2.mp4", stride=10)
