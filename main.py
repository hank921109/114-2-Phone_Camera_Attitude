import sys
import os
import cv2
import numpy as np

# 將 src 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from vp_calib.engine import (
        get_vp_inliers, draw_axes_on_image, calculate_camera_attitude, 
        choose_vanishing_points, determine_focal_lenth, read_image
    )
    from vp_calib.gui import Ui_MainWindow
    from PyQt5 import QtWidgets
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure requirements are installed: pip install -r requirements.txt")
    sys.exit(1)

def process_single_image(img_path, output_path=None):
    """處理單張影像並儲存/顯示結果"""
    print(f"Processing image: {img_path}")
    frame = read_image(img_path)
    if frame is None:
        print(f"Error: Could not read image {img_path}")
        return

    height, width = frame.shape[:2]
    try:
        inlier_lines_list, hypothesis_list, viz_stuff = get_vp_inliers(
            frame, contrast=1.5, sharpness=2.0, sigma=3, 
            iterations=800, line_len=30, line_gap=10, threshold=2,
            processing_width=640
        )
        
        # 在畫面上畫軸 (原點設在中心)
        origin = [width // 2, height // 2]
        processed_frame = draw_axes_on_image(frame, hypothesis_list, origin, length=height//4)
        
        vps = choose_vanishing_points(hypothesis_list[0], hypothesis_list[1], hypothesis_list[2], frame)
        focal = determine_focal_lenth(vps, frame)

        if focal:
            cv2.putText(processed_frame, f"Focal: {focal[0]:.1f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"Result saved to {output_path}")
        else:
            cv2.imshow("Calibration Result", processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error processing image: {e}")

def process_video(video_path, output_path, stride=5):
    """處理影片並輸出視覺化結果"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用 mp4v 編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / stride, (width, height))

    count = 0
    print(f"Processing video: {video_path} ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % stride == 0:
            try:
                inlier_lines_list, hypothesis_list, viz_stuff = get_vp_inliers(
                    frame, contrast=1.5, sharpness=2.0, sigma=3, 
                    iterations=800, line_len=30, line_gap=10, threshold=2,
                    processing_width=640
                )
                
                origin = [width // 2, height // 2]
                processed_frame = draw_axes_on_image(frame, hypothesis_list, origin, length=height//4)
                
                vps = choose_vanishing_points(hypothesis_list[0], hypothesis_list[1], hypothesis_list[2], frame)
                focal = determine_focal_lenth(vps, frame)
                
                cv2.putText(processed_frame, f"Frame: {count}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if focal:
                    cv2.putText(processed_frame, f"Focal: {focal[0]:.1f}", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                out.write(processed_frame)
                print(f"Processed frame {count}/{total_frames}", end='\r')
            except Exception as e:
                print(f"\nError processing frame {count}: {e}")
                out.write(frame)
        
        count += 1
        # 測試用，可調整
        if count > 200: 
            break

    cap.release()
    out.release()
    print(f"\nFinished. Video saved to {output_path}")

def is_video(path):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    return os.path.splitext(path)[1].lower() in video_exts

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return

        os.makedirs("outputs", exist_ok=True)
        filename = os.path.basename(input_path)
        output_path = os.path.join("outputs", f"result_{filename}")

        if is_video(input_path):
            process_video(input_path, output_path, stride=5)
        else:
            process_single_image(input_path, output_path)
    else:
        # 無參數則啟動 GUI
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
