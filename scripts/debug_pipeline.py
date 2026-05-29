import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from vp_calib.engine import get_vp_inliers, draw_inliers

def main():
    img_path = "data/samples/rt7.jpg"
    print(f"Running pipeline debug for {img_path}")
    
    # Run the get_vp_inliers function with the latest parameters
    inliers, vps, viz_stuff = get_vp_inliers(
        img_path, contrast=1.5, sharpness=2.0, sigma=3,
        iterations=3000, line_len=60, line_gap=15, threshold=2.0,
        processing_width=960
    )
    
    full_image, enhanced_img, edges_small, lines = viz_stuff
    
    os.makedirs("outputs/pipeline", exist_ok=True)
    
    # 1. Enhanced Image
    cv2.imwrite("outputs/pipeline/01_enhanced.jpg", cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
    
    # 2. Canny Edges
    cv2.imwrite("outputs/pipeline/02_edges.jpg", edges_small)
    
    # 3. All Detected Lines
    lines_img = full_image.copy()
    for line in lines:
        cv2.line(lines_img, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (0, 255, 255), 2)
    cv2.imwrite("outputs/pipeline/03_all_lines.jpg", cv2.cvtColor(lines_img, cv2.COLOR_RGB2BGR))
    
    # 4. RANSAC Inliers
    inliers_img = draw_inliers(full_image, inliers, lines)
    cv2.imwrite("outputs/pipeline/04_ransac_inliers.jpg", cv2.cvtColor(inliers_img, cv2.COLOR_RGB2BGR))
    
    print("Pipeline images saved in outputs/pipeline/")

if __name__ == "__main__":
    main()