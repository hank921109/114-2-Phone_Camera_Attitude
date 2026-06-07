import sys
import os
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from vp_calib.engine import get_vp_inliers, draw_inliers

def main():
    img_path = "data/samples/rt1.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    # parameters from the filename: rt1_inliers_iter3000_thresh2_sigma5_hlen11_hgap7.png
    iterations = 3000
    threshold = 2.0
    sigma = 5.0
    line_len = 11
    line_gap = 7
    
    inliers, vps, viz_stuff = get_vp_inliers(
        img_path, contrast=1.5, sharpness=2.0, sigma=sigma,
        iterations=iterations, line_len=line_len, line_gap=line_gap, threshold=threshold,
        processing_width=960
    )
    
    full_image, enhanced_img, edges_small, lines = viz_stuff
    
    # 3. All Detected Lines
    lines_img = full_image.copy()
    if lines is not None and len(lines) > 0:
        for line in lines:
            cv2.line(lines_img, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (0, 255, 255), 1)
            
    # 4. RANSAC Inliers
    inliers_img = draw_inliers(full_image, inliers, lines)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vanishing Point Pipeline (rt1, iter=3000, thresh=2, sigma=5, hlen=11, hgap=7)', fontsize=16)
    
    axes[0, 0].imshow(enhanced_img)
    axes[0, 0].set_title('1. Enhanced Image (CLAHE + GaussianBlur)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edges_small, cmap='gray')
    axes[0, 1].set_title('2. Canny Edges')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(lines_img)
    axes[1, 0].set_title('3. Hough Lines')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(inliers_img)
    axes[1, 1].set_title('4. RANSAC Inliers (Clustered)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    out_dir = "docs/images"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "rt1_inliers_iter3000_thresh2_sigma5_hlen11_hgap7.png")
    plt.savefig(out_file, dpi=150)
    print(f"Saved multiple subplots to {out_file}")

if __name__ == "__main__":
    main()
