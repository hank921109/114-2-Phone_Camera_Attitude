import sys, cv2, os
sys.path.append(os.path.abspath('src'))
from vp_calib.engine import get_vp_inliers, estimate_origin_from_inliers
img = "data/samples/rt0.jpg"
frame = cv2.imread(img)
inliers, vps, viz = get_vp_inliers(img, 1.5, 2.0, 3, 3000, 60, 15, 2.0, 960)
origin = estimate_origin_from_inliers(viz[0].shape, inliers, viz[3])
print(f"Calculated Origin: {origin}")
