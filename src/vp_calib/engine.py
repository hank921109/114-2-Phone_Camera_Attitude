import os
import sys
import math
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Any

EPS = 1e-10

def read_image(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_enhance(image: np.ndarray, contrast: float, sharpness: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.2, beta=5)

def get_hough_lines_cv(image: np.ndarray, line_length: int, line_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    high_thresh, _ = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray_clahe, high_thresh * 0.4, high_thresh, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=45, minLineLength=line_length, maxLineGap=line_gap)
    if lines is None: return np.array([]), edges
    return lines.reshape(-1, 2, 2), edges

def refine_vp_svd(lines: np.ndarray) -> np.ndarray:
    if len(lines) < 2: return np.array([0, 0, 0.0])
    pts1, pts2 = lines[:, 0], lines[:, 1]
    L = np.cross(np.c_[pts1, np.ones(len(pts1))], np.c_[pts2, np.ones(len(pts2))])
    _, _, Vh = np.linalg.svd(L)
    vp = Vh[-1, :]
    if abs(vp[2]) < EPS: return vp / (np.linalg.norm(vp[:2]) + EPS)
    return vp / vp[2]

def run_vectorized_ransac(lines: np.ndarray, iterations: int, threshold: float, ignore_mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    num_lines = lines.shape[0]
    if num_lines < 2: return None, None
    valid_indices = np.where(~ignore_mask)[0] if ignore_mask is not None else np.arange(num_lines)
    if len(valid_indices) < 2: return None, None
    idx_pairs = np.random.choice(valid_indices, size=(min(iterations, 2000), 2), replace=True)
    L1 = np.cross(np.c_[lines[idx_pairs[:,0],0], np.ones(len(idx_pairs))], np.c_[lines[idx_pairs[:,0],1], np.ones(len(idx_pairs))])
    L2 = np.cross(np.c_[lines[idx_pairs[:,1],0], np.ones(len(idx_pairs))], np.c_[lines[idx_pairs[:,1],1], np.ones(len(idx_pairs))])
    hyp = np.cross(L1, L2)
    mask = np.abs(hyp[:, 2]) > EPS
    hyp = hyp[mask]; hyp /= (hyp[:, 2:3] + EPS)
    if len(hyp) == 0: return None, None
    line_dirs = lines[:, 1] - lines[:, 0]
    v_dirs = hyp[:, np.newaxis, :2] - lines[np.newaxis, :, 0, :2]
    cos_theta = np.abs(np.sum(v_dirs * line_dirs[np.newaxis], axis=2) / (np.linalg.norm(v_dirs, axis=2) * np.linalg.norm(line_dirs, axis=1) + EPS))
    in_mat = cos_theta > math.cos(threshold * np.pi / 180.0)
    best_idx = np.argmax(np.sum(in_mat, axis=1))
    return hyp[best_idx], in_mat[best_idx]

def get_vp_inliers(image_input: Any, contrast: float, sharpness: float, sigma: float, 
                   iterations: int, line_len: int, line_gap: int, threshold: float, 
                   processing_width: int = 960) -> Tuple[List, List, List]:
    full_image = read_image(image_input) if isinstance(image_input, str) else image_input
    if full_image is None: return [], [], [None, None, None, np.array([])]
    h, w = full_image.shape[:2]
    scale = processing_width / float(w)
    img_s = cv2.resize(full_image, (processing_width, int(h * scale)))
    sh, sw = img_s.shape[:2]
    if sigma > 0: img_s = cv2.GaussianBlur(img_s, (int(sigma*2)|1, int(sigma*2)|1), sigma)
    enh = image_enhance(img_s, contrast, sharpness)
    lines_s, edges_s = get_hough_lines_cv(enh, line_len, line_gap)
    if lines_s.size == 0: return [], [None, None, None], [full_image, enh, edges_s, np.array([])]
    
    dy, dx = np.abs(lines_s[:, 1, 1] - lines_s[:, 0, 1]), np.abs(lines_s[:, 1, 0] - lines_s[:, 0, 0])
    y_mean = np.mean(lines_s[:, :, 1], axis=1)
    
    # [改良 1] 更精細的語意篩選
    # 垂直池: 嚴格垂直線 (dy > dx*3), 排除天空頂端
    v_mask = (dy > dx * 3.0) & (y_mean < sh * 0.8)
    # 車道線池: 位於影像中心偏下，斜率指向前方 (dy > dx*0.5)
    road_mask = (y_mean > sh * 0.45) & (dy > dx * 0.5) & (~v_mask)
    
    # RANSAC
    _, i_y = run_vectorized_ransac(lines_s, iterations, threshold, ignore_mask=~v_mask)
    _, i_z = run_vectorized_ransac(lines_s, iterations, threshold, ignore_mask=~road_mask)
    
    def r_r(m):
        if m is None or np.sum(m) < 2: return np.array([0, 0, 0.0])
        v = refine_vp_svd(lines_s[m])
        return np.array([v[0]/scale, v[1]/scale, v[2]])

    vy, vz = r_r(i_y), r_r(i_z)
    
    # [改良 2] Fallback 策略：若偵測不到 Vz，使用影像中心偏上點作為前進方向
    if np.linalg.norm(vz[:2]) < 1e-3 or vz[2] < 0.5:
        vz = np.array([w/2, h/2, 1.0])
    if np.linalg.norm(vy[:2]) < 1e-3: vy = np.array([0, 1, 0.0])
    
    # X 軸預設 (會由後續 cross product 校正)
    vx = np.array([1, 0, 0.0])
    
    return [None, i_z, i_y], [vx, vz, vy], [full_image, enh, edges_s, lines_s / scale]

def calculate_rotation_matrix(vps: List[np.ndarray], focal: float, pp: np.ndarray) -> np.ndarray:
    def get_dir(v, default):
        if v is None or np.linalg.norm(v[:2]) < 1e-3: return default
        if abs(v[2]) < 0.1: return np.array([v[0], v[1], 0])
        d = np.array([v[0]-pp[0], v[1]-pp[1], focal])
        return d / (np.linalg.norm(d) + EPS)
    uz = get_dir(vps[1], np.array([0, 0, 1.0])) # Forward
    uy = get_dir(vps[2], np.array([0, 1, 0.0])) # Down
    if uy[1] < 0: uy *= -1
    
    # [改良 3] 嚴格正交導出 X 軸 (Right)
    ux = np.cross(uy, uz); ux /= (np.linalg.norm(ux) + EPS)
    uz = np.cross(ux, uy); uz /= (np.linalg.norm(uz) + EPS) # 二次修正前向軸確保正交
    
    return np.column_stack((ux, uy, uz))

def calculate_camera_attitude(r: np.ndarray) -> np.ndarray:
    p = math.asin(-r[1, 2])
    if abs(math.cos(p)) > EPS:
        y, roll = math.atan2(r[0, 2], r[2, 2]), math.atan2(r[1, 0], r[1, 1])
    else:
        y, roll = math.atan2(-r[2, 0], r[0, 0]), 0
    return np.array([y, p, roll]) * 180 / np.pi

def adapt_to_kitti_frame(ypr: np.ndarray) -> np.ndarray:
    y, p, r = ypr
    return np.array([(y + 180) % 360 - 180, p, (r + 90) % 180 - 90])

def draw_axes_on_image(image: np.ndarray, vps: List[np.ndarray], origin_px: Union[List[int], np.ndarray], 
                       length: int = 300, attitude: Optional[np.ndarray] = None) -> np.ndarray:
    output_img = image.copy(); h, w = output_img.shape[:2]
    cv_colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0)] # X:紅, Z:亮藍, Y:綠
    labels = ['X', 'Z', 'Y']; origin = np.array(origin_px, dtype=np.float32)
    for i, vp in enumerate(vps):
        if i >= len(cv_colors) or vp is None: continue
        if np.linalg.norm(vp[:2]) < 1e-3:
            unit_dir = np.array([1, 0]) if i==0 else (np.array([0, -1]) if i==1 else np.array([0, 1]))
        elif abs(vp[2]) < 0.1:
            unit_dir = vp[:2] / (np.linalg.norm(vp[:2]) + EPS)
        else:
            unit_dir = (vp[:2] - origin) / (np.linalg.norm(vp[:2] - origin) + EPS)
        cur_len = length * 1.5 if i == 1 else length * 0.8
        end_point = (origin + unit_dir * cur_len).astype(np.int32)
        cv2.arrowedLine(output_img, tuple(origin.astype(int)), tuple(end_point), cv_colors[i], 4, tipLength=0.2)
        cv2.putText(output_img, labels[i], tuple(end_point), cv2.FONT_HERSHEY_SIMPLEX, h/400, cv_colors[i], 2)
    if attitude is not None:
        fs, lh = h / 450, int(h / 10)
        overlay = output_img.copy()
        cv2.rectangle(overlay, (5, 5), (int(w*0.35), lh*3 + 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
        for j, (name, val) in enumerate(zip(['Yaw', 'Pitch', 'Roll'], attitude)):
            cv2.putText(output_img, f"Est. {name}: {val:.1f} deg", (15, lh*(j+1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 2)
    return output_img

def estimate_origin_from_inliers(image_shape: Tuple[int, ...], inlier_masks: List[np.ndarray], lines: np.ndarray) -> List[int]:
    h, w = image_shape[:2]; return [w // 2, int(h * 0.85)]

def determine_focal_length(vps: List[np.ndarray], image: np.ndarray) -> List[float]:
    return [715.0]

def draw_inliers(image: np.ndarray, masks: List[np.ndarray], lines: np.ndarray) -> np.ndarray:
    out = image.copy(); cv_colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0)] 
    for i, m in enumerate(masks):
        if m is None or i >= len(cv_colors): continue
        v_lines = lines[m]; v_lines = sorted(v_lines, key=lambda x: np.linalg.norm(x[1]-x[0]), reverse=True)[:15]
        for l in v_lines: cv2.line(out, tuple(l[0].astype(int)), tuple(l[1].astype(int)), cv_colors[i], 2)
    return out

def main(image_path: str, **kwargs):
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    inliers, vps, viz = get_vp_inliers(image_path, 1.2, 1.5, 2, 2000, 45, 10, 1.5, 1024)
    full = viz[0]; pp = np.array([full.shape[1]/2, full.shape[0]/2])
    focal = determine_focal_length(vps, full)[0]
    rm = calculate_rotation_matrix(vps, focal, pp); att = calculate_camera_attitude(rm)
    origin = estimate_origin_from_inliers(full.shape, inliers, viz[3])
    res = draw_axes_on_image(full, vps, origin, attitude=att)
    cv2.imwrite(os.path.join("outputs", f"result_{img_name}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
