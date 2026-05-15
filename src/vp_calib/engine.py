import os
import sys
import math
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Any

def read_image(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_enhance(image: np.ndarray, contrast: float, sharpness: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.3, beta=5)

def get_hough_lines_cv(image: np.ndarray, line_length: int, line_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=line_length, maxLineGap=line_gap)
    if lines is None: return np.array([]), edges
    return lines.reshape(-1, 2, 2), edges

def refine_vp_svd(lines: np.ndarray) -> np.ndarray:
    if len(lines) < 2: return np.array([0, 0, 1.0])
    pts1 = np.concatenate([lines[:, 0], np.ones((len(lines), 1))], axis=1)
    pts2 = np.concatenate([lines[:, 1], np.ones((len(lines), 1))], axis=1)
    L = np.cross(pts1, pts2)
    _, _, vh = np.linalg.svd(L)
    vp = vh[-1, :]
    if abs(vp[2]) < 1e-9: return vp
    return vp / vp[2]

def estimate_origin_from_inliers(image_shape: Tuple[int, ...], inlier_masks: List[np.ndarray], lines: np.ndarray) -> List[int]:
    """
    自動估計物理原點：尋找影像下半部 X 軸與 Z 軸線段的交點，並確保其不超出螢幕。
    """
    h, w = image_shape[:2]
    default_origin = [w // 2, int(h * 0.85)]
    
    if len(inlier_masks) < 3 or inlier_masks[0] is None or inlier_masks[2] is None:
        return default_origin
        
    x_lines = lines[inlier_masks[0]]
    z_lines = lines[inlier_masks[2]]
    
    # 過濾：只看影像下半部的線段
    x_bottom = x_lines[np.mean(x_lines[:, :, 1], axis=1) > h * 0.4]
    z_bottom = z_lines[np.mean(z_lines[:, :, 1], axis=1) > h * 0.4]
    
    if len(x_bottom) == 0 or len(z_bottom) == 0:
        return default_origin
        
    # 取最長的前 3 條線求加權交點
    lx_idxs = np.argsort(np.linalg.norm(x_bottom[:, 1] - x_bottom[:, 0], axis=1))[-3:]
    lz_idxs = np.argsort(np.linalg.norm(z_bottom[:, 1] - z_bottom[:, 0], axis=1))[-3:]
    
    intersections = []
    for i in lx_idxs:
        for j in lz_idxs:
            L1 = np.cross(np.append(x_bottom[i, 0], 1), np.append(x_bottom[i, 1], 1))
            L2 = np.cross(np.append(z_bottom[j, 0], 1), np.append(z_bottom[j, 1], 1))
            pt = np.cross(L1, L2)
            if abs(pt[2]) > 1e-9:
                pt /= pt[2]
                # 交點必須在合理範圍內
                if 0 <= pt[0] <= w and h * 0.3 <= pt[1] < h - 10:
                    intersections.append(pt[:2])
    
    if not intersections:
        return default_origin
        
    origin = np.mean(intersections, axis=0).astype(int)
    return [int(origin[0]), int(origin[1])]

def draw_axes_on_image(image: np.ndarray, vps: List[np.ndarray], origin_px: Union[List[int], np.ndarray], 
                       length: int = 300, attitude: Optional[np.ndarray] = None) -> np.ndarray:
    output_img = image.copy()
    h, w = output_img.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # X, Y, Z
    labels = ['X', 'Y', 'Z']
    origin = np.array(origin_px, dtype=np.int32)
    
    for i, vp in enumerate(vps):
        if i >= len(colors) or vp is None or np.all(vp[:2] == 0): continue
        vp_pt = vp[:2]
        direction = vp_pt - origin
        dist = np.linalg.norm(direction)
        
        if dist > 1e-6:
            unit_dir = direction / dist
            
            # --- 自動調整長度，確保不超出螢幕 ---
            # 找出與螢幕邊界的交點，動態縮放 length
            current_len = length
            end_point_raw = origin + unit_dir * current_len
            
            # 邊界限制檢查 (留 20px 邊距給 Label)
            margin = 30
            if end_point_raw[0] < margin or end_point_raw[0] > w - margin or \
               end_point_raw[1] < margin or end_point_raw[1] > h - margin:
                
                # 計算縮放比例
                scale_factors = []
                if unit_dir[0] < 0: scale_factors.append((margin - origin[0]) / unit_dir[0])
                elif unit_dir[0] > 0: scale_factors.append((w - margin - origin[0]) / unit_dir[0])
                if unit_dir[1] < 0: scale_factors.append((margin - origin[1]) / unit_dir[1])
                elif unit_dir[1] > 0: scale_factors.append((h - margin - origin[1]) / unit_dir[1])
                
                if scale_factors:
                    current_len = min(current_len, min(scale_factors))
            
            end_point = (origin + unit_dir * current_len).astype(np.int32)
            cv2.arrowedLine(output_img, tuple(origin), tuple(end_point), colors[i], 4, tipLength=0.1)
            
            # 確保 Label 也在螢幕內
            label_pos = tuple(end_point)
            cv2.putText(output_img, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[i], 3)
            
    if attitude is not None:
        yaw, pitch, roll = attitude
        overlay = output_img.copy()
        cv2.rectangle(overlay, (10, 10), (700, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, output_img, 0.5, 0, output_img)
        cv2.putText(output_img, f"Yaw:   {yaw:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 255), 6)
        cv2.putText(output_img, f"Pitch: {pitch:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 255), 6)
        cv2.putText(output_img, f"Roll:  {roll:.2f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 255), 6)
    return output_img

def run_vectorized_ransac(lines: np.ndarray, iterations: int, threshold: float, 
                          ignore_mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    num_lines = lines.shape[0]
    if num_lines < 2: return None, None
    valid_indices = np.where(~ignore_mask)[0] if ignore_mask is not None else np.arange(num_lines)
    if len(valid_indices) < 2: return None, None
    idx_pairs = np.random.choice(valid_indices, size=(iterations, 2), replace=True)
    pts1 = np.concatenate([lines[idx_pairs[:, 0], 0], np.ones((iterations, 1))], axis=1)
    pts2 = np.concatenate([lines[idx_pairs[:, 0], 1], np.ones((iterations, 1))], axis=1)
    pts3 = np.concatenate([lines[idx_pairs[:, 1], 0], np.ones((iterations, 1))], axis=1)
    pts4 = np.concatenate([lines[idx_pairs[:, 1], 1], np.ones((iterations, 1))], axis=1)
    L1, L2 = np.cross(pts1, pts2), np.cross(pts3, pts4)
    hypotheses = np.cross(L1, L2)
    mask = np.abs(hypotheses[:, 2]) > 1e-8
    hypotheses = hypotheses[mask]
    if len(hypotheses) == 0: return None, None
    hypotheses /= hypotheses[:, 2:3]
    vp_dirs = hypotheses[:, np.newaxis, :2] - lines[np.newaxis, :, 0, :2]
    line_dirs = lines[:, 1, :] - lines[:, 0, :]
    mag_prod = np.linalg.norm(vp_dirs, axis=2) * np.linalg.norm(line_dirs, axis=1)[np.newaxis, :]
    mag_prod[mag_prod == 0] = 1e-5
    cos_theta = np.abs(np.sum(vp_dirs * line_dirs[np.newaxis, :, :], axis=2) / mag_prod)
    inliers_matrix = cos_theta > math.cos(threshold * np.pi / 180.0)
    if ignore_mask is not None: inliers_matrix[:, ignore_mask] = False
    best_idx = np.argmax(np.sum(inliers_matrix, axis=1))
    return hypotheses[best_idx], inliers_matrix[best_idx]

def get_vp_inliers(image_input: Any, contrast: float, sharpness: float, sigma: float, 
                   iterations: int, line_len: int, line_gap: int, threshold: float, 
                   processing_width: int = 640) -> Tuple[List, List, List]:
    full_image = read_image(image_input) if isinstance(image_input, str) else image_input
    if full_image is None: return [], [], [None, None, None, np.array([])]
    h, w = full_image.shape[:2]
    scale = processing_width / float(w)
    image_small = cv2.resize(full_image, (processing_width, int(h * scale)))
    if sigma > 0:
        k = int(sigma * 2) | 1
        image_small = cv2.GaussianBlur(image_small, (k, k), sigma)
    enhanced_img = image_enhance(image_small, contrast, sharpness)
    lines_small, edges_small = get_hough_lines_cv(enhanced_img, line_len, line_gap)
    if lines_small.size == 0: return [], [], [full_image, enhanced_img, edges_small, np.array([])]
    
    _, i1 = run_vectorized_ransac(lines_small, iterations, threshold)
    ignore = i1 if i1 is not None else np.zeros(len(lines_small), dtype=bool)
    _, i2 = run_vectorized_ransac(lines_small, iterations, threshold, ignore_mask=ignore)
    ignore = np.logical_or(ignore, i2) if i2 is not None else ignore
    _, i3 = run_vectorized_ransac(lines_small, iterations, threshold, ignore_mask=ignore)
    
    def r_r(m, s):
        if m is None or np.sum(m) < 2: return np.array([0, 0, 1.0])
        v_s = refine_vp_svd(lines_small[m])
        return np.array([v_s[0]/s, v_s[1]/s, 1.0])

    v_raw = [r_r(m, scale) for m in [i1, i2, i3]]
    i_raw = [i1, i2, i3]
    pp = np.array([full_image.shape[1]/2, full_image.shape[0]/2])
    
    # 智慧型軸向排序：
    z_sc = [abs(np.dot((v[:2]-pp)/np.linalg.norm(v[:2]-pp), [0, 1])) if np.linalg.norm(v[:2]-pp)>1 else 0 for v in v_raw]
    zi = np.argmax(z_sc); zv = v_raw.pop(zi); zi_m = i_raw.pop(zi)
    d_pp = [np.linalg.norm(v[:2]-pp) for v in v_raw]
    xi = np.argmin(d_pp); xv = v_raw.pop(xi); xi_m = i_raw.pop(xi)
    yv = v_raw[0] if v_raw else np.array([0,0,1]); yi_m = i_raw[0] if i_raw else None

    return [xi_m, yi_m, zi_m], [xv, yv, zv], [full_image, enhanced_img, edges_small, lines_small / scale]

class AttitudeSmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha, self.state = alpha, None
    def smooth(self, attitude: np.ndarray) -> np.ndarray:
        self.state = attitude if self.state is None else self.alpha * attitude + (1 - self.alpha) * self.state
        return self.state

def choose_vanishing_points(vps: List[np.ndarray], image: np.ndarray) -> List[np.ndarray]:
    return vps

def determine_focal_length(vps: List[np.ndarray], image: np.ndarray) -> List[float]:
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    v1, v2 = vps[0], vps[2] 
    if np.linalg.norm(v1[:2]) < 1 or np.linalg.norm(v2[:2]) < 1: return [1500.0]
    k = (v1[1]-v2[1])/(v1[0]-v2[0]) if (v1[0]-v2[0])!=0 else 999
    b = v2[1]- k*v2[0]
    p_uv = math.fabs(k*pp[0]-pp[1]+b)/math.pow(k*k+1, 0.5)
    l_uv = math.sqrt((v1[1]-v2[1])**2 + (v1[0]-v2[0])**2)
    l_pu = math.sqrt((v1[1]-pp[1])**2 + (v1[0]-pp[0])**2)
    u_uv = math.sqrt(max(0, l_pu**2 - p_uv**2))
    v_uv = abs(l_uv - u_uv)
    res = math.sqrt(abs(u_uv*v_uv - p_uv**2))
    return [res if 1000 < res < 8000 else 3500.0]

def calculate_rotation_matrix(vps: List[np.ndarray], focal: float, pp: np.ndarray) -> np.ndarray:
    ux = np.array([vps[0][0] - pp[0], vps[0][1] - pp[1], focal])
    ux /= np.linalg.norm(ux)
    uz = np.array([vps[2][0] - pp[0], vps[2][1] - pp[1], focal])
    if uz[1] > 0: uz *= -1
    uz /= np.linalg.norm(uz)
    uy = np.cross(uz, ux)
    uy /= np.linalg.norm(uy)
    uz = np.cross(ux, uy)
    return np.column_stack((ux, uy, uz))

def calculate_camera_attitude(r_m: np.ndarray) -> np.ndarray:
    r_c2w = r_m.T
    sy = math.sqrt(r_c2w[0, 0]**2 + r_c2w[1, 0]**2)
    if sy > 1e-6:
        p, y, r = math.atan2(-r_c2w[2, 0], sy), math.atan2(r_c2w[1, 0], r_c2w[0, 0]), math.atan2(r_c2w[2, 1], r_c2w[2, 2])
    else:
        p, y, r = math.atan2(-r_c2w[2, 0], sy), math.atan2(-r_c2w[0, 1], r_c2w[1, 1]), 0
    return np.array([y, p, r]) * 180 / np.pi

def draw_inliers(image: np.ndarray, inlier_masks: List[np.ndarray], lines: np.ndarray) -> np.ndarray:
    output_img = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    for i, mask in enumerate(inlier_masks):
        if mask is None or i >= len(colors): continue
        v_lines = lines[mask]
        v_lines = sorted(v_lines, key=lambda l: np.linalg.norm(l[1]-l[0]), reverse=True)[:15]
        for line in v_lines:
            cv2.line(output_img, tuple(line[0].astype(int)), tuple(line[1].astype(int)), colors[i], 3)
    return output_img

def main(image_path: str, px_x: float, px_y: float, h: float, **kwargs) -> Tuple[float, np.ndarray, np.ndarray]:
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    inliers, vps, viz = get_vp_inliers(image_path, 1.5, 2.0, 3, 3000, 11, 7, 2, processing_width=1280)
    full_image = viz[0]
    pp = np.array([full_image.shape[1]/2, full_image.shape[0]/2])
    focal = determine_focal_length(vps, full_image)[0]
    rot_matrix = calculate_rotation_matrix(vps, focal, pp)
    attitude = calculate_camera_attitude(rot_matrix)
    origin = estimate_origin_from_inliers(full_image.shape, inliers, viz[3])
    res_img = draw_axes_on_image(full_image, vps, origin, attitude=attitude)
    cv2.imwrite(os.path.join("outputs", f"{img_name}_axes.png"), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
    return focal, rot_matrix, np.array([0, 0, h])
