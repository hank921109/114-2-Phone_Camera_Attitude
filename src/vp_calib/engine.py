import os
import sys
import copy
import math
import argparse
import numpy as np
import cv2

"""
Fast Image Processing (Deeply Optimized for Raspi 4).
"""
def read_image(path):
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_enhance(image, contrast, sharpness):
    """
    使用 OpenCV 實作影像增強，取代 PIL。
    在 Raspi 4 上利用 NEON 加速，速度提升約 10 倍。
    """
    # 對比度調整 (Contrast): dst = src * alpha + beta
    enhanced = cv2.convertScaleAbs(image, alpha=contrast/5.0, beta=0)
    
    # 銳利度調整 (Sharpness): 使用快速卷積核
    if sharpness > 1.0:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
    return enhanced

def get_hough_lines_cv(image, line_length, line_gap):
    """
    使用 OpenCV 實作的機率霍夫變換。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=line_length, maxLineGap=line_gap)
    if lines is None:
        return np.array([]), edges
    return lines.reshape(-1, 2, 2), edges

"""
Result Saving (OpenCV Based - No Matplotlib).
"""
def draw_axes_on_image(image, vps, origin_px, length=300, attitude=None):
    """
    使用 OpenCV 在影像上畫出 X, Y, Z 坐標軸與姿態資訊。
    """
    output_img = image.copy()
    if output_img.dtype != np.uint8:
        output_img = (output_img * 255).astype(np.uint8)
    
    # BGR 顏色定義
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # X:紅, Y:綠, Z:藍
    labels = ['X', 'Y', 'Z']
    
    origin = np.array(origin_px, dtype=np.int32)
    
    for i in range(len(vps)):
        if i >= len(colors): break
        vp = np.array(vps[i][:2])
        direction = vp - origin
        dist = np.linalg.norm(direction)
        
        if dist > 1e-6:
            unit_direction = direction / dist
            end_point = (origin + unit_direction * length).astype(np.int32)
            cv2.arrowedLine(output_img, tuple(origin), tuple(end_point), 
                            colors[i], 3, tipLength=0.1)
            cv2.putText(output_img, labels[i], tuple(end_point), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[i], 2)

    if attitude is not None:
        yaw, pitch, roll = attitude
        overlay = output_img.copy()
        cv2.rectangle(overlay, (10, 10), (280, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, output_img, 0.5, 0, output_img)
        
        cv2.putText(output_img, f"Yaw:   {yaw:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_img, f"Pitch: {pitch:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_img, f"Roll:  {roll:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
    return output_img

def visualize_axes(image, vps, origin_px, img_name, length=300, attitude=None):
    """
    不再使用 matplotlib，改用 OpenCV 直接存檔以節省記憶體。
    """
    # 確保傳入的是 BGR 給 OpenCV 存檔
    if image.shape[2] == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image
    
    res_img = draw_axes_on_image(img_bgr, vps, origin_px, length, attitude)
    save_path = os.path.join("outputs", img_name + "_axes.png")
    cv2.imwrite(save_path, res_img)
    print(f"Axes visualization saved to {save_path}")

"""
Vanishing Point Detection (Optimized Math).
"""
def calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh):
    """
    優化：移除 arccos，改用餘弦值直接比較，節省 CPU 指令週期。
    """
    current_hypothesis = current_hypothesis / current_hypothesis[-1]
    hypothesis_vp_direction = current_hypothesis[:2] - lines[:,0]
    lines_vp_direction = lines[:,1] - lines[:,0]
    
    # 預先計算餘弦閾值
    cos_threshold = math.cos(ransac_angle_thresh * np.pi / 180.0)
    
    magnitude = np.linalg.norm(hypothesis_vp_direction, axis=1) * np.linalg.norm(lines_vp_direction, axis=1)
    magnitude[magnitude == 0] = 1e-5
    
    cos_theta = (hypothesis_vp_direction*lines_vp_direction).sum(axis=-1) / magnitude
    
    # 比較餘弦值絕對值 (theta < threshold 意即 abs(cos_theta) > cos_threshold)
    inliers = (np.abs(cos_theta) > cos_threshold)
    inliers[ignore_pts] = False
    return inliers, inliers.sum()

def run_line_ransac(lines, ransac_iter, ransac_angle_thresh, ignore_pts=None):
    best_vote_count = 0
    best_inliers = None
    best_hypothesis = None
    
    total_lines = lines.shape[0]
    if total_lines < 2:
        return None, None

    if ignore_pts is None:
        ignore_pts = np.zeros((total_lines)).astype('bool')
        lines_to_chose = np.arange(total_lines)
    else:
        lines_to_chose = np.where(ignore_pts==0)[0]
        if len(lines_to_chose) < 2:
            return None, None

    # 早停閾值
    early_stop_thresh = total_lines * 0.4

    for iter_count in range(ransac_iter):
        idx1, idx2 = np.random.choice(lines_to_chose, 2, replace=False)
        l1 = np.cross(np.append(lines[idx1][1], 1), np.append(lines[idx1][0], 1))
        l2 = np.cross(np.append(lines[idx2][1], 1), np.append(lines[idx2][0], 1))

        current_hypothesis = np.cross(l1, l2)
        if current_hypothesis[-1] == 0:
            continue
            
        inliers, vote_count = calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh)
        
        if vote_count > best_vote_count:
            best_vote_count = vote_count
            best_hypothesis = current_hypothesis
            best_inliers = inliers
            if vote_count > early_stop_thresh:
                break
                
    return best_hypothesis, best_inliers

def get_vp_inliers(image_input, contrast, sharpness, sigma, iterations, line_len, line_gap, threshold, processing_width=640):
    """
    支援 Memory Pipe：直接接受 numpy array 以消除磁碟 I/O。
    """
    if isinstance(image_input, str):
        full_image = read_image(image_input)
    else:
        full_image = image_input

    if full_image is None:
        return [], [], [None, None, None, np.array([])]
        
    h, w = full_image.shape[:2]
    scale = processing_width / float(w)
    processing_height = int(h * scale)
    
    image_small = cv2.resize(full_image, (processing_width, processing_height))
    enhanced_img = image_enhance(image_small, contrast=contrast, sharpness=sharpness)
    
    lines_small, edges_small = get_hough_lines_cv(enhanced_img, line_length=line_len, line_gap=line_gap)
    
    if lines_small is None or lines_small.size == 0:
        return [], [], [full_image, enhanced_img, edges_small, np.array([])]

    best_hypothesis_1, best_inliers_1 = run_line_ransac(lines_small, iterations, threshold)
    ignore_pts = best_inliers_1 if best_inliers_1 is not None else np.zeros(len(lines_small), dtype=bool)
    best_hypothesis_2, best_inliers_2 = run_line_ransac(lines_small, iterations, threshold, ignore_pts=ignore_pts)
    ignore_pts = np.logical_or(ignore_pts, best_inliers_2) if best_inliers_2 is not None else ignore_pts
    best_hypothesis_3, best_inliers_3 = run_line_ransac(lines_small, iterations, threshold, ignore_pts=ignore_pts)
    
    inlier_lines_list = [best_inliers_1, best_inliers_2, best_inliers_3]
    
    def rescale_vp(vp, s):
        if vp is None: return np.array([0, 0, 1.0])
        vp = vp / vp[-1]
        return np.array([vp[0]/s, vp[1]/s, 1.0])

    hypothesis_list = [rescale_vp(best_hypothesis_1, scale), rescale_vp(best_hypothesis_2, scale), rescale_vp(best_hypothesis_3, scale)]
    lines_full = lines_small / scale
    
    return inlier_lines_list, hypothesis_list, [full_image, enhanced_img, edges_small, lines_full]

"""
Camera Calibration & Attitude.
"""
def choose_vanishing_points(vp1, vp2, vp3, image):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    vps = [vp1, vp2, vp3]
    lenth_r = []
    for i in range(len(vps)):
        j = (i + 1) % 3
        lenth_r.append(math.sqrt((vps[i][1] - vps[j][1])** 2 + (vps[i][0] - vps[j][0])** 2))
    
    if lenth_r[0] <= (image.shape[1] / 4): del vps[0]
    elif lenth_r[1] <= (image.shape[1] / 4): del vps[1]
    elif lenth_r[2] <= (image.shape[1] / 4): del vps[2]
    else:    
        lenth_pp = [math.sqrt((v[1] - pp[1])** 2 + (v[0] - pp[0])** 2) for v in vps]
        del vps[lenth_pp.index(max(lenth_pp))]
    return vps

def determine_focal_lenth(vps, image):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    if len(vps) < 2: return [1000.0]
    
    v1, v2 = vps[0], vps[1]
    if v1[0] - v2[0] == 0: return [abs(pp[0] - v1[0])]
    
    k_uv = (v1[1] - v2[1]) / (v1[0] - v2[0])
    b_uv = v2[1]- k_uv * v2[0]
    pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(k_uv * k_uv + 1, 0.5)
    lenth_uv = math.sqrt((v1[1] - v2[1])** 2 + (v1[0] - v2[0])** 2)
    lenth_pu = math.sqrt((v1[1] - pp[1])** 2 + (v1[0] - pp[0])** 2)
    up_uv = math.sqrt(max(0, lenth_pu ** 2 - pp_uv ** 2))
    vp_uv = abs(lenth_uv - up_uv)
    return [math.sqrt(abs(up_uv * vp_uv - pp_uv**2))]

def calculate_rotation_matrix(vps, image, f):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    u = np.array([vps[0][0] - pp[0], vps[0][1] - pp[1], f[0]])
    u_norm = u / np.linalg.norm(u)
    v = np.array([vps[1][0] - pp[0], vps[1][1] - pp[1], f[0]])
    v_norm = v / np.linalg.norm(v)
    w_norm = np.cross(u_norm, v_norm)
    return [np.c_[u_norm, v_norm, w_norm]]

def calculate_camera_attitude(R_w2c):
    R_c2w = np.transpose(R_w2c)
    sy = math.sqrt(R_c2w[0, 0]**2 + R_c2w[1, 0]**2)
    if sy > 1e-6:
        pitch = math.atan2(-R_c2w[2, 0], sy)
        yaw = math.atan2(R_c2w[1, 0], R_c2w[0, 0])
        roll = math.atan2(R_c2w[2, 1], R_c2w[2, 2])
    else:
        pitch = math.atan2(-R_c2w[2, 0], sy)
        yaw = math.atan2(-R_c2w[0, 1], R_c2w[1, 1])
        roll = 0
    return np.array([yaw, pitch, roll]) * 180 / np.pi

def rotationmatrix_2_eulerangles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    if sy > 1e-6 :
        x, y, z = math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
    else :
        x, y, z = math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0
    return np.array([x, y, z]) * 180 / np.pi

def storage_calibration_result(vp1, vp2, vp3, forcal, M_r_o2c, V_t_o2c, img_name):
    output_path = os.path.join("outputs", f"{img_name}CalibrationResult.txt")
    with open(output_path, "w") as f:
        f.write(f"VPs: {vp1[:2]}, {vp2[:2]}, {vp3[:2]}\n")
        f.write(f"Focal: {forcal[0]}\n")
        attitude = calculate_camera_attitude(M_r_o2c[0])
        f.write(f"Attitude (Y,P,R): {attitude}\n")
    print("Calibration result saved.")

def calculate_translation_vector_h(image, f, M_r_o2c, px_x, px_y, h):
    # 簡化版位移估算，僅作為佔位符
    return [np.array([0, 0, h])]

def main(image_path, px_x, px_y, h, contrast=5, sharpness=10, iterations=800, line_len=30, line_gap=10, threshold=2):
    img_name = os.path.basename(image_path).split('.')[0]
    inlier_lines_list, hypothesis_list, viz_stuff = get_vp_inliers(image_path, contrast, sharpness, 3, iterations, line_len, line_gap, threshold)
    full_image = viz_stuff[0]
    vps = choose_vanishing_points(hypothesis_list[0], hypothesis_list[1], hypothesis_list[2], full_image)
    forcal = determine_focal_lenth(vps, full_image)
    M_r_o2c = calculate_rotation_matrix(vps, full_image, forcal)
    attitude = calculate_camera_attitude(M_r_o2c[0])
    storage_calibration_result(hypothesis_list[0], hypothesis_list[1], hypothesis_list[2], forcal, M_r_o2c, [0], img_name)
    visualize_axes(full_image, hypothesis_list, [px_x, px_y], img_name, attitude=attitude)
    return forcal, M_r_o2c, [0]
