import pykitti
import numpy as np

basedir = '/home/nvidia/Documents/VanishingPointCameraCalibration/visual_odometry/temp_kitti'
date = '2011_09_26'
drive = '0001'

dataset = pykitti.raw(basedir, date, drive)

# The ground truth poses are stored in dataset.oxts
# dataset.oxts is a list of namedtuples with 'T_w_imu'
# T_w_imu is the IMU pose in world frame.
# We want the Camera 0 pose in world frame, relative to the first frame.
# T_w_cam0 = T_w_imu * T_imu_velo * T_velo_cam0 
# Actually, pykitti's dataset.oxts provides T_w_imu.
# We can just get T_cam0_imu = inv(T_velo_cam0) * inv(T_imu_velo)
# But pykitti might already have a helper for this or we can just compute:

poses = []
T_w_cam0_list = []

for oxts in dataset.oxts:
    T_w_imu = oxts.T_w_imu
    # Calib
    T_imu_velo = dataset.calib.T_velo_imu # Actually it's velo to imu or imu to velo?
    # Wait, dataset.calib.T_velo_imu is Velo to IMU ? No, it's IMU to Velo?
    # Let's check pykitti documentation: T_cam0_velo is velo -> cam0.
    # T_velo_imu is imu -> velo.
    # So T_w_cam0 = T_w_imu * T_imu_velo * T_velo_cam0 ?
    # Let's just use T_w_imu because the relative translation of IMU and Camera are practically the same (rigid body).
    # But for rotation, we MUST align the axes!
    
    # IMU to CAM0 transformation matrix
    # camera 0 coordinates: x right, y down, z forward
    # IMU coordinates: x forward, y left, z up
    # T_cam0_imu: transforms from IMU to CAM0.
    # pykitti has T_cam0_imu ! Let's check if it exists:
    try:
        T_cam0_imu = np.linalg.inv(dataset.calib.T_cam0_velo).dot(np.linalg.inv(dataset.calib.T_velo_imu))
    except AttributeError:
        # Just manually define IMU to Camera rotation if T_cam0_velo is not available
        pass
        
    T_w_cam0 = T_w_imu # We will just use the IMU poses but rotated to Camera frame.
    
    T_w_cam0_list.append(T_w_cam0)

# Rotate to Camera frame
# The IMU frame is: x forward, y left, z up
# The Camera frame is: x right, y down, z forward
R_cam_imu = np.array([
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 1,  0,  0]
])
T_cam_imu = np.eye(4)
T_cam_imu[:3, :3] = R_cam_imu

# We want relative poses w.r.t the first frame, in the first frame's Camera coordinate system!
T_w_imu_0 = T_w_cam0_list[0]
T_imu0_w = np.linalg.inv(T_w_imu_0)

with open("/home/nvidia/Documents/VanishingPointCameraCalibration/visual_odometry/KITTI/dataset/poses/00.txt", "w") as f:
    for T_w_imu in T_w_cam0_list:
        # Pose of IMU current in IMU 0 frame
        T_imu0_imucurr = T_imu0_w.dot(T_w_imu)
        
        # We want the pose of CAM current in CAM 0 frame
        # T_cam0_camcurr = T_cam_imu * T_imu0_imucurr * inv(T_cam_imu)
        T_cam0_camcurr = T_cam_imu.dot(T_imu0_imucurr).dot(np.linalg.inv(T_cam_imu))
        
        pose_3x4 = T_cam0_camcurr[:3, :].flatten()
        f.write(" ".join(map(str, pose_3x4)) + "\n")

print("Generated poses/00.txt")
