from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from numba import jit

from odo.features import FeatureDetectorAndDescriptor, FeatureMatcher
from odo.motion import estimate_stereo_motion
from odo.stereo_matching import StereoMatcher, disp_to_depth


import math

@jit(nopython=True)
def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([y, x, z]) * 180.0 / math.pi

@dataclass
class Frame:
    """A class to represent a frame in a visual odometry pipeline."""

    image_left: np.ndarray
    image_right: np.ndarray
    disp_map: np.ndarray = None
    depth_map: np.ndarray = None
    matches: List[cv2.DMatch] = None
    kp1: List[cv2.KeyPoint] = None
    dp1: np.ndarray = None
    rmat: np.ndarray = None
    tvec: np.ndarray = None
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        if self.image_left.shape != self.image_right.shape:
            raise ValueError("Images must have the same shape")

        self.height, self.width = self.image_left.shape[:2]

    def compute_disparity_map(self, stereo_matcher: StereoMatcher) -> None:
        if self.disp_map is not None:
            return
        self.disp_map = stereo_matcher(self.image_left, self.image_right)

    def compute_depth_map(self, bf: float):
        if self.disp_map is None:
            raise ValueError("Disparity map must be computed first")
        if self.depth_map is not None:
            return
        self.depth_map = disp_to_depth(self.disp_map, bf)

    def detect_and_compute_features(
        self, feature_detector: FeatureDetectorAndDescriptor
    ):
        if (self.kp1 is not None) and (self.dp1 is not None):
            return
        self.kp1, self.dp1 = feature_detector.detect_and_compute(self.image_left)


class VisualOdometryEstimator:
    """A class to estimate the camera pose using stereo visual odometry.

    The visual odometry pipeline consists of the following steps:
    1. Detect and compute features in the left image at time t.
    2. Compute the disparity map using stereo matching at time t.
    3. Compute the depth map using the disparity map and bf.
    4. Detect and compute features in the left image at time t+1.
    5. Match features between the left images at time t and t+1.
    6. Estimate the relative motion between the two consecutive frames.
    7. Update the camera pose using the estimated motion.

    Parameters
    ----------
    feature_detector : FeatureDetectorAndDescriptor
        A feature detector and descriptor object.
    feature_matcher : FeatureMatcher
        A feature matcher object.
    stereo_matcher : StereoMatcher
        A stereo matcher object.
    camera_matrix : np.ndarray
        3x3 intrinsic camera matrix.
    bf : float
        Baseline times the focal length.

    """

    def __init__(
        self,
        feature_detector: FeatureDetectorAndDescriptor,
        feature_matcher: FeatureMatcher,
        stereo_matcher: StereoMatcher,
        camera_matrix: np.ndarray,
        bf: float,
    ) -> None:
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.stereo_matcher = stereo_matcher
        self.bf = bf
        self.camera_matrix = camera_matrix
        self._is_initialized = False

    def init(self, init_frame: Frame) -> None:
        """Initialize the visual odometry pipeline.

        Parameters
        ----------
        init_frame : Frame
            The initial frame to initialize the pipeline.
        """
        init_frame.detect_and_compute_features(self.feature_detector)
        init_frame.compute_disparity_map(self.stereo_matcher)
        init_frame.compute_depth_map(self.bf)
        self._prev_frame: Frame = init_frame
        self._current_frame: Frame = init_frame
        rmat = init_frame.rmat if init_frame.rmat is not None else np.eye(3)
        tvec = init_frame.tvec if init_frame.tvec is not None else np.zeros(3)
        tmat = np.eye(4)
        tmat[:3, :3] = rmat
        tmat[:3, 3] = tvec.squeeze()
        self._previous_pose = tmat
        self._current_pose: np.ndarray = None
        self._transformation: np.ndarray = None
        self._is_initialized = True

    def set_current_frame(self, frame: Frame) -> None:
        """Set the current frame in the visual odometry pipeline."""
        if self._current_frame is not None:
            self._prev_frame = self._current_frame
        self._current_frame = frame

    def run(self) -> bool:
        """Run the visual odometry pipeline.

        Returns
        -------
        bool
            True if the pipeline ran successfully, False otherwise.
        """
        if not self._is_initialized:
            raise ValueError("Pipeline must be initialized first")

        from concurrent.futures import ThreadPoolExecutor
        
        def track_features():
            if not self._prev_frame.kp1:
                return [], [], []
            p0 = np.array([kp.pt for kp in self._prev_frame.kp1], dtype=np.float32).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self._prev_frame.image_left, 
                self._current_frame.image_left, 
                p0, None, 
                winSize=(21, 21), maxLevel=3, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_new = p1[st == 1]
            old_indices = np.where(st == 1)[0]
            
            tracked_kp = []
            matches = []
            for idx2, (idx1, pt) in enumerate(zip(old_indices, good_new)):
                tracked_kp.append(cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1))
                matches.append(cv2.DMatch(int(idx1), idx2, 0.0))
            return tracked_kp, matches, good_new

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_disp = executor.submit(self._current_frame.compute_disparity_map, self.stereo_matcher)
            future_track = executor.submit(track_features)
            
            future_disp.result()
            tracked_kp, matches, good_new = future_track.result()

        self._current_frame.compute_depth_map(self.bf)
        self._current_frame.matches = matches

        if len(matches) < 10:
            return False

        # Estimate motion
        rmat, tvec = estimate_stereo_motion(
            matches=matches,
            kp1=self._prev_frame.kp1,
            kp2=tracked_kp,
            k_left=self.camera_matrix,
            depth_map=self._prev_frame.depth_map,
        )

        if len(good_new) < 300:
            self._current_frame.kp1 = None
            self._current_frame.dp1 = None
            self._current_frame.detect_and_compute_features(self.feature_detector)
        else:
            self._current_frame.kp1 = tracked_kp

        self._current_frame.rmat = rmat
        self._current_frame.tvec = tvec
        return True

    def compute_transformation(self):
        """Compute the transformation matrix between the current and previous frame."""
        if self._current_frame.rmat is None or self._current_frame.tvec is None:
            return np.eye(4)

        tmat = np.eye(4)
        tmat[:3, :3] = self._current_frame.rmat
        tmat[:3, 3] = self._current_frame.tvec.T
        tmat = np.linalg.inv(tmat)
        self._transformation = tmat

    def update_pose(self):
        """Update the camera pose using the estimated motion."""
        self._current_pose = self._previous_pose @ self._transformation
        self._previous_pose = self._current_pose

    @property
    def current_pose(self):
        """Return the current camera pose."""
        return self._current_pose

    @property
    def transformation(self):
        """Return the transformation matrix between the current and previous frame."""
        return self._transformation

    @property
    def current_translation(self):
        """Return the current translation vector."""
        return self._current_pose[:3, 3]

    @property
    def current_rotation(self):
        """Return the current rotation matrix."""
        return self._current_pose[:3, :3]

    @property
    def current_euler_angles(self):
        """Return the current euler angles (yaw, pitch, roll) in degrees."""
        R = self._current_pose[:3, :3]
        return rotation_to_euler(R)
