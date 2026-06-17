import sys
import os
import time
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from odo.pipeline import VisualOdometryEstimator, Frame
from odo.motion import estimate_stereo_motion
from odo.dataset import KittiDataset
from odo.stereo_matching import StereoMatcher, StereoMatchingConfig
from odo.features import FeatureDetectorAndDescriptor, FeatureDetectorAndDescriptorConfig, FeatureMatcher, FeatureMatcherConfig
from odo.utils import compute_bf, decompose_projection_matrix
from concurrent.futures import ThreadPoolExecutor

timings_list = []

class TimedVisualOdometryEstimator(VisualOdometryEstimator):
    def run(self) -> bool:
        if not self._is_initialized:
            raise ValueError("Pipeline must be initialized first")

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

        t_start = time.time()
        timings = {}

        def timed_disp():
            ts = time.time()
            self._current_frame.compute_disparity_map(self.stereo_matcher)
            return ts, time.time()

        def timed_track():
            ts = time.time()
            res = track_features()
            return ts, time.time(), res

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_disp = executor.submit(timed_disp)
            future_track = executor.submit(timed_track)
            
            ts_disp, te_disp = future_disp.result()
            ts_track, te_track, (tracked_kp, matches, good_new) = future_track.result()

        timings['1. Stereo Matching (Disparity)'] = (ts_disp - t_start, te_disp - t_start)
        timings['2. Feature Tracking (Optical Flow)'] = (ts_track - t_start, te_track - t_start)

        t0 = time.time()
        self._current_frame.compute_depth_map(self.bf)
        t1 = time.time()
        timings['3. Depth Map'] = (t0 - t_start, t1 - t_start)

        self._current_frame.matches = matches
        if len(matches) < 10:
            return False

        t2 = time.time()
        rmat, tvec = estimate_stereo_motion(
            matches=matches,
            kp1=self._prev_frame.kp1,
            kp2=tracked_kp,
            k_left=self.camera_matrix,
            depth_map=self._prev_frame.depth_map,
        )
        t3 = time.time()
        timings['4. Motion Estimation (PnP RANSAC)'] = (t2 - t_start, t3 - t_start)

        t4 = time.time()
        if len(good_new) < 300:
            self._current_frame.kp1 = None
            self._current_frame.dp1 = None
            self._current_frame.detect_and_compute_features(self.feature_detector)
        else:
            self._current_frame.kp1 = tracked_kp
        t5 = time.time()
        
        timings['5. Feature Re-detection (ORB)'] = (t4 - t_start, t5 - t_start)

        self._current_frame.rmat = rmat
        self._current_frame.tvec = tvec

        timings_list.append(timings)
        return True

from pathlib import Path
print("Running pipeline to collect metrics...")
dataset = KittiDataset(Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "KITTI", "dataset"))))
dataset.current_sequence_name = "00"

stereo_matcher = StereoMatcher(StereoMatchingConfig(method="bm", num_disparities=64, block_size=15))
feature_detector = FeatureDetectorAndDescriptor(FeatureDetectorAndDescriptorConfig(detector="orb", descriptor="orb"))
feature_matcher = FeatureMatcher(FeatureMatcherConfig(method="bf", config={"normType": cv2.NORM_HAMMING, "crossCheck": False}, lowe_ratio=0.8))

k_left, r_left, t_left = decompose_projection_matrix(dataset.projections()["P0"])
k_right, r_right, t_right = decompose_projection_matrix(dataset.projections()["P1"])
bf = compute_bf(t_left, t_right, k_left)

pipeline = TimedVisualOdometryEstimator(
    stereo_matcher=stereo_matcher,
    feature_detector=feature_detector,
    feature_matcher=feature_matcher,
    bf=bf,
    camera_matrix=k_left,
)

init_frame = Frame(
    image_left=cv2.imread(dataset[0].left_image.__str__(), cv2.IMREAD_GRAYSCALE),
    image_right=cv2.imread(dataset[0].right_image.__str__(), cv2.IMREAD_GRAYSCALE),
)
pipeline.init(init_frame)

for i in range(1, 100):
    frame = Frame(
        image_left=cv2.imread(dataset[i].left_image.__str__(), cv2.IMREAD_GRAYSCALE),
        image_right=cv2.imread(dataset[i].right_image.__str__(), cv2.IMREAD_GRAYSCALE),
    )
    pipeline.set_current_frame(frame)
    pipeline.run()
    pipeline.compute_transformation()
    pipeline.update_pose()

avg_starts = defaultdict(float)
avg_ends = defaultdict(float)
n_frames = len(timings_list)

for timings in timings_list:
    for k, (s, e) in timings.items():
        avg_starts[k] += s
        avg_ends[k] += e

for k in avg_starts.keys():
    avg_starts[k] /= n_frames
    avg_ends[k] /= n_frames

fig, ax = plt.subplots(figsize=(10, 6))
labels = sorted(list(avg_starts.keys()), reverse=True)
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']

for i, task in enumerate(labels):
    start = avg_starts[task] * 1000 # to ms
    end = avg_ends[task] * 1000 # to ms
    duration = end - start
    ax.barh(i, duration, left=start, height=0.5, align='center', color=colors[i%len(colors)], edgecolor='black')
    ax.text(start + duration / 2, i, f'{duration:.1f} ms', ha='center', va='center', color='black', fontweight='bold')

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('Time (ms)')
ax.set_title('Average Pipeline Latency (Stereo Visual Odometry)')
ax.grid(axis='x', linestyle='--', alpha=0.7)

max_time = max(avg_ends.values()) * 1000
ax.set_xlim(0, max_time * 1.1)
plt.tight_layout()
os.makedirs('../docs/images', exist_ok=True)
plt.savefig('../docs/images/pipeline_gantt.png', dpi=300)
print("Saved Gantt chart to ../docs/images/pipeline_gantt.png")
