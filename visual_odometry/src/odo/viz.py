from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from odo.metrics import EvaluationMetrics


class Plotter:
    """A class to visualize the stereo visual odometry pipeline using matplotlib.

    The plotter creates a 3x2 grid of subplots to visualize the following:
    | Left Image | Right Image |
    | Disparity  | 3D Trajectory spanning two rows |
    | Error Metrics | 3D Trajectory |

    Parameters
    ----------
    min_pose : np.ndarray
        A 3D array (x, y, z) representing the minimum pose values.
    max_pose : np.ndarray
        A 3D array (x, y, z) representing the maximum pose values.
    """

    def __init__(self, min_pose: np.ndarray, max_pose: np.ndarray) -> None:
        self.fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(4, 3, height_ratios=[1.5, 1, 1, 1])

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        
        self.ax4 = self.fig.add_subplot(gs[1:, 0:2], projection="3d")
        self.ax5 = self.fig.add_subplot(gs[1, 2])
        self.ax6 = self.fig.add_subplot(gs[2, 2])
        self.ax7 = self.fig.add_subplot(gs[3, 2])

        self.ax5.set_xlabel("Frame")
        self.ax5.set_ylabel("Error (m)")
        self.ax5.set_title("Visual Odometry Metrics")
        self.ax5.grid(True)

        self.ax6.set_xlabel("Frame")
        self.ax6.set_ylabel("Degrees")
        self.ax6.set_title("Euler Angles")
        self.ax6.grid(True)

        self.ax7.set_xlabel("Frame")
        self.ax7.set_ylabel("FPS")
        self.ax7.set_title("Processing FPS")
        self.ax7.grid(True)

        self.ax4.set_xlabel("X")
        self.ax4.set_ylabel("Y")
        self.ax4.set_zlabel("Z")
        self.ax4.view_init(-20, 270)

        min_x, max_x = min_pose[0], max_pose[0]
        min_y, max_y = min_pose[1], max_pose[1]
        min_z, max_z = min_pose[2], max_pose[2]

        self.ax4.set_xlim(min_x - 10, max_x + 10)
        self.ax4.set_ylim(min_y - 10, max_y + 10)
        self.ax4.set_zlim(min_z - 10, max_z + 10)

        # Initialize the plots with empty data.
        (self.gt_plot,) = self.ax4.plot(
            [], [], [], color="blue", markersize=1, label="Ground Truth"
        )
        (self.pred_plot,) = self.ax4.plot(
            [], [], [], color="red", markersize=1, label="Prediction"
        )
        self.idx = 0
        self.fig.tight_layout()

    def update(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        disp_map: np.ndarray,
        gt_pose: np.ndarray,
        pred_pose: np.ndarray,
        metrics: EvaluationMetrics,
        ypr: np.ndarray = None,
        fps: float = None,
    ) -> None:
        """Update the subplots with the latest images, disparity map, and poses.

        Parameters
        ----------
        left_image : np.ndarray
            A HxW grayscale image from the left camera.
        right_image : np.ndarray
            A HxW grayscale image from the right camera.
        disp_map : np.ndarray
            A HxW disparity map.
        gt_pose : np.ndarray
            A Nx3 array representing the ground truth poses.
        pred_pose : np.ndarray
            A Nx3 array representing the predicted poses.
        metrics : EvaluationMetrics
            A class containing error metrics (mse, rmse, mae).
        """
        self.idx += 1
        self.ax1.imshow(left_image, cmap="gray")
        self.ax2.imshow(right_image, cmap="gray")
        self.ax3.imshow(disp_map, cmap="jet")

        self.gt_plot.set_data(gt_pose[:, 0], gt_pose[:, 1])
        self.gt_plot.set_3d_properties(gt_pose[:, 2])
        self.pred_plot.set_data(pred_pose[:, 0], pred_pose[:, 1])
        self.pred_plot.set_3d_properties(pred_pose[:, 2])

        self.ax5.scatter(self.idx, metrics.mse, color="red", label="MSE" if self.idx == 1 else "")
        self.ax5.scatter(self.idx, metrics.rmse, color="blue", label="RMSE" if self.idx == 1 else "")
        self.ax5.scatter(self.idx, metrics.mae, color="green", label="MAE" if self.idx == 1 else "")
        self.ax5.scatter(self.idx, metrics.rpe, color="purple", label="RPE" if self.idx == 1 else "")
        if self.idx == 1:
            self.ax5.legend(loc="upper left")

        if ypr is not None:
            self.ax6.scatter(self.idx, ypr[0], color="red", label="Yaw" if self.idx == 1 else "")
            self.ax6.scatter(self.idx, ypr[1], color="blue", label="Pitch" if self.idx == 1 else "")
            self.ax6.scatter(self.idx, ypr[2], color="green", label="Roll" if self.idx == 1 else "")
            if self.idx == 1:
                self.ax6.legend(loc="upper left")

        if fps is not None:
            self.ax7.scatter(self.idx, fps, color="purple", label="FPS" if self.idx == 1 else "")
            if self.idx == 1:
                self.ax7.legend(loc="upper left")

        self.ax1.axis("off")
        self.ax2.axis("off")
        self.ax3.axis("off")
        self.ax1.set_title("Left Image")
        self.ax2.set_title("Right Image")
        self.ax3.set_title("Disparity")
        plt.pause(0.000000000000000001)


class VideoSaver:
    """A class to save matplotlib figures as a video.

    Parameters
    ----------
    filename : str
        The name of the output video file.
    fps : int
        Frames per second of the output video.
    frame_size : Tuple[int, int]
        A tuple containing the width and height of the video frame.

    """

    def __init__(self, filename: str, fps: int, frame_size: Tuple[int, int]) -> None:
        self.filename = filename
        self.fps = fps
        self._current_frame: np.ndarray = None
        self.frames = []

    @property
    def current_frame(self) -> np.ndarray:
        """Return the current frame."""
        return self._current_frame

    def write_frame(self, fig: plt.Figure) -> None:
        """Write the current matplotlib figure to the video.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure to write to the video.
        """
        # Draw the canvas, then convert it to an image.
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        
        # Ensure all frames have the exact same shape
        if self.frames:
            expected_shape = self.frames[0].shape
            if img.shape != expected_shape:
                img = cv2.resize(img, (expected_shape[1], expected_shape[0]))
                
        self._current_frame = img
        self.frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def release(self) -> None:
        """Release the video writer."""
        if self.frames:
            import imageio.v2 as imageio
            imageio.mimsave(
                self.filename,
                self.frames,
                fps=self.fps,
                loop=0
            )
