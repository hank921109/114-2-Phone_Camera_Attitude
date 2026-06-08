from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    """A dataclass to store the evaluation metrics for the visual odometry pipeline.

    Attributes
    ----------
    mse : float
        The mean squared error between the ground truth and predicted poses.
    rmse : float
        The root mean squared error between the ground truth and predicted poses.
    mae : float
        The mean absolute error between the ground truth and predicted poses.
    """

    mse: float
    rmse: float
    mae: float
    rpe: float


class VisualOdometryEvaluator:
    """A class to evaluate the visual odometry pipeline.

    Attributes
    ----------
    gt_pose_list : List[np.ndarray]
        A list of ground truth poses.
    pred_pose_list : List[np.ndarray]
        A list of predicted poses.
    gt_pose : np.ndarray
        The current ground truth pose.
    pred_pose : np.ndarray
        The current predicted pose.
    mse : float
        The mean squared error between the ground truth and predicted poses.
    rmse : float
        The root mean squared error between the ground truth and predicted poses.
    mae : float
        The mean absolute error between the ground truth and predicted poses.
    rpe : float
        The relative pose error (translational drift per frame).
    """

    def __init__(self) -> None:
        self.gt_pose_list = [np.array([0, 0, 0])]
        self.pred_pose_list = [np.array([0, 0, 0])]
        self.gt_pose = np.array([0, 0, 0])
        self.pred_pose = np.array([0, 0, 0])
        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.rpe = 0.0

    def update(self, gt_pose: np.ndarray, pred_pose: np.ndarray) -> None:
        """Update the evaluator with the latest ground truth and predicted poses.

        Parameters
        ----------
        gt_pose : np.ndarray
            A 3x1 ground truth pose vector.
        pred_pose : np.ndarray
            A 3x1 predicted pose vector.
        """

        self.gt_pose = gt_pose
        self.pred_pose = pred_pose
        self.gt_pose_list.append(gt_pose)
        self.pred_pose_list.append(pred_pose)

    def compute(self) -> EvaluationMetrics:
        """Compute the evaluation metrics for the visual odometry pipeline.

        Returns
        -------
        EvaluationMetrics
            A dataclass containing the evaluation metrics.
        """
        self.mse = np.mean(
            (np.array(self.gt_pose_list) - np.array(self.pred_pose_list)) ** 2
        )
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(
            np.abs(np.array(self.gt_pose_list) - np.array(self.pred_pose_list))
        )
        
        # Calculate RPE (Relative Pose Error on translation)
        if len(self.gt_pose_list) > 1:
            gt_diff = np.diff(np.array(self.gt_pose_list), axis=0)
            pred_diff = np.diff(np.array(self.pred_pose_list), axis=0)
            rpe_errors = np.linalg.norm(gt_diff - pred_diff, axis=1)
            self.rpe = np.sqrt(np.mean(rpe_errors ** 2))
        else:
            self.rpe = 0.0
            
        return EvaluationMetrics(self.mse, self.rmse, self.mae, self.rpe)

    def reset(self) -> None:
        """Reset the evaluator."""
        self.gt_pose_list = [np.array([0, 0, 0])]
        self.pred_pose_list = [np.array([0, 0, 0])]
        self.gt_pose = np.array([0, 0, 0])
        self.pred_pose = np.array([0, 0, 0])
        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.rpe = 0.0
