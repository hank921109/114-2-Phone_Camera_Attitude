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
    rpe_rot: float
    mae_rot: float
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
        self.gt_pose_list = [np.eye(4)[:3, :]]
        self.pred_pose_list = [np.eye(4)[:3, :]]
        self.mse = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.rpe = 0.0
        self.rpe_rot = 0.0
        self.mae_rot = 0.0

    def update(self, gt_pose: np.ndarray, pred_pose: np.ndarray) -> None:
        """Update the evaluator with the latest ground truth and predicted poses.

        Parameters
        ----------
        gt_pose : np.ndarray
            A 3x1 ground truth pose vector.
        pred_pose : np.ndarray
            A 3x1 predicted pose vector.
        """

        self.gt_pose_list.append(gt_pose)
        self.pred_pose_list.append(pred_pose)

    def compute(self) -> EvaluationMetrics:
        """Compute the evaluation metrics for the visual odometry pipeline.

        Returns
        -------
        EvaluationMetrics
            A dataclass containing the evaluation metrics.
        """
        gt_trans = np.array([p[:3, 3] for p in self.gt_pose_list])
        pred_trans = np.array([p[:3, 3] for p in self.pred_pose_list])
        
        self.mse = np.mean((gt_trans - pred_trans) ** 2)
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(gt_trans - pred_trans))
        
        # Calculate RPE (Relative Pose Error on translation)
        if len(self.gt_pose_list) > 1:
            gt_diff = np.diff(gt_trans, axis=0)
            pred_diff = np.diff(pred_trans, axis=0)
            rpe_errors = np.linalg.norm(gt_diff - pred_diff, axis=1)
            self.rpe = np.sqrt(np.mean(rpe_errors ** 2))
            
            # Calculate Rotational RPE and MAE
            rot_rpe_list = []
            rot_mae_list = []
            import cv2
            for i in range(1, len(self.gt_pose_list)):
                gt_R_prev = self.gt_pose_list[i-1][:3, :3]
                gt_R_curr = self.gt_pose_list[i][:3, :3]
                gt_R_rel = gt_R_curr.dot(gt_R_prev.T)
                
                pred_R_prev = self.pred_pose_list[i-1][:3, :3]
                pred_R_curr = self.pred_pose_list[i][:3, :3]
                pred_R_rel = pred_R_curr.dot(pred_R_prev.T)
                
                R_err_rel = gt_R_rel.T.dot(pred_R_rel)
                vec_rel, _ = cv2.Rodrigues(R_err_rel)
                rot_rpe_list.append(np.linalg.norm(vec_rel))
                
                R_err_abs = self.gt_pose_list[i][:3, :3].T.dot(self.pred_pose_list[i][:3, :3])
                vec_abs, _ = cv2.Rodrigues(R_err_abs)
                rot_mae_list.append(np.linalg.norm(vec_abs))
                
            self.rpe_rot = np.mean(rot_rpe_list) * 180.0 / np.pi
            self.mae_rot = np.mean(rot_mae_list) * 180.0 / np.pi
        else:
            self.rpe = 0.0
            self.rpe_rot = 0.0
            self.mae_rot = 0.0
            
        return EvaluationMetrics(self.mse, self.rmse, self.mae, self.rpe, self.rpe_rot, self.mae_rot)

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
