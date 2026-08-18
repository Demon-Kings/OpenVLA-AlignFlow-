"""
Kinetic Anomaly Filtering Module.
Computes 1st (velocity) and 2nd (acceleration) derivatives of End-Effector (EEF) trajectories,
identifying and discarding noisy, jittery, and idle teleoperation demonstrations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class KineticJitterFilter:
    def __init__(
        self,
        dt: float = 0.1,
        max_vel_thresh: float = 0.75,
        max_acc_thresh: float = 2.5,
        max_idle_ratio: float = 0.35,
        vel_eps: float = 1e-3,
    ):
        """
        Args:
            dt: Time step duration (seconds)
            max_vel_thresh: Maximum allowable EEF linear velocity (m/s)
            max_acc_thresh: Maximum allowable EEF linear acceleration (m/s^2)
            max_idle_ratio: Maximum allowable fraction of zero-motion idle steps
            vel_eps: Velocity threshold to consider a step 'idle'
        """
        self.dt = dt
        self.max_vel_thresh = max_vel_thresh
        self.max_acc_thresh = max_acc_thresh
        self.max_idle_ratio = max_idle_ratio
        self.vel_eps = vel_eps

    def compute_kinetics(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes velocity, acceleration, and jerk profiles for an EEF position sequence.
        positions: (T, 3) XYZ coordinates in meters.
        Returns:
            velocities: (T-1, 3)
            accelerations: (T-2, 3)
            jerks: (T-3, 3)
        """
        T = len(positions)
        if T < 4:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
        
        # 1st order derivative (Velocity)
        velocities = (positions[1:] - positions[:-1]) / self.dt
        # 2nd order derivative (Acceleration)
        accelerations = (velocities[1:] - velocities[:-1]) / self.dt
        # 3rd order derivative (Jerk)
        jerks = (accelerations[1:] - accelerations[:-1]) / self.dt
        
        return velocities, accelerations, jerks

    def evaluate_trajectory(self, trajectory: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluates whether a trajectory passes quality filters.
        trajectory dictionary must contain:
            - 'eef_pos': np.ndarray of shape (T, 3) [x, y, z]
            - 'actions': np.ndarray of shape (T, action_dim)
        Returns:
            is_clean: bool (True if clean, False if noisy/rejected)
            metrics: dict containing max_vel, max_acc, mean_jerk, idle_ratio
        """
        positions = np.asarray(trajectory.get("eef_pos", []))
        T = len(positions)
        
        if T < 8:
            return False, {"reason": "too_short", "max_vel": 0.0, "max_acc": 0.0, "mean_jerk": 0.0, "idle_ratio": 1.0}
        
        vels, accs, jerks = self.compute_kinetics(positions)
        
        vel_norms = np.linalg.norm(vels, axis=1) if len(vels) > 0 else np.zeros(1)
        acc_norms = np.linalg.norm(accs, axis=1) if len(accs) > 0 else np.zeros(1)
        jerk_norms = np.linalg.norm(jerks, axis=1) if len(jerks) > 0 else np.zeros(1)
        
        max_vel = float(np.max(vel_norms))
        max_acc = float(np.max(acc_norms))
        mean_jerk = float(np.mean(jerk_norms))
        
        # Calculate idle ratio (where velocity is virtually zero)
        idle_steps = np.sum(vel_norms < self.vel_eps)
        idle_ratio = float(idle_steps / len(vel_norms))
        
        # Check criteria
        rejection_reason = "clean"
        if max_vel > self.max_vel_thresh:
            rejection_reason = "overspeed"
        elif max_acc > self.max_acc_thresh:
            rejection_reason = "high_acceleration_jitter"
        elif idle_ratio > self.max_idle_ratio:
            rejection_reason = "excessive_idle"
            
        is_clean = (rejection_reason == "clean")
        
        metrics = {
            "is_clean": is_clean,
            "rejection_reason": rejection_reason,
            "max_vel": max_vel,
            "max_acc": max_acc,
            "mean_jerk": mean_jerk,
            "idle_ratio": idle_ratio,
            "length": T,
        }
        return is_clean, metrics

    def batch_filter(self, trajectories: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Splits raw dataset trajectories into clean expert trajectories and noisy rejected trajectories.
        """
        clean_trajs = []
        noisy_trajs = []
        stats = {
            "total_raw": len(trajectories),
            "clean_count": 0,
            "noisy_count": 0,
            "rejection_reasons": {}
        }
        
        for traj in trajectories:
            is_clean, metrics = self.evaluate_trajectory(traj)
            traj["kinetic_metrics"] = metrics
            if is_clean:
                clean_trajs.append(traj)
            else:
                noisy_trajs.append(traj)
                reason = metrics["rejection_reason"]
                stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
                
        stats["clean_count"] = len(clean_trajs)
        stats["noisy_count"] = len(noisy_trajs)
        stats["rejection_rate"] = stats["noisy_count"] / max(1, stats["total_raw"])
        
        return clean_trajs, noisy_trajs, stats
