import cv2
import tensorflow as tf
import numpy as np
from helpers.utils.visualization_utils import (
    draw_prediction_on_image_simple,
    draw_prediction_on_image_adaptive,
    draw_prediction_on_image_enhanced,
    draw_prediction_on_image_bench_press,
    EDGE_COLORS,
)
from .feedback_utils import PoseFeedback, draw_comprehensive_feedback_overlay

def interpolate_keypoints(prev_kp, curr_kp, next_kp):
    """Deprecated: temporal interpolation removed (no spatio-temporal smoothing)."""
    return curr_kp


class PoseProcessor:
    # Store locked y for knees and ankles during squat bottom
    _locked_leg_y = None

    # Store knee y-coordinates at standing phase
    _standing_knee_y = None

    """Handles pose detection processing with improved stability and squat form analysis."""
    
    def __init__(self, movenet_model, input_size, kalman_enabled=False):
        self.movenet = movenet_model
        self.input_size = input_size
        self.feedback = PoseFeedback()

        # Smoothing options
        self.spatiotemporal_enabled = False  # Spatio temporal smoothing toggle
        self.kalman_enabled = False  # Kalman filter smoothing toggle

        # Pure spatio-temporal smoothing controls
        self.temporal_alpha = 0.22    # more weight to previous for stability
        self.spatial_beta = 0.12      # moderate spatial smoothing
        self.conf_threshold = 0.15    # min confidence to trust current point
        self.spatial_min_conf = 0.15  # min confidence to include neighbor
        self._prev_smoothed = None    # (17,3) last smoothed
        # Build adjacency list from skeleton edges
        self._neighbors = self._build_neighbors()
        # Outlier guard history (store last N smoothed frames for robust stats)
        self._hist_len = 6  # Longer history for better median filtering
        self._hist = []  # list of (17,2) arrays of smoothed yx
        self._mad_k = 4.0  # looser threshold for outlier rejection
        self._pixel_step_frac = 0.10  # allow more per-frame movement
        self.lead_gain = 0.0          # disable lookahead (causes overshoot with noisy data)
        self.last_frame_shape = None

        # Movement consistency tracking
        self._velocity_history = []  # Track velocity for consistency
        self._velocity_hist_len = 3

        # Kalman filter state for each keypoint (17 keypoints, 2D)
        self._kalman_filters = None  # Will be initialized on first use

    def _init_kalman_filters(self, initial_kps):
        # Each keypoint gets its own Kalman filter (x and y)
        # State: [x, y, dx, dy]
        self._kalman_filters = []
        for i in range(17):
            kf = {
                'x': float(initial_kps[i, 0]),
                'y': float(initial_kps[i, 1]),
                'dx': 0.0,
                'dy': 0.0,
                'P': np.eye(4) * 1e-3,  # Covariance
            }
            self._kalman_filters.append(kf)

    def _kalman_predict_update(self, kf, meas_x, meas_y, conf, dt=1.0):
        # Simple constant velocity Kalman filter for 2D point
        # State: [x, y, dx, dy]
        # Predict
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        Q = np.eye(4) * 1e-5  # Process noise
        x = np.array([kf['x'], kf['y'], kf['dx'], kf['dy']])
        P = kf['P']
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Measurement
        z = np.array([meas_x, meas_y])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R_base = 1e-3
        R = np.eye(2) * (R_base + (1.0 - conf) * 2e-2)  # More noise if low conf

        # Kalman gain
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        y = z - (H @ x_pred)
        x_new = x_pred + K @ y
        P_new = (np.eye(4) - K @ H) @ P_pred

        # Save state
        kf['x'], kf['y'], kf['dx'], kf['dy'] = x_new
        kf['P'] = P_new
        return kf['x'], kf['y']

    def smooth_keypoints_kalman(self, keypoints_with_scores):
        """Apply Kalman filter smoothing to keypoints. Expects shape (1,1,17,3) with (y,x,score)."""
        if keypoints_with_scores is None:
            return keypoints_with_scores
        try:
            kps = keypoints_with_scores[0, 0, :, :].copy()  # (17,3)
            # Detect squat phase from feedback (if available)
            squat_phase = None
            if hasattr(self, 'feedback') and hasattr(self.feedback, 'last_feedback'):
                last_feedback = getattr(self.feedback, 'last_feedback', None)
                if last_feedback and isinstance(last_feedback, dict):
                    squat_phase = last_feedback.get('phase', None)

            # Detect hip-knee parallel depth (hip y ≈ knee y)
            hip_y = (kps[11, 0] + kps[12, 0]) / 2
            knee_y = (kps[13, 0] + kps[14, 0]) / 2
            parallel_depth = abs(hip_y - knee_y) < 0.02  # 2% of image height (normalized)

            # Lock knees and ankles y at parallel depth during descending phase
            if squat_phase == "descending" and parallel_depth:
                # Lock current y for knees and ankles (only prevent upward movement)
                if self._locked_leg_y is None:
                    self._locked_leg_y = {
                        13: kps[13, 0],  # left knee
                        14: kps[14, 0],  # right knee
                        15: kps[15, 0],  # left ankle
                        16: kps[16, 0],  # right ankle
                    }
            # Unlock when ascending or standing
            if squat_phase in ["ascending", "standing"]:
                self._locked_leg_y = None

            # If locked, prevent knees/ankles from moving upwards (lower y)
            if self._locked_leg_y:
                for idx in [13, 14, 15, 16]:
                    # Only clamp if new y is less than locked y (i.e., moving upwards)
                    if kps[idx, 0] < self._locked_leg_y[idx]:
                        kps[idx, 0] = self._locked_leg_y[idx]

            # Record knee y at standing phase
            if squat_phase == "standing":
                self._standing_knee_y = (kps[13, 0], kps[14, 0])  # left, right knee y

            # Clamp knee y so it cannot go higher than standing value
            if self._standing_knee_y:
                # Only clamp if not in standing phase
                if squat_phase != "standing":
                    left_standing_y, right_standing_y = self._standing_knee_y
                    if kps[13, 0] < left_standing_y:
                        kps[13, 0] = left_standing_y
                    if kps[14, 0] < right_standing_y:
                        kps[14, 0] = right_standing_y

            conf = kps[:, 2].copy()
            # ...existing code using kps, conf, squat_phase, clamping, locking, etc...
            # (All logic that uses kps is now inside this try block)
            if self._kalman_filters is None:
                self._init_kalman_filters(kps)
            # ...existing Kalman smoothing and clamping logic...
            arm_indices = [7, 8, 9, 10]
            # Only clamp wrist y below elbow for squat, not deadlift
            mode = getattr(self.feedback, 'exercise_mode', 'squat')
            if mode == 'squat':
                left_elbow_y = kps[7, 0]
                right_elbow_y = kps[8, 0]
                if kps[9, 0] > left_elbow_y:
                    kps[9, 0] = left_elbow_y
                if kps[10, 0] > right_elbow_y:
                    kps[10, 0] = right_elbow_y
            # Clamp wrist x so it cannot cross to the opposite side of the elbow (relative to previous forearm direction)
            prev_left_elbow_x = self._kalman_filters[7]['y']
            prev_left_wrist_x = self._kalman_filters[9]['y']
            curr_left_elbow_x = kps[7, 1]
            curr_left_wrist_x = kps[9, 1]
            prev_left_dir = np.sign(prev_left_wrist_x - prev_left_elbow_x)
            curr_left_dir = np.sign(curr_left_wrist_x - curr_left_elbow_x)
            if prev_left_dir != 0 and curr_left_dir != 0 and prev_left_dir != curr_left_dir:
                kps[9, 1] = curr_left_elbow_x
            prev_right_elbow_x = self._kalman_filters[8]['y']
            prev_right_wrist_x = self._kalman_filters[10]['y']
            curr_right_elbow_x = kps[8, 1]
            curr_right_wrist_x = kps[10, 1]
            prev_right_dir = np.sign(prev_right_wrist_x - prev_right_elbow_x)
            curr_right_dir = np.sign(curr_right_wrist_x - curr_right_elbow_x)
            if prev_right_dir != 0 and curr_right_dir != 0 and prev_right_dir != curr_right_dir:
                kps[10, 1] = curr_right_elbow_x
            # ...existing Kalman update loop...
            shoulder_indices = [5, 6]
            hips = kps[[11, 12], 1]
            knees = kps[[13, 14], 1]
            squat_bottom = np.all(np.abs(hips - knees) < 0.08)
            left_arm_conf = (conf[7] + conf[9]) / 2.0
            right_arm_conf = (conf[8] + conf[10]) / 2.0
            mirror_threshold = 0.18
            prev_sh_x = [self._kalman_filters[5]['x'], self._kalman_filters[6]['x']]
            prev_sh_y = [self._kalman_filters[5]['y'], self._kalman_filters[6]['y']]
            prev_arm_x = [self._kalman_filters[7]['x'], self._kalman_filters[8]['x']]
            prev_arm_y = [self._kalman_filters[7]['y'], self._kalman_filters[8]['y']]
            prev_wrist_x = [self._kalman_filters[9]['x'], self._kalman_filters[10]['x']]
            prev_wrist_y = [self._kalman_filters[9]['y'], self._kalman_filters[10]['y']]
            left_elbow_offset = [prev_arm_x[0] - prev_sh_x[0], prev_arm_y[0] - prev_sh_y[0]]
            right_elbow_offset = [prev_arm_x[1] - prev_sh_x[1], prev_arm_y[1] - prev_sh_y[1]]
            left_wrist_offset = [prev_wrist_x[0] - prev_sh_x[0], prev_wrist_y[0] - prev_sh_y[0]]
            right_wrist_offset = [prev_wrist_x[1] - prev_sh_x[1], prev_wrist_y[1] - prev_sh_y[1]]
            freeze_thresh = 0.045
            prev_pos = {idx: (self._kalman_filters[idx]['x'], self._kalman_filters[idx]['y']) for idx in [7,8,9,10,13,14]}
            for i in range(17):
                x, y = kps[i, 0], kps[i, 1]
                c = conf[i]
                predict_arm = False
                left_elbow_dist = np.linalg.norm(kps[7, :2] - kps[13, :2])
                left_wrist_dist = np.linalg.norm(kps[9, :2] - kps[13, :2])
                if (left_elbow_dist < freeze_thresh or left_wrist_dist < freeze_thresh):
                    if conf[7] < 0.25 or conf[9] < 0.25 or conf[13] < 0.25:
                        if i in [7, 9]:
                            predict_arm = True
                right_elbow_dist = np.linalg.norm(kps[8, :2] - kps[14, :2])
                right_wrist_dist = np.linalg.norm(kps[10, :2] - kps[14, :2])
                if (right_elbow_dist < freeze_thresh or right_wrist_dist < freeze_thresh):
                    if conf[8] < 0.25 or conf[10] < 0.25 or conf[14] < 0.25:
                        if i in [8, 10]:
                            predict_arm = True
                if predict_arm:
                    if i in [7, 9]:
                        sh_idx = 5
                    else:
                        sh_idx = 6
                    prev_sh_x = self._kalman_filters[sh_idx]['x']
                    prev_sh_y = self._kalman_filters[sh_idx]['y']
                    prev_arm_x = self._kalman_filters[i]['x']
                    prev_arm_y = self._kalman_filters[i]['y']
                    offset_x = prev_arm_x - prev_sh_x
                    offset_y = prev_arm_y - prev_sh_y
                    xk = kps[sh_idx, 0] + offset_x
                    yk = kps[sh_idx, 1] + offset_y
                    xk = 0.7 * xk + 0.3 * prev_arm_x
                    yk = 0.7 * yk + 0.3 * prev_arm_y
                    kps[i, 0] = xk
                    kps[i, 1] = yk
                    continue
                if i in arm_indices:
                    if (left_arm_conf > right_arm_conf + 0.10) and (right_arm_conf < mirror_threshold) and (i in [8, 10]):
                        sh_x = self._kalman_filters[6]['x']
                        sh_y = self._kalman_filters[6]['y']
                        if i == 8:
                            xk = sh_x + left_elbow_offset[0]
                            yk = sh_y + left_elbow_offset[1]
                        else:
                            xk = sh_x + left_wrist_offset[0]
                            yk = sh_y + left_wrist_offset[1]
                        kps[i, 0] = xk
                        kps[i, 1] = yk
                        continue
                    elif (right_arm_conf > left_arm_conf + 0.10) and (left_arm_conf < mirror_threshold) and (i in [7, 9]):
                        sh_x = self._kalman_filters[5]['x']
                        sh_y = self._kalman_filters[5]['y']
                        if i == 7:
                            xk = sh_x + right_elbow_offset[0]
                            yk = sh_y + right_elbow_offset[1]
                        else:
                            xk = sh_x + right_wrist_offset[0]
                            yk = sh_y + right_wrist_offset[1]
                        kps[i, 0] = xk
                        kps[i, 1] = yk
                        continue
                    if squat_bottom and c < 0.35:
                        if i in [7, 9]:
                            sh_idx = 5
                        else:
                            sh_idx = 6
                        prev_sh_xi = self._kalman_filters[sh_idx]['x']
                        prev_sh_yi = self._kalman_filters[sh_idx]['y']
                        prev_arm_xi = self._kalman_filters[i]['x']
                        prev_arm_yi = self._kalman_filters[i]['y']
                        offset_x = prev_arm_xi - prev_sh_xi
                        offset_y = prev_arm_yi - prev_sh_yi
                        xk = prev_sh_xi + np.clip(offset_x, -0.18, 0.18)
                        yk = prev_sh_yi + np.clip(offset_y, -0.18, 0.18)
                    else:
                        if c < 0.20:
                            xk, yk = self._kalman_predict_update(self._kalman_filters[i], self._kalman_filters[i]['x'], self._kalman_filters[i]['y'], 0.01)
                        else:
                            prev_x, prev_y = self._kalman_filters[i]['x'], self._kalman_filters[i]['y']
                            xk, yk = self._kalman_predict_update(self._kalman_filters[i], x, y, min(c, 0.7))
                            max_move = 0.04
                            dx = np.clip(xk - prev_x, -max_move, max_move)
                            dy = np.clip(yk - prev_y, -max_move, max_move)
                            xk = prev_x + dx
                            yk = prev_y + dy
                    kps[i, 0] = xk
                    kps[i, 1] = yk
                else:
                    if c < 0.10:
                        xk, yk = self._kalman_predict_update(self._kalman_filters[i], self._kalman_filters[i]['x'], self._kalman_filters[i]['y'], 0.01)
                    else:
                        xk, yk = self._kalman_predict_update(self._kalman_filters[i], x, y, c)
                    kps[i, 0] = xk
                    kps[i, 1] = yk
            # Enforce knee/ankle upward lock after Kalman update loop
            if self._locked_leg_y:
                for idx in [13, 14, 15, 16]:
                    if kps[idx, 0] < self._locked_leg_y[idx]:
                        kps[idx, 0] = self._locked_leg_y[idx]
            out = keypoints_with_scores.copy()
            out[0, 0, :, :] = kps
            return out
        except Exception:
            return keypoints_with_scores

    def _build_neighbors(self):
        neighbors = {i: set() for i in range(17)}
        try:
            for (a, b) in EDGE_COLORS.keys():
                neighbors[a].add(b)
                neighbors[b].add(a)
        except Exception:
            pass
        return {i: sorted(list(s)) for i, s in neighbors.items()}
    
    def get_keypoint_confidence(self, keypoints, kp_idx):
        """Return confidence for a keypoint; kept for API compatibility."""
        try:
            if keypoints.shape[1] == 3:
                return float(keypoints[kp_idx, 2])
            elif keypoints.shape[1] == 1:
                return float(keypoints[kp_idx, 0])
            else:
                return 1.0
        except (IndexError, TypeError):
            return 1.0
    
    def get_keypoint_coords(self, keypoints, kp_idx):
        """Get coordinates for a keypoint, handling different formats."""
        try:
            if keypoints.shape[1] == 3:
                # Format: (x, y, confidence)
                return float(keypoints[kp_idx, 0]), float(keypoints[kp_idx, 1])
            elif keypoints.shape[1] == 1:
                # This might be just confidence scores, need to check actual format
                # For now, return dummy coordinates
                return 0.5, 0.5
            else:
                # Unknown format
                return 0.5, 0.5
        except (IndexError, TypeError):
            return 0.5, 0.5
    
    def calculate_movement(self, current_kp, previous_kp):
        """Deprecated: movement-based smoothing removed."""
        return np.zeros(current_kp.shape[0])
    
    def smooth_keypoints_ema(self, keypoints_with_scores):
        """Pure spatio-temporal smoothing: blend current with previous smoothed (temporal)
        and neighbor average (spatial). Expects shape (1,1,17,3) with (y,x,score)."""
        if not self.spatiotemporal_enabled or keypoints_with_scores is None:
            return keypoints_with_scores

        kps = keypoints_with_scores[0, 0, :, :].copy()  # (17,3)
        if self._prev_smoothed is None:
            # First frame: initialize
            self._prev_smoothed = kps.copy()
            return keypoints_with_scores

        conf = kps[:, 2].copy()
        prev = self._prev_smoothed.copy()
        alphaT = float(self.temporal_alpha)
        betaS = float(self.spatial_beta)
        gammaC = max(0.0, 1.0 - alphaT - betaS)  # weight for current

        # Compute neighbor averages for spatial term
        neigh_avg = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            neigh = self._neighbors.get(i, [])
            pts = []
            for j in neigh:
                pts.append([kps[j, 0], kps[j, 1]])
            if pts:
                pts_arr = np.array(pts, dtype=np.float32)
                neigh_avg[i, 0] = float(np.mean(pts_arr[:, 0]))
                neigh_avg[i, 1] = float(np.mean(pts_arr[:, 1]))
            else:
                neigh_avg[i, 0] = prev[i, 0]
                neigh_avg[i, 1] = prev[i, 1]

        # Basic EMA smoothing: blend current, previous, and neighbor average
        for i in range(17):
            kps[i, 0] = gammaC * kps[i, 0] + alphaT * prev[i, 0] + betaS * neigh_avg[i, 0]
            kps[i, 1] = gammaC * kps[i, 1] + alphaT * prev[i, 1] + betaS * neigh_avg[i, 1]

        # Update prev smoothed and history
        self._prev_smoothed[:, 0:2] = kps[:, 0:2]
        self._prev_smoothed[:, 2] = kps[:, 2]
        self._push_history(kps[:, 0:2])

        out = keypoints_with_scores.copy()
        out[0, 0, :, :] = kps
        return out

    def _push_history(self, yx):
        self._hist.append(yx.copy())
        if len(self._hist) > self._hist_len:
            self._hist.pop(0)
    
    def _apply_torso_constraints(self, yx, prev_yx, conf):
        """Validate and constrain torso keypoints to maintain realistic body proportions.
        Checks shoulder-hip relationships to prevent unrealistic configurations."""
        try:
            # Keypoint indices: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
            left_shoulder, right_shoulder = 5, 6
            left_hip, right_hip = 11, 12
            
            # Check if we have enough confident torso detections
            torso_indices = [left_shoulder, right_shoulder, left_hip, right_hip]
            confident_torso = sum([1 for i in torso_indices if conf[i] > 0.10])
            
            if confident_torso >= 3:  # Need at least 3 torso points
                # Calculate shoulder width and hip width
                shoulder_width = np.linalg.norm(yx[right_shoulder] - yx[left_shoulder])
                hip_width = np.linalg.norm(yx[right_hip] - yx[left_hip])
                
                # Calculate torso height (average shoulder to hip distance)
                left_torso_height = np.linalg.norm(yx[left_hip] - yx[left_shoulder])
                right_torso_height = np.linalg.norm(yx[right_hip] - yx[right_shoulder])
                avg_torso_height = (left_torso_height + right_torso_height) / 2.0
                
                # Realistic body proportion checks
                # 1. Shoulder width should be 20-50% of torso height
                # 2. Hip width should be 20-50% of torso height
                # 3. Shoulder and hip widths should be similar (within 2x of each other)
                
                if avg_torso_height > 0.01:  # Avoid division by zero
                    shoulder_ratio = shoulder_width / avg_torso_height
                    hip_ratio = hip_width / avg_torso_height
                    
                    # If proportions are unrealistic, use previous values
                    if shoulder_ratio < 0.15 or shoulder_ratio > 0.60:
                        # Shoulder width unrealistic, revert to previous
                        if conf[left_shoulder] < 0.25:
                            yx[left_shoulder] = prev_yx[left_shoulder]
                        if conf[right_shoulder] < 0.25:
                            yx[right_shoulder] = prev_yx[right_shoulder]
                    
                    if hip_ratio < 0.15 or hip_ratio > 0.60:
                        # Hip width unrealistic, revert to previous
                        if conf[left_hip] < 0.25:
                            yx[left_hip] = prev_yx[left_hip]
                        if conf[right_hip] < 0.25:
                            yx[right_hip] = prev_yx[right_hip]
                    
                    # Check if shoulder/hip width ratio is reasonable (0.5x to 2x)
                    if shoulder_width > 0.01 and hip_width > 0.01:
                        width_ratio = shoulder_width / hip_width
                        if width_ratio < 0.4 or width_ratio > 2.5:
                            # Use the more confident pair
                            shoulder_conf = (conf[left_shoulder] + conf[right_shoulder]) / 2.0
                            hip_conf = (conf[left_hip] + conf[right_hip]) / 2.0
                            
                            if shoulder_conf < hip_conf and shoulder_conf < 0.25:
                                yx[left_shoulder] = prev_yx[left_shoulder]
                                yx[right_shoulder] = prev_yx[right_shoulder]
                            elif hip_conf < shoulder_conf and hip_conf < 0.25:
                                yx[left_hip] = prev_yx[left_hip]
                                yx[right_hip] = prev_yx[right_hip]
            
            return yx
        except Exception:
            # If any error occurs, return unmodified
            return yx

    def _apply_outlier_guard(self, yx, prev_yx, keypoints_with_scores):
        """Strict outlier detection to prevent jitter from noisy MoveNet detections."""
        try:
            # Get confidence scores
            conf = keypoints_with_scores[0, 0, :, 2]
            
            # Normalized per-frame movement cap
            pixel_cap_norm = float(self._pixel_step_frac)

            # Strict MAD-based outlier detection with sufficient history
            if len(self._hist) >= 4:  # Need less history for faster response
                hist_stack = np.stack(self._hist, axis=0)  # (T,17,2)
                med = np.median(hist_stack, axis=0)
                mad = np.median(np.abs(hist_stack - med), axis=0) + 1e-6
                
                diff = yx - med
                
                # Apply strict MAD filtering
                for i in range(17):
                    deviation = np.abs(diff[i]) / mad[i]
                    
                    # If deviation is extreme and confidence is not very high, clamp it
                    if np.any(deviation > self._mad_k):
                        if conf[i] < 0.65:  # Lowered from 0.7 for more responsiveness
                            # Clamp to MAD boundary
                            yx[i] = med[i] + np.sign(diff[i]) * np.minimum(np.abs(diff[i]), self._mad_k * mad[i])

            # Per-frame step clamp - stricter for low confidence
            delta = yx - prev_yx
            for i in range(17):
                # Scale cap based on confidence
                if conf[i] > 0.65:
                    cap = pixel_cap_norm * 4.0  # Allow large movement for high confidence
                elif conf[i] > 0.45:
                    cap = pixel_cap_norm * 2.0  # Moderate movement for medium confidence
                else:
                    cap = pixel_cap_norm * 0.6  # Restrictive for low confidence (likely noise)
                
                delta[i] = np.clip(delta[i], -cap, cap)
            
            yx = prev_yx + delta
            return yx
        except Exception:
            return yx
    
    def process_frame(self, frame, show_feedback=True):
        """Process a single frame and return the result."""
        # Convert BGR to RGB for model input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Remember frame shape for outlier guard pixel clamp
        self.last_frame_shape = frame.shape
        
        # Resize and pad to model input size
        input_image = tf.expand_dims(image_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        
        # Run MoveNet
        keypoints_with_scores = self.movenet(input_image)

        # Apply smoothing (Kalman or EMA)
        if getattr(self, 'kalman_enabled', False):
            keypoints_with_scores = self.smooth_keypoints_kalman(keypoints_with_scores)
        else:
            keypoints_with_scores = self.smooth_keypoints_ema(keypoints_with_scores)

        # Get comprehensive feedback if requested
        feedback = None
        if show_feedback:
            feedback = self.feedback.get_comprehensive_feedback(
                keypoints_with_scores, frame.shape[0], frame.shape[1]
            )

        # Determine exercise mode for visualization routing
        exercise_mode = getattr(self.feedback, 'exercise_mode', 'squat')

        # Configure visualization per mode
        if exercise_mode == 'bench':
            # Bench press benefits from lower thresholds and relaxed bounds
            threshold = 0.20
            output_overlay = draw_prediction_on_image_bench_press(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold,
            )
        else:
            # Squat/default path: choose based on feedback richness
            if feedback and feedback.get('form_analysis'):
                threshold = 0.22  # Reduced from 0.25 for better torso tracking
                output_overlay = draw_prediction_on_image_enhanced(
                    frame.copy(),
                    keypoints_with_scores,
                    keypoint_threshold=threshold,
                )
            elif feedback and feedback['distance_status'] in ['very_close', 'close']:
                threshold = 0.22  # Reduced from 0.25 for better torso tracking
                output_overlay = draw_prediction_on_image_adaptive(
                    frame.copy(),
                    keypoints_with_scores,
                    keypoint_threshold=threshold,
                )
            else:
                threshold = 0.25  # Reduced from 0.30 for better torso tracking
                output_overlay = draw_prediction_on_image_simple(
                    frame.copy(),
                    keypoints_with_scores,
                    keypoint_threshold=threshold,
                )

        # Add comprehensive feedback overlay if requested
        if show_feedback and feedback:
            output_overlay = draw_comprehensive_feedback_overlay(output_overlay, feedback)

        return output_overlay, keypoints_with_scores, feedback

def process_video_with_squat_analysis(video_path, movenet_model, input_size, output_path=None):
    """Process video with comprehensive squat form analysis - no temporal interpolation."""
    processor = PoseProcessor(movenet_model, input_size)
    
    # Open video with FFMPEG backend and disable automatic rotation
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Try to disable automatic rotation (this may not work on all systems)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        print("Disabled automatic rotation")
    except:
        print("Could not disable automatic rotation (not supported on this system)")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame to check actual dimensions and detect any automatic rotation
    ret, test_frame = cap.read()
    if ret:
        actual_height, actual_width = test_frame.shape[:2]
        print(f"Video metadata dimensions: {width}x{height}")
        print(f"Actual frame dimensions: {actual_width}x{actual_height}")
        
        # Check if OpenCV automatically rotated the video
        if actual_width != width or actual_height != height:
            print("WARNING: Frame dimensions don't match metadata - OpenCV may have applied rotation!")
            print("This can happen with phone videos that have rotation metadata.")
            print("The video will be processed as-is to preserve the original orientation.")
            # Use actual frame dimensions instead of metadata
            width, height = actual_width, actual_height
            print(f"Using actual dimensions: {width}x{height}")
        else:
            print("Frame dimensions match metadata - no automatic rotation detected")
            
        # Special case: Detect if a portrait video was rotated to landscape by OpenCV
        # This happens when the video was recorded as 1080x1920 but OpenCV reads it as 1920x1080
        # We need to counter-rotate it back to portrait
        needs_counter_rotation = False
        if (width == 1920 and height == 1080 and 
            actual_width == 1920 and actual_height == 1080):
            # This looks like a portrait video that was rotated to landscape
            print("DETECTED: Portrait video (1080x1920) was rotated to landscape (1920x1080) by OpenCV")
            print("Will counter-rotate frames back to portrait orientation")
            needs_counter_rotation = True
            # Update dimensions to reflect the original portrait orientation
            width, height = 1080, 1920
        
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Keep the same dimensions as input video - no rotation
    # Process and output in the same orientation as input
    out_w, out_h = width, height  # Always keep original dimensions
    is_portrait = height > width  # Determine if portrait based on dimensions
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Video orientation: {'Portrait' if is_portrait else 'Landscape'}")
    print("Squat Form Analysis System (Mobile Optimized):")
    print("- Analyzes back rounding, knee alignment, depth, and arm position")
    print("- Provides real-time form score and recommendations")
    print("- Detects squat phases: standing, descending, bottom, ascending")
    print("- Enhanced keypoint visualization for better form analysis")
    print("- Compact feedback overlay for better visibility")
    print("- Preserves original video orientation")
    print("- Green: Good form | Orange: Needs improvement | Red: Form issues")
    
    # Determine processing cadence: process at input fps up to a maximum of 60 fps
    # If fps is unavailable, assume 30
    try:
        import math
        max_fps = 60
        input_fps = fps if fps and fps > 0 else 30
        process_step = max(1, int(math.ceil(input_fps / max_fps)))
        effective_output_fps = max(1, int(round(input_fps / process_step)))
    except Exception:
        process_step = 1
        effective_output_fps = fps if fps and fps > 0 else 30

    # Output video writing is now handled in main.py
    
    # This function should only process and return frames, not handle output
    # (Implementation of frame processing loop should be refactored to main.py)
    # For now, raise NotImplementedError to indicate this responsibility has moved
    raise NotImplementedError("Video output and display logic should be handled in main.py, not pose_processor.py.")

def process_webcam_with_squat_analysis(movenet_model, input_size):
    """Process webcam feed with comprehensive squat form analysis - no temporal interpolation."""
    processor = PoseProcessor(movenet_model, input_size)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set camera properties for mobile-friendly resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 720p width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720p height
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available

    print("MoveNet Lightning - Squat Form Analysis (Mobile Optimized)")
    print("Form Analysis Features:")
    print("- Back rounding detection")
    print("- Knee alignment and valgus detection")
    print("- Squat depth analysis")
    print("- Arm position feedback")
    print("- Real-time form scoring")
    print("- Enhanced keypoint visualization")
    print("- Compact feedback overlay")
    print("- 720p resolution for mobile performance")
    print("Press 'q' to quit.")

    # No buffering for temporal interpolation

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process and display each frame directly
            output_overlay, _, _ = processor.process_frame(frame, show_feedback=True)
            cv2.namedWindow('MoveNet Lightning - Squat Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('MoveNet Lightning - Squat Analysis', output_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Keep the old functions for backward compatibility
def process_video_with_improved_feedback(video_path, movenet_model, input_size, output_path=None):
    """Process video with improved feedback system (legacy function)."""
    return process_video_with_squat_analysis(video_path, movenet_model, input_size, output_path)

def process_webcam_with_improved_feedback(movenet_model, input_size):
    """Process webcam feed with improved feedback system (legacy function)."""
    return process_webcam_with_squat_analysis(movenet_model, input_size)