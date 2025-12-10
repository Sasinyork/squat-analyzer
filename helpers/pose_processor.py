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
        self.spatiotemporal_enabled = True  # Spatio temporal smoothing toggle
        self.kalman_enabled = False  # Kalman filter smoothing toggle
        self.occlusion_handling_enabled = False  # Toggle for occlusion handling

        # Pure spatio-temporal smoothing controls
        self.temporal_alpha = 0.35    # Increased weight to previous for more stability
        self.spatial_beta = 0.25      # Moderate spatial smoothing
        self.conf_threshold = 0.15    # min confidence to trust current point
        self.spatial_min_conf = 0.15  # min confidence to include neighbor
        self._prev_smoothed = None    # (17,3) last smoothed
        # Build adjacency list from skeleton edges
        self._neighbors = self._build_neighbors()
        # Outlier guard history (store last N smoothed frames for robust stats)
        self._hist_len = 8  # Longer history for smoother median filtering
        self._hist = []  # list of (17,2) arrays of smoothed yx
        self._mad_k = 3.5  # Tighter threshold for outlier rejection (smoother)
        self._pixel_step_frac = 0.08  # Reduced per-frame movement for smoother motion
        self.lead_gain = 0.0          # disable lookahead (causes overshoot with noisy data)
        self.last_frame_shape = None
        
        # Occlusion handling state
        self._occlusion_state = np.zeros(17, dtype=int)  # 0=visible, >0=frames occluded
        self._occlusion_threshold = 3  # Frames to confirm occlusion
        self._occlusion_conf_threshold = 0.20  # Confidence below this suggests occlusion
        self._occlusion_history_len = 5  # Frames to track for temporal consistency
        self._confidence_history = []  # Track confidence over time
        self._velocity_history = []  # Track velocity for consistency
        self._velocity_hist_len = 3  # Frames to track velocity
        self._max_velocity_jump = 0.15  # Max normalized velocity change per frame
        
        # Side mirroring for occlusion (for side-view videos)
        self._enable_side_mirroring = True  # Mirror visible side to occluded side
        self._side_mirror_conf_diff = 0.10  # Min confidence difference to trigger mirroring (lowered for more sensitivity)
        self._mirror_blend_alpha = 0.3  # Smooth transition for mirroring (lower = smoother)
        self._prev_mirrored = {}  # Store previous mirrored positions for smooth transition
        
        # One Euro Filter parameters for adaptive smoothing
        self._one_euro_enabled = True
        self._one_euro_filters = None  # Will store filter state per keypoint
        self._one_euro_min_cutoff = 0.5  # Lower = more smoothing
        self._one_euro_beta = 0.007  # Speed coefficient - lower = less responsive to speed
        self._one_euro_dcutoff = 1.0  # Derivative cutoff frequency

        # Kalman filter state for each keypoint (17 keypoints, 2D)
        self._kalman_filters = None  # Will be initialized on first use
        
    def _build_neighbors(self):
        """Build adjacency list from KEYPOINT_EDGES for spatial smoothing."""
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        neighbors = {}
        for i in range(17):
            neighbors[i] = []
        for a, b in edges:
            neighbors[a].append(b)
            neighbors[b].append(a)
        return neighbors

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
            # Only clamp wrist y below elbow for bench/deadlift, NOT for squat
            # In squat, arms are typically UP holding the barbell
            mode = getattr(self.feedback, 'exercise_mode', 'squat')
            if mode != 'squat':  # Changed condition - only apply for non-squat exercises
                left_elbow_y = kps[7, 0]
                right_elbow_y = kps[8, 0]
                if kps[9, 0] > left_elbow_y:
                    kps[9, 0] = left_elbow_y
                if kps[10, 0] > right_elbow_y:
                    kps[10, 0] = right_elbow_y
            
            # Disable wrist crossing prevention for squat (arms can move freely)
            if mode != 'squat':
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
                
                # Disable arm-leg collision detection for squat (arms are up, not near legs)
                if mode != 'squat':
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
                    
                # Old arm mirroring logic - skip for squat, let new occlusion handling take over
                if i in arm_indices and mode != 'squat':
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
    
    def _init_one_euro_filters(self):
        """Initialize One Euro Filter state for each keypoint (x and y separately)."""
        self._one_euro_filters = []
        for i in range(17):
            # Each keypoint gets x and y filters
            filter_state = {
                'x': {'prev_val': None, 'prev_dx': 0.0, 'prev_time': None},
                'y': {'prev_val': None, 'prev_dy': 0.0, 'prev_time': None},
            }
            self._one_euro_filters.append(filter_state)
    
    def _one_euro_filter_step(self, val, filter_state, timestamp):
        """Apply One Euro Filter for a single dimension (x or y).
        
        One Euro Filter combines exponential smoothing with speed adaptation:
        - Low speed -> more smoothing (reduces jitter)
        - High speed -> less smoothing (reduces lag)
        """
        if filter_state['prev_val'] is None:
            # First frame - initialize
            filter_state['prev_val'] = val
            filter_state['prev_time'] = timestamp
            # Initialize derivative based on which dimension we're tracking
            if 'prev_dx' not in filter_state and 'prev_dy' not in filter_state:
                filter_state['prev_d'] = 0.0
            return val
        
        # Calculate time delta
        dt = timestamp - filter_state['prev_time']
        if dt <= 0:
            dt = 1.0 / 30.0  # Assume 30fps if time doesn't advance
        
        # Estimate current velocity (derivative)
        if filter_state['prev_val'] is not None:
            d_val = (val - filter_state['prev_val']) / dt
        else:
            d_val = 0.0
        
        # Get previous derivative (check both possible keys)
        if 'prev_dx' in filter_state:
            prev_d = filter_state['prev_dx']
        elif 'prev_dy' in filter_state:
            prev_d = filter_state['prev_dy']
        else:
            prev_d = filter_state.get('prev_d', 0.0)
        
        # Smooth the derivative
        alpha_d = self._smoothing_factor(dt, self._one_euro_dcutoff)
        d_smooth = alpha_d * d_val + (1.0 - alpha_d) * prev_d
        
        # Adaptive cutoff based on velocity
        cutoff = self._one_euro_min_cutoff + self._one_euro_beta * abs(d_smooth)
        
        # Smooth the value
        alpha = self._smoothing_factor(dt, cutoff)
        val_smooth = alpha * val + (1.0 - alpha) * filter_state['prev_val']
        
        # Update state (save to the correct key)
        filter_state['prev_val'] = val_smooth
        if 'prev_dx' in filter_state:
            filter_state['prev_dx'] = d_smooth
        elif 'prev_dy' in filter_state:
            filter_state['prev_dy'] = d_smooth
        else:
            filter_state['prev_d'] = d_smooth
        filter_state['prev_time'] = timestamp
        
        return val_smooth
    
    def _smoothing_factor(self, dt, cutoff):
        """Calculate exponential smoothing factor for given cutoff frequency."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        alpha = dt / (dt + tau)
        return alpha
    
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
    
    # ========== OCCLUSION HANDLING METHODS ==========
    
    def _update_occlusion_state(self, conf):
        """Track which keypoints are likely occluded over time.
        
        Args:
            conf: (17,) array of confidence scores
            
        Updates self._occlusion_state with frame counts since occlusion started
        """
        for i in range(17):
            if conf[i] < self._occlusion_conf_threshold:
                # Low confidence - increment occlusion counter
                self._occlusion_state[i] += 1
            else:
                # Good confidence - reset counter
                self._occlusion_state[i] = 0
    
    def _is_occluded(self, kp_idx):
        """Check if a keypoint is currently considered occluded.
        
        Args:
            kp_idx: Keypoint index
            
        Returns:
            bool: True if keypoint is occluded
        """
        return self._occlusion_state[kp_idx] >= self._occlusion_threshold
    
    def _update_confidence_history(self, conf):
        """Track confidence over time for temporal consistency analysis.
        
        Args:
            conf: (17,) array of confidence scores
        """
        self._confidence_history.append(conf.copy())
        if len(self._confidence_history) > self._occlusion_history_len:
            self._confidence_history.pop(0)
    
    def _check_temporal_confidence_consistency(self, kp_idx):
        """Check if a keypoint has consistently low confidence (likely occluded).
        
        Args:
            kp_idx: Keypoint index
            
        Returns:
            bool: True if confidence is consistently low
        """
        if len(self._confidence_history) < 3:
            return False
        
        recent_conf = [hist[kp_idx] for hist in self._confidence_history[-3:]]
        return np.mean(recent_conf) < self._occlusion_conf_threshold
    
    def _mirror_visible_side_to_occluded(self, yx, conf):
        """Mirror the visible side of the body to the occluded side.
        
        For side-view videos, this copies the movement pattern from the visible side
        to the occluded side, maintaining the relative position structure.
        
        Args:
            yx: (17,2) array of current keypoint positions
            conf: (17,) array of confidence scores
            
        Returns:
            (17,2) array of corrected keypoint positions
        """
        if not self._enable_side_mirroring:
            return yx
        
        yx_corrected = yx.copy()
        
        # Define left-right keypoint pairs with their parent keypoints for proper mirroring
        # Format: (left_idx, right_idx, left_parent, right_parent)
        keypoint_pairs = [
            # Arms - mirror relative to respective shoulders
            (7, 8, 5, 6),   # left_elbow, right_elbow, anchored to shoulders
            (9, 10, 7, 8),  # left_wrist, right_wrist, anchored to elbows
            
            # Legs - mirror relative to respective hips  
            (13, 14, 11, 12), # left_knee, right_knee, anchored to hips
            (15, 16, 13, 14), # left_ankle, right_ankle, anchored to knees
        ]
        
        # Process each pair
        for left_idx, right_idx, left_parent, right_parent in keypoint_pairs:
            left_conf = conf[left_idx]
            right_conf = conf[right_idx]
            
            # Determine which side is more visible
            conf_diff = abs(left_conf - right_conf)
            
            # Only mirror if confidence difference is significant OR one side is very low
            should_mirror = (conf_diff > self._side_mirror_conf_diff) or \
                          (left_conf < self._occlusion_conf_threshold) or \
                          (right_conf < self._occlusion_conf_threshold)
            
            if not should_mirror:
                continue
            
            # Determine source and target
            if left_conf > right_conf:
                # Left is visible, mirror to right
                source_idx = left_idx
                target_idx = right_idx
                source_parent = left_parent
                target_parent = right_parent
            else:
                # Right is visible, mirror to left
                source_idx = right_idx
                target_idx = left_idx
                source_parent = right_parent
                target_parent = left_parent
            
            # Check if parents are visible
            if conf[source_parent] > self._occlusion_conf_threshold and \
               conf[target_parent] > self._occlusion_conf_threshold:
                
                # Calculate the offset from source parent to source keypoint
                offset_y = yx[source_idx, 0] - yx[source_parent, 0]  # vertical offset
                offset_x = yx[source_idx, 1] - yx[source_parent, 1]  # horizontal offset
                
                # Apply SAME offset to target (not mirrored) - this maintains movement pattern
                # For side view, both sides move similarly
                new_pos = np.array([
                    yx[target_parent, 0] + offset_y,  # Same Y offset
                    yx[target_parent, 1] + offset_x   # Same X offset (NOT mirrored)
                ])
                
                # Smooth the transition to avoid jitter
                if target_idx in self._prev_mirrored:
                    # Blend with previous mirrored position
                    alpha = self._mirror_blend_alpha
                    new_pos = alpha * new_pos + (1 - alpha) * self._prev_mirrored[target_idx]
                
                # Store for next frame
                self._prev_mirrored[target_idx] = new_pos.copy()
                
                # Apply the mirrored position
                yx_corrected[target_idx] = new_pos
        
        return yx_corrected

    
    def _estimate_occluded_keypoint_kinematic(self, kp_idx, yx, conf):
        """Use biomechanical constraints to estimate occluded keypoint position.
        
        Uses body structure and limb length constraints to predict position
        of occluded keypoints based on connected visible keypoints.
        
        Args:
            kp_idx: Index of occluded keypoint
            yx: (17,2) array of current keypoint positions
            conf: (17,) array of confidence scores
            
        Returns:
            (y, x) tuple of estimated position, or None if can't estimate
        """
        # Define kinematic chains (parent -> child relationships)
        kinematic_chains = {
            # Head/neck
            1: [0],  # left_eye <- nose
            2: [0],  # right_eye <- nose
            3: [1],  # left_ear <- left_eye
            4: [2],  # right_ear <- right_eye
            
            # Arms
            7: [5],  # left_elbow <- left_shoulder
            9: [7],  # left_wrist <- left_elbow
            8: [6],  # right_elbow <- right_shoulder
            10: [8],  # right_wrist <- right_elbow
            
            # Torso and legs
            11: [5],  # left_hip <- left_shoulder
            12: [6],  # right_hip <- right_shoulder
            13: [11],  # left_knee <- left_hip
            15: [13],  # left_ankle <- left_knee
            14: [12],  # right_knee <- right_hip
            16: [14],  # right_ankle <- right_knee
        }
        
        # Define typical limb length ratios (relative to torso)
        limb_length_ratios = {
            # Arms (relative to shoulder position)
            7: 0.3,  # shoulder to elbow
            9: 0.3,  # elbow to wrist
            8: 0.3,
            10: 0.3,
            
            # Legs (relative to hip position)
            13: 0.35,  # hip to knee
            15: 0.35,  # knee to ankle
            14: 0.35,
            16: 0.35,
        }
        
        if kp_idx not in kinematic_chains:
            return None
        
        # Find a confident parent keypoint
        parents = kinematic_chains[kp_idx]
        for parent_idx in parents:
            if conf[parent_idx] > self.spatial_min_conf:
                parent_pos = yx[parent_idx]
                
                # If we have history, use previous displacement
                if self._prev_smoothed is not None:
                    prev_offset = self._prev_smoothed[kp_idx, :2] - self._prev_smoothed[parent_idx, :2]
                    estimated_pos = parent_pos + prev_offset
                    return estimated_pos
                
                # Otherwise use typical limb length
                elif kp_idx in limb_length_ratios:
                    # Use previous direction if available
                    if self._prev_smoothed is not None:
                        direction = self._prev_smoothed[kp_idx, :2] - self._prev_smoothed[parent_idx, :2]
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 1e-6:
                            direction = direction / direction_norm
                            estimated_pos = parent_pos + direction * limb_length_ratios[kp_idx]
                            return estimated_pos
        
        return None
    
    def _apply_spatial_interpolation_for_occlusion(self, kp_idx, yx, conf):
        """Enhanced spatial smoothing specifically for occluded keypoints.
        
        More aggressively uses neighbor positions to interpolate occluded points.
        
        Args:
            kp_idx: Index of keypoint to interpolate
            yx: (17,2) array of current keypoint positions
            conf: (17,) array of confidence scores
            
        Returns:
            (y, x) tuple of interpolated position, or None if can't interpolate
        """
        neighbors = self._neighbors.get(kp_idx, [])
        if not neighbors:
            return None
        
        # Collect confident neighbor positions
        neighbor_positions = []
        neighbor_weights = []
        
        for neighbor_idx in neighbors:
            if conf[neighbor_idx] > self.spatial_min_conf and not self._is_occluded(neighbor_idx):
                neighbor_positions.append(yx[neighbor_idx])
                # Weight by confidence
                neighbor_weights.append(conf[neighbor_idx])
        
        if len(neighbor_positions) == 0:
            return None
        
        # Weighted average of neighbor positions
        neighbor_positions = np.array(neighbor_positions)
        neighbor_weights = np.array(neighbor_weights)
        neighbor_weights = neighbor_weights / np.sum(neighbor_weights)  # Normalize
        
        interpolated_pos = np.sum(neighbor_positions * neighbor_weights[:, np.newaxis], axis=0)
        return interpolated_pos
    
    def _check_velocity_consistency(self, yx, prev_yx):
        """Check for unrealistic velocity jumps and flag inconsistent keypoints.
        
        Args:
            yx: (17,2) array of current keypoint positions
            prev_yx: (17,2) array of previous keypoint positions
            
        Returns:
            (17,) boolean array where True indicates unrealistic velocity
        """
        if prev_yx is None:
            return np.zeros(17, dtype=bool)
        
        # Calculate current velocity
        velocity = yx - prev_yx
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # Update velocity history
        self._velocity_history.append(velocity.copy())
        if len(self._velocity_history) > self._velocity_hist_len:
            self._velocity_history.pop(0)
        
        # Check for unrealistic jumps
        inconsistent = np.zeros(17, dtype=bool)
        
        if len(self._velocity_history) >= 2:
            # Compare current velocity to recent average
            recent_velocities = np.array(self._velocity_history[:-1])  # Exclude current
            avg_velocity = np.mean(recent_velocities, axis=0)
            avg_velocity_magnitude = np.linalg.norm(avg_velocity, axis=1)
            
            # Flag if velocity jump is too large
            for i in range(17):
                velocity_change = abs(velocity_magnitude[i] - avg_velocity_magnitude[i])
                if velocity_change > self._max_velocity_jump:
                    inconsistent[i] = True
        
        return inconsistent
    
    def _handle_occluded_keypoints(self, yx, conf, prev_yx):
        """Main occlusion handling logic - combines all occlusion strategies.
        
        Strategy:
        1. First, apply side mirroring (mirror visible side to occluded side)
        2. Then apply kinematic/spatial interpolation for any remaining issues
        
        Args:
            yx: (17,2) array of current keypoint positions
            conf: (17,) array of confidence scores
            prev_yx: (17,2) array of previous keypoint positions
            
        Returns:
            (17,2) array of corrected keypoint positions
        """
        if not self.occlusion_handling_enabled:
            return yx
        
        # Update occlusion tracking
        self._update_occlusion_state(conf)
        self._update_confidence_history(conf)
        
        # STEP 1: Apply side mirroring first (for side-view videos)
        # This mirrors the visible side to the occluded side
        yx_corrected = self._mirror_visible_side_to_occluded(yx, conf)
        
        # Check velocity consistency (using mirrored positions)
        velocity_inconsistent = self._check_velocity_consistency(yx_corrected, prev_yx)
        
        # STEP 2: Apply kinematic/spatial correction for remaining issues
        for i in range(17):
            should_correct = False
            correction_method = None
            
            # Determine if keypoint needs additional correction
            if self._is_occluded(i):
                should_correct = True
                correction_method = "occlusion"
            elif velocity_inconsistent[i] and conf[i] < 0.5:
                should_correct = True
                correction_method = "velocity"
            elif conf[i] < self._occlusion_conf_threshold and self._check_temporal_confidence_consistency(i):
                should_correct = True
                correction_method = "low_confidence"
            
            if should_correct:
                # Try kinematic estimation first (more accurate)
                estimated_pos = self._estimate_occluded_keypoint_kinematic(i, yx_corrected, conf)
                
                # Fall back to spatial interpolation
                if estimated_pos is None:
                    estimated_pos = self._apply_spatial_interpolation_for_occlusion(i, yx_corrected, conf)
                
                # Fall back to previous position if all else fails
                if estimated_pos is None and prev_yx is not None:
                    estimated_pos = prev_yx[i]
                
                # Apply correction with blending based on confidence
                if estimated_pos is not None:
                    # Blend between current and estimated based on confidence
                    blend_factor = max(0.0, min(1.0, (self._occlusion_conf_threshold - conf[i]) / self._occlusion_conf_threshold))
                    blend_factor = max(0.7, blend_factor)  # At least 70% correction
                    
                    yx_corrected[i] = (1.0 - blend_factor) * yx_corrected[i] + blend_factor * estimated_pos
        
        return yx_corrected
    
    # ========== END OCCLUSION HANDLING METHODS ==========
    
    def smooth_keypoints_ema(self, keypoints_with_scores):
        """Enhanced spatio-temporal smoothing with One Euro Filter.
        
        Combines:
        1. One Euro Filter for adaptive temporal smoothing (reduces jitter while maintaining responsiveness)
        2. Spatial smoothing using neighbor averaging
        3. Confidence-weighted blending
        4. Outlier detection and removal
        
        Expects shape (1,1,17,3) with (y,x,score).
        """
        if not self.spatiotemporal_enabled or keypoints_with_scores is None:
            return keypoints_with_scores

        kps = keypoints_with_scores[0, 0, :, :].copy()  # (17,3)
        
        # Initialize One Euro Filter on first use
        if self._one_euro_enabled and self._one_euro_filters is None:
            self._init_one_euro_filters()
        
        if self._prev_smoothed is None:
            # First frame: initialize
            self._prev_smoothed = kps.copy()
            if self._one_euro_enabled:
                # Initialize filter states with first frame values
                for i in range(17):
                    self._one_euro_filters[i]['x']['prev_val'] = kps[i, 0]
                    self._one_euro_filters[i]['y']['prev_val'] = kps[i, 1]
                    self._one_euro_filters[i]['x']['prev_time'] = 0.0
                    self._one_euro_filters[i]['y']['prev_time'] = 0.0
            return keypoints_with_scores

        conf = kps[:, 2].copy()
        prev = self._prev_smoothed.copy()
        
        # Generate synthetic timestamp (could be replaced with actual timestamps if available)
        import time
        current_time = time.time()
        
        # Apply One Euro Filter for temporal smoothing (if enabled)
        if self._one_euro_enabled:
            for i in range(17):
                # Only apply filter if confidence is reasonable
                if conf[i] > self.conf_threshold:
                    kps[i, 0] = self._one_euro_filter_step(
                        kps[i, 0], 
                        self._one_euro_filters[i]['x'], 
                        current_time
                    )
                    kps[i, 1] = self._one_euro_filter_step(
                        kps[i, 1], 
                        self._one_euro_filters[i]['y'], 
                        current_time
                    )
                else:
                    # Low confidence - use previous smoothed value
                    kps[i, 0] = prev[i, 0]
                    kps[i, 1] = prev[i, 1]
        else:
            # Fallback to basic EMA if One Euro Filter is disabled
            alphaT = float(self.temporal_alpha)
            betaS = float(self.spatial_beta)
            gammaC = max(0.0, 1.0 - alphaT - betaS)  # weight for current

            # Compute neighbor averages for spatial term
            neigh_avg = np.zeros((17, 2), dtype=np.float32)
            for i in range(17):
                neigh = self._neighbors.get(i, [])
                pts = []
                for j in neigh:
                    if conf[j] >= self.spatial_min_conf:  # Only use confident neighbors
                        pts.append([kps[j, 0], kps[j, 1]])
                if pts:
                    pts_arr = np.array(pts, dtype=np.float32)
                    neigh_avg[i, 0] = float(np.mean(pts_arr[:, 0]))
                    neigh_avg[i, 1] = float(np.mean(pts_arr[:, 1]))
                else:
                    neigh_avg[i, 0] = prev[i, 0]
                    neigh_avg[i, 1] = prev[i, 1]

            # Confidence-weighted blending: blend current, previous, and neighbor average
            for i in range(17):
                if conf[i] >= self.conf_threshold:
                    # Use confidence to adjust blending weights
                    conf_weight = min(conf[i], 0.8)  # Cap confidence influence
                    temp_weight = alphaT * (1.0 + (1.0 - conf_weight) * 0.5)  # More previous if low conf
                    spat_weight = betaS
                    curr_weight = max(0.1, 1.0 - temp_weight - spat_weight)  # Ensure some current weight
                    
                    kps[i, 0] = curr_weight * kps[i, 0] + temp_weight * prev[i, 0] + spat_weight * neigh_avg[i, 0]
                    kps[i, 1] = curr_weight * kps[i, 1] + temp_weight * prev[i, 1] + spat_weight * neigh_avg[i, 1]
                else:
                    # Very low confidence - heavily favor previous position
                    kps[i, 0] = 0.1 * kps[i, 0] + 0.9 * prev[i, 0]
                    kps[i, 1] = 0.1 * kps[i, 1] + 0.9 * prev[i, 1]

        # Apply occlusion handling (corrects occluded keypoints using kinematic constraints)
        kps_yx = kps[:, 0:2].copy()
        prev_yx = prev[:, 0:2].copy()
        kps_yx = self._handle_occluded_keypoints(kps_yx, conf, prev_yx)
        kps[:, 0:2] = kps_yx
        
        # Apply outlier guard with updated keypoints
        kps_yx = kps[:, 0:2].copy()
        prev_yx = prev[:, 0:2].copy()
        kps_yx = self._apply_outlier_guard(kps_yx, prev_yx, keypoints_with_scores)
        
        # Apply torso constraints for realistic body proportions
        kps_yx = self._apply_torso_constraints(kps_yx, prev_yx, conf)
        
        kps[:, 0:2] = kps_yx

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
        """Improved outlier detection with smoother transitions.
        
        Uses MAD (Median Absolute Deviation) for robust outlier detection,
        but applies gradual clamping instead of hard rejection for smoother motion.
        """
        try:
            # Get confidence scores
            conf = keypoints_with_scores[0, 0, :, 2]
            
            # Normalized per-frame movement cap
            pixel_cap_norm = float(self._pixel_step_frac)

            # Smoother MAD-based outlier detection with sufficient history
            if len(self._hist) >= 5:  # Reduced from 4 for more stable median
                hist_stack = np.stack(self._hist, axis=0)  # (T,17,2)
                med = np.median(hist_stack, axis=0)
                mad = np.median(np.abs(hist_stack - med), axis=0) + 1e-6
                
                diff = yx - med
                
                # Apply gradual MAD filtering for smoother motion
                for i in range(17):
                    deviation = np.abs(diff[i]) / mad[i]
                    
                    # Use confidence-adaptive thresholding
                    conf_factor = 1.0 + (1.0 - conf[i]) * 2.0  # Higher threshold for low conf
                    adaptive_mad_k = self._mad_k * conf_factor
                    
                    # Gradual clamping instead of hard rejection
                    if np.any(deviation > adaptive_mad_k):
                        if conf[i] < 0.70:
                            # Soft clamp: gradually pull back outliers
                            max_allowed_dev = adaptive_mad_k * mad[i]
                            for dim in range(2):
                                if abs(diff[i, dim]) > max_allowed_dev[dim]:
                                    # Use exponential decay for smooth transition
                                    excess = abs(diff[i, dim]) - max_allowed_dev[dim]
                                    reduction = excess * 0.7  # Reduce 70% of excess
                                    yx[i, dim] = med[i, dim] + np.sign(diff[i, dim]) * (max_allowed_dev[dim] + excess - reduction)

            # Confidence-adaptive per-frame step clamp with smooth scaling
            delta = yx - prev_yx
            for i in range(17):
                # Smooth confidence-based scaling curve
                if conf[i] > 0.70:
                    cap = pixel_cap_norm * 5.0  # Allow larger movement for very high confidence
                elif conf[i] > 0.50:
                    # Smooth interpolation between medium and high
                    interp = (conf[i] - 0.50) / 0.20
                    cap = pixel_cap_norm * (2.5 + interp * 2.5)
                elif conf[i] > 0.30:
                    # Smooth interpolation between low and medium
                    interp = (conf[i] - 0.30) / 0.20
                    cap = pixel_cap_norm * (1.0 + interp * 1.5)
                else:
                    cap = pixel_cap_norm * 0.8  # Restrictive for very low confidence
                
                # Apply smooth clamping
                for dim in range(2):
                    if abs(delta[i, dim]) > cap:
                        # Soft clamp with smooth transition
                        delta[i, dim] = np.sign(delta[i, dim]) * (cap + (abs(delta[i, dim]) - cap) * 0.2)
            
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