import cv2
import numpy as np
import math

class BenchPressFormAnalyzer:
    """Analyzes bench press form and provides keypoint detection for side view."""
    
    def __init__(self):
        self.prev_keypoints = None
        self.bench_phase = "stable"  # stable, descending, bottom, ascending
        self.phase_frames = 0
        # Pending phase (debounce) state
        self.pending_phase = None
        self.pending_frames = 0
        self.phase_change_min_frames = 5  # default debounce frames for any phase change
        # Optional per-phase minima; can be tuned per orientation
        self.phase_min_frames = {
            'stable': 8,
            'descending': 5,
            'bottom': 7,
            'ascending': 5
        }
        
        # Phase detection parameters
        self.max_history = 10  # Number of frames to track for movement
        self.movement_threshold = 0.6  # Minimum pixel movement to detect direction (legacy, not used in new FSM)
        self.rest_threshold = 50  # Wrist must be this much above chest to be "rest" (legacy)
        self.bottom_threshold = 40  # Wrist must be close to chest level to be "bottom" (legacy)
        self.lowering_threshold = 30  # Minimum distance from rest position to detect lowering (legacy)
        self.last_rest_position = None  # Store the wrist position when last at rest (legacy)
        
    # Video orientation settings
        self.video_orientation = "landscape"  # Default orientation
        self.needs_counter_rotation = False
        
        # Keypoint smoothing parameters
        self.keypoint_history = []  # Store recent keypoints for smoothing
        self.smoothing_window = 5  # Number of frames to average
        self.min_frames_for_smoothing = 3  # Minimum frames needed before smoothing
        
        # Stable position detection (similar to squat/deadlift standing detection)
        self.stable_frames_required = 10  # Require sustained stable position before allowing transitions (reduced from 15)
        self.stable_frame_counter = 0
        self.has_reached_stable = False  # Track if we've reached a proper stable position
        
        # Minimum frames per phase to prevent rapid flickering
        self.min_phase_frames = {
            'stable': 12,      # Require at least 12 frames in stable before leaving
            'descending': 8,   # Require at least 8 frames descending
            'bottom': 5,       # Reduced to 5 frames - allows brief touch-and-go reps
            'ascending': 10    # Require at least 10 frames ascending
        }
        
        # Keypoint indices for MoveNet
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16

        # Depth trend tracking (for simple phase logic)
        self.baseline_depth = None       # Depth baseline to compute relative depth
        self.prev_rel_depth = None       # Previous relative depth (for trend)
        self.wrist_positions = []        # Track wrist Y positions for movement detection
    
    def set_video_orientation(self, orientation, counter_rotation=False):
        """Set video orientation for proper analysis."""
        self.video_orientation = orientation
        self.needs_counter_rotation = counter_rotation
        
        # Adjust movement thresholds based on orientation
        if orientation == "portrait":
            # Portrait videos may need different thresholds
            self.movement_threshold = 0.4  # More sensitive for portrait
            self.rest_threshold = 40
            self.bottom_threshold = 30
            self.lowering_threshold = 20
            # Slightly increase debounce to reduce flicker on tall aspect
            self.phase_change_min_frames = 7
            self.phase_min_frames.update({'stable': 9, 'bottom': 8, 'ascending': 6, 'descending': 6})
        else:
            # Landscape (original) thresholds
            self.movement_threshold = 0.6
            self.rest_threshold = 50
            self.bottom_threshold = 40
            self.lowering_threshold = 30
            self.phase_change_min_frames = 6
            self.phase_min_frames.update({'stable': 8, 'bottom': 7, 'ascending': 5, 'descending': 5})
        
        print(f"Bench analyzer configured for {orientation} orientation, counter_rotation: {counter_rotation}")
        
    def get_keypoint_coords(self, keypoints, index, image_height, image_width):
        """Get pixel coordinates for a keypoint with adaptive threshold for bench press."""
        # For bench press, we need extremely low thresholds for lower body
        # since the pose is unusual (lying down) and may have lower confidence scores
        
        # Get the confidence score for this keypoint
        confidence = keypoints[index, 2]
        
        # Define base thresholds for different body parts
        if index in [15, 16]:  # Left and right ankle
            threshold = 0.02  # Extremely low threshold for ankles
        elif index in [13, 14]:  # Left and right knee
            threshold = 0.03  # Very low threshold for knees
        elif index in [11, 12]:  # Left and right hip
            threshold = 0.04  # Low threshold for hips
        else:
            threshold = 0.15  # Standard threshold for upper body keypoints
        
        # If we detect a clear upper body (indicating person is definitely in frame),
        # we can be even more lenient with lower body detection
        upper_body_visible = any(keypoints[[5,6], 2] > 0.3)  # Check if shoulders are visible
        if upper_body_visible and index >= 11:  # Lower body keypoints
            threshold *= 0.5  # Reduce threshold by half for lower body when upper body is clear
            
        if confidence > threshold:
            x = int(keypoints[index, 1] * image_width)
            y = int(keypoints[index, 0] * image_height)
            return (x, y)
        return None
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)."""
        if point1 is None or point2 is None or point3 is None:
            return None
            
        # Calculate vectors
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def detect_movement_direction(self, current_wrist_y):
        """Detect if the person is moving up or down based on wrist position history."""
        if len(self.wrist_positions) < 4:
            return "unknown"
        
        # Calculate recent movement trend using the last few frames
        recent_positions = self.wrist_positions[-5:]  # Use last 5 frames for stability
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, len(recent_positions)):
            movement = recent_positions[i] - recent_positions[i-1]
            movements.append(movement)
        
        # Use the average of recent movements
        avg_movement = sum(movements) / len(movements)
        
        # Raise the threshold slightly to avoid flicker
        movement_threshold = self.movement_threshold  # use configured threshold directly
        
        # Determine direction based on movement - FIXED direction logic
        if avg_movement > movement_threshold:
            return "descending"  # Wrist Y increasing means moving down (descending)
        elif avg_movement < -movement_threshold:
            return "ascending"  # Wrist Y decreasing means moving up (ascending)
        else:
            return "stable"    # Minimal movement
    
    def detect_movement_direction_wrist(self, current_wrist_y):
        """Detect movement direction based on wrist Y position (NO ROTATION - lying down).
        
        In lying-down position:
        - Y increasing (wrist moving DOWN in image) = descending toward chest
        - Y decreasing (wrist moving UP in image) = ascending, pressing up
        
        Args:
            current_wrist_y: Current Y coordinate of wrist (lying down, no rotation)
        
        Returns:
            tuple: (direction_str, movement_delta) where direction is "ascending", "descending", or "stable"
        """
        debug = True  # Set to True to see wrist movement
        
        self.wrist_positions.append(current_wrist_y)
        
        if len(self.wrist_positions) < 5:
            return "stable", 0.0
        
        # Average recent positions
        recent_avg = sum(self.wrist_positions[-5:]) / 5
        older_avg = sum(self.wrist_positions[-10:-5]) / 5 if len(self.wrist_positions) >= 10 else recent_avg
        
        # Movement delta (positive = Y increasing = descending)
        delta = recent_avg - older_avg
        
        # Threshold: Movement must be significant to change direction detection
        descending_threshold = 9.0  # Higher threshold = must be clearly descending (increased from 7.0)
        ascending_threshold = 6.0   # Higher threshold = must be clearly ascending (increased from 4.0)
        stable_threshold = 3.0      # Movement must be within ±3.0 pixels to be considered "stable"
        
        if debug:
            print(f"  wrist_y: {current_wrist_y:.1f} | delta: {delta:.2f} | desc_thresh: {descending_threshold} | asc_thresh: {ascending_threshold}")
        
        # Only return "stable" when movement is truly minimal (within ±1.0 pixels)
        # This prevents premature bottom phase detection when bar is just slowing down
        if abs(delta) < stable_threshold:
            return "stable", delta  # Movement within ±1.0 pixels = truly at bottom
        elif delta > descending_threshold:
            return "descending", delta  # Y increasing by 7+ = moving down toward chest
        elif delta < -ascending_threshold:
            return "ascending", delta   # Y decreasing = moving up, pressing
        elif delta > 0:
            # Positive but less than threshold = still descending, just slower
            return "descending", delta
        else:
            # Negative but less than threshold = still ascending, just slower
            return "ascending", delta

    def is_at_stable_position(self, keypoints, image_height, image_width):
        """Check if person is in stable lockout position (similar to is_standing in squat/deadlift).
        
        In lying-down bench press, stable position means:
        1. Wrists are above shoulders (arms extended)
        2. All keypoints are aligned (not twisted or setting up)
        3. Minimal movement (not actively pressing)
        
        This prevents premature phase detection during setup.
        """
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        left_elbow = self.get_keypoint_coords(keypoints, self.LEFT_ELBOW, image_height, image_width)
        right_elbow = self.get_keypoint_coords(keypoints, self.RIGHT_ELBOW, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        
        if not all([left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder]):
            return False
        
        # Average positions
        wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        elbow_y = (left_elbow[1] + right_elbow[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Check 1: Wrists should be above shoulders (lockout position) - LESS RESTRICTIVE
        wrist_shoulder_diff = shoulder_y - wrist_y  # Positive = wrists above shoulders
        if wrist_shoulder_diff <= 10:  # Wrists must be 10px+ above shoulders (reduced from 20)
            return False
        
        # Check 2: Elbow should be between wrist and shoulder (proper alignment)
        # In lockout: wrist < elbow < shoulder (in Y coordinates, lower = higher in image)
        # Allow some tolerance for elbow position
        if not (wrist_y - 30 < elbow_y < shoulder_y + 30):  # Added 30px tolerance on both sides
            return False
        
        # Check 3: Arms should be relatively straight (not bent)
        # Elbow should be close to midpoint between wrist and shoulder
        expected_elbow_y = (wrist_y + shoulder_y) / 2
        elbow_deviation = abs(elbow_y - expected_elbow_y)
        if elbow_deviation > 60:  # Allow more bend (increased from 40)
            return False
        
        return True
    
    def is_at_lockout(self, keypoints, image_height, image_width):
        """Quick lockout check for phase transitions (less strict than is_at_stable_position)."""
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        
        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False
        
        # Average positions
        wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # At lockout (lying down): wrists should be ABOVE (lower Y) shoulders
        wrist_shoulder_diff = shoulder_y - wrist_y  # Positive = wrists above shoulders
        
        # Debug output
        if False:  # Set to True to enable debug
            is_lockout = wrist_shoulder_diff > 10
            print(f"[Lockout Check] wrist_y: {wrist_y:.1f}, shoulder_y: {shoulder_y:.1f}, diff: {wrist_shoulder_diff:.1f}, lockout: {is_lockout}")
        
        # Lockout: wrists must be clearly above shoulders (10px+, less strict)
        return wrist_shoulder_diff > 10

    def detect_bench_phase(self, keypoints, image_height, image_width):
        """Detect bench phase using FSM logic - NO ROTATION, direct lying-down analysis.
        
        In lying-down position:
        - stable: arms extended, wrists up (low Y)
        - descending: wrists moving down toward chest (Y increasing)
        - bottom: pause at chest
        - ascending: wrists moving up, pressing (Y decreasing)
        """
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        
        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return self.bench_phase, 0.0
        
        # Track wrist position (NO ROTATION)
        wrist_y = (left_wrist[1] + right_wrist[1]) / 2.0
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        
        # Depth: distance from shoulders (positive = wrists below shoulders = at chest)
        depth_from_chest = wrist_y - shoulder_y
        
        # Initialize baseline if needed
        if self.baseline_depth is None:
            self.baseline_depth = depth_from_chest
        
        # Get movement direction (based on wrist Y movement)
        movement_direction, movement_delta = self.detect_movement_direction_wrist(wrist_y)
        
        # Check if at proper stable position (robust check like squat/deadlift is_standing)
        is_stable_position = self.is_at_stable_position(keypoints, image_height, image_width)
        
        # Track stable position frames (similar to squat/deadlift)
        if is_stable_position:
            self.stable_frame_counter += 1
            if self.stable_frame_counter >= self.stable_frames_required:
                self.has_reached_stable = True
        else:
            self.stable_frame_counter = 0
        
        # Check if at lockout position (quick check for transitions)
        is_lockout = self.is_at_lockout(keypoints, image_height, image_width)
        
        # Don't allow phase transitions until we've reached stable position (prevents setup triggering)
        if not self.has_reached_stable:
            # Stay in stable phase during setup
            if self.bench_phase != "stable":
                self.bench_phase = "stable"
                self.phase_frames = 1
            else:
                self.phase_frames += 1
            return self.bench_phase, float(depth_from_chest)
        
        # FSM logic (similar to squat/deadlift)
        prev_phase = self.bench_phase
        new_phase = prev_phase
        
        # Check if we've been in current phase long enough to transition
        min_frames_for_current = self.min_phase_frames.get(prev_phase, 5)
        can_transition = self.phase_frames >= min_frames_for_current
        
        # Stable/lockout position detection
        if is_lockout and can_transition:
            new_phase = "stable"
            # Reset baseline at lockout
            self.baseline_depth = depth_from_chest
        elif prev_phase == "stable":
            # Can only transition to descending from stable (after min frames)
            if movement_direction == "descending" and can_transition:
                new_phase = "descending"
        elif prev_phase == "descending":
            # From descending, transition to bottom when:
            # 1. Movement stops/pauses (stable), OR
            # 2. Movement reverses to ascending (even without pause)
            # Don't use depth threshold - only use movement direction
            
            if movement_direction == "stable" and can_transition:
                # Movement stabilized/paused - at bottom (wherever that is)
                new_phase = "bottom"
            elif movement_direction == "ascending" and can_transition:
                # Started moving up - must go through bottom first
                new_phase = "bottom"
            else:
                # Continue descending
                new_phase = "descending"
        elif prev_phase == "bottom":
            # From bottom, can only go to ascending (after min frames)
            if movement_direction == "ascending" and can_transition:
                new_phase = "ascending"
            else:
                # Stay in bottom
                new_phase = "bottom"
        elif prev_phase == "ascending":
            # From ascending, can return to stable or continue ascending (after min frames)
            if is_lockout and can_transition:
                # Clear lockout position detected
                new_phase = "stable"
            elif movement_direction == "stable" and can_transition:
                # Movement stopped after ascending - assume at top/lockout
                new_phase = "stable"
            elif movement_direction == "descending" and movement_delta > 7.0 and can_transition:
                # Started another rep with clear downward movement
                new_phase = "descending"
            else:
                # Continue ascending or minor movements
                new_phase = "ascending"
        
        # Update phase counters
        if new_phase == self.bench_phase:
            self.phase_frames += 1
        else:
            print(f"Phase: {self.bench_phase} -> {new_phase} | depth: {depth_from_chest:.1f}px | movement: {movement_direction} ({movement_delta:.2f})")
            self.bench_phase = new_phase
            self.phase_frames = 1
        
        return self.bench_phase, float(depth_from_chest)
    
    def estimate_lower_body_keypoints(self, keypoints, image_height, image_width):
        """Estimate lower body keypoint positions based on bench position."""
        # Get hip position (or estimate from shoulders if not detected)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        
        if left_shoulder and right_shoulder:
            # Calculate center point between shoulders
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
            
            # Estimate hip position (slightly below shoulders for lying position)
            hip_y = shoulder_center_y + 20  # Slightly lower than shoulders
            
            # Update hip keypoints with estimated positions
            keypoints[self.LEFT_HIP, 1] = hip_y / image_width
            keypoints[self.RIGHT_HIP, 1] = hip_y / image_width
            keypoints[self.LEFT_HIP, 0] = (shoulder_center_x - 30) / image_height  # Slightly to the left
            keypoints[self.RIGHT_HIP, 0] = (shoulder_center_x + 30) / image_height  # Slightly to the right
            keypoints[self.LEFT_HIP, 2] = 0.5  # Set confidence
            keypoints[self.RIGHT_HIP, 2] = 0.5
            
            # Estimate knee positions (extend from hips)
            knee_y = hip_y + 150  # Further down
            keypoints[self.LEFT_KNEE, 1] = knee_y / image_width
            keypoints[self.RIGHT_KNEE, 1] = knee_y / image_width
            keypoints[self.LEFT_KNEE, 0] = (shoulder_center_x - 30) / image_height
            keypoints[self.RIGHT_KNEE, 0] = (shoulder_center_x + 30) / image_height
            keypoints[self.LEFT_KNEE, 2] = 0.5
            keypoints[self.RIGHT_KNEE, 2] = 0.5
            
            # Estimate ankle positions (extend from knees)
            ankle_y = knee_y + 150  # Further down
            keypoints[self.LEFT_ANKLE, 1] = ankle_y / image_width
            keypoints[self.RIGHT_ANKLE, 1] = ankle_y / image_width
            keypoints[self.LEFT_ANKLE, 0] = (shoulder_center_x - 30) / image_height
            keypoints[self.RIGHT_ANKLE, 0] = (shoulder_center_x + 30) / image_height
            keypoints[self.LEFT_ANKLE, 2] = 0.5
            keypoints[self.RIGHT_ANKLE, 2] = 0.5
            
        return keypoints

    def smooth_keypoints(self, keypoints):
        """Apply moving average smoothing to keypoints."""
        # Add current keypoints to history
        self.keypoint_history.append(keypoints.copy())
        
        # Keep only recent frames within smoothing window
        if len(self.keypoint_history) > self.smoothing_window:
            self.keypoint_history.pop(0)
        
        # If we don't have enough history yet, return current keypoints
        if len(self.keypoint_history) < self.min_frames_for_smoothing:
            return keypoints
        
        # Calculate smoothed keypoints using weighted moving average
        smoothed = np.zeros_like(keypoints)
        weights = np.linspace(0.5, 1.0, len(self.keypoint_history))  # More weight to recent frames
        weights = weights / np.sum(weights)  # Normalize weights
        
        for i, hist_keypoints in enumerate(self.keypoint_history):
            # Only smooth keypoints that are consistently detected
            mask = hist_keypoints[:, 2] > 0.1  # Check confidence scores
            smoothed[mask] += hist_keypoints[mask] * weights[i]
        
        # For keypoints that weren't consistently detected, use current frame
        mask = keypoints[:, 2] > 0.1
        smoothed[~mask] = keypoints[~mask]
        
        return smoothed

    def rotate_keypoints(self, keypoints, image_height, image_width, direction='ccw'):
        """Rotate keypoints 90 degrees to handle lying down position."""
        rotated = keypoints.copy()
        
        if direction == 'ccw':  # Counter-clockwise rotation
            for i in range(keypoints.shape[0]):
                if keypoints[i, 2] > 0:  # If keypoint is detected
                    # Swap x and y, and flip y
                    old_x = keypoints[i, 1]
                    old_y = keypoints[i, 0]
                    rotated[i, 1] = old_y  # New x = old y
                    rotated[i, 0] = 1.0 - old_x  # New y = 1 - old x
        else:  # Clockwise rotation
            for i in range(keypoints.shape[0]):
                if keypoints[i, 2] > 0:  # If keypoint is detected
                    # Swap x and y, and flip x
                    old_x = keypoints[i, 1]
                    old_y = keypoints[i, 0]
                    rotated[i, 1] = 1.0 - old_y  # New x = 1 - old y
                    rotated[i, 0] = old_x  # New y = old x
                    
        return rotated

    def analyze_arm_elbow_mechanics(self, keypoints, image_height, image_width, phase):
        """Analyze arm and elbow mechanics for bench press.
        
        Checks:
        - Elbow flare angle (too far out = shoulder stress)
        - Elbow tuck (proper 45-75° angle from torso)
        - Forearm verticality (perpendicular to ground at bottom)
        - Elbow lockout (full extension at top)
        """
        issues = []
        
        # Get keypoints
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_elbow = self.get_keypoint_coords(keypoints, self.LEFT_ELBOW, image_height, image_width)
        right_elbow = self.get_keypoint_coords(keypoints, self.RIGHT_ELBOW, image_height, image_width)
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        
        if not all([left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
            return issues
        
        # Calculate average positions for both sides
        avg_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        avg_elbow = ((left_elbow[0] + right_elbow[0]) / 2, (left_elbow[1] + right_elbow[1]) / 2)
        avg_wrist = ((left_wrist[0] + right_wrist[0]) / 2, (left_wrist[1] + right_wrist[1]) / 2)
        
        # Use left side for elbow flare angle (side with higher confidence)
        left_confidence = (keypoints[self.LEFT_SHOULDER, 2] + keypoints[self.LEFT_ELBOW, 2] + keypoints[self.LEFT_WRIST, 2]) / 3
        right_confidence = (keypoints[self.RIGHT_SHOULDER, 2] + keypoints[self.RIGHT_ELBOW, 2] + keypoints[self.RIGHT_WRIST, 2]) / 3
        
        if left_confidence >= right_confidence:
            shoulder_pt = left_shoulder
            elbow_pt = left_elbow
            wrist_pt = left_wrist
            hip_pt = left_hip if left_hip else None
        else:
            shoulder_pt = right_shoulder
            elbow_pt = right_elbow
            wrist_pt = right_wrist
            hip_pt = right_hip if right_hip else None
        
        # 1. Elbow flare angle during descending/bottom phase
        # Calculate angle between shoulder-elbow line and torso line (shoulder-hip)
        if phase in ['descending', 'bottom'] and hip_pt:
            # Vector from shoulder to elbow
            se_vec = np.array([elbow_pt[0] - shoulder_pt[0], elbow_pt[1] - shoulder_pt[1]])
            # Vector from shoulder to hip (torso)
            sh_vec = np.array([hip_pt[0] - shoulder_pt[0], hip_pt[1] - shoulder_pt[1]])
            
            # Calculate angle between vectors
            cos_angle = np.dot(se_vec, sh_vec) / (np.linalg.norm(se_vec) * np.linalg.norm(sh_vec) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            elbow_flare_angle = np.degrees(np.arccos(cos_angle))
            
            # Ideal range: 45-75° from torso
            if elbow_flare_angle > 85:  # Too much flare (approaching 90°)
                issues.append({
                    'type': 'elbow_flare',
                    'severity': 'high',
                    'message': 'Elbows flaring too wide',
                    'recommendation': 'Tuck elbows to 45-75deg angle',
                    'angle': elbow_flare_angle
                })
            elif elbow_flare_angle < 35:  # Too tucked
                issues.append({
                    'type': 'elbow_tuck_excessive',
                    'severity': 'medium',
                    'message': 'Elbows too tucked',
                    'recommendation': 'Allow elbows to be at 45-75deg angle',
                    'angle': elbow_flare_angle
                })
            elif 75 < elbow_flare_angle <= 85:  # Moderate flare
                issues.append({
                    'type': 'elbow_flare_moderate',
                    'severity': 'medium',
                    'message': 'Elbows slightly wide',
                    'recommendation': 'Tuck elbows slightly more',
                    'angle': elbow_flare_angle
                })
        
        # 2. Forearm verticality at bottom position
        if phase == 'bottom':
            # Forearm should be perpendicular to ground (vertical)
            # Calculate angle of forearm from vertical
            forearm_vec = np.array([wrist_pt[0] - elbow_pt[0], wrist_pt[1] - elbow_pt[1]])
            vertical_vec = np.array([0, 1])  # Downward vertical
            
            cos_angle = np.dot(forearm_vec, vertical_vec) / (np.linalg.norm(forearm_vec) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            forearm_angle_from_vertical = np.degrees(np.arccos(cos_angle))
            
            # Allow some tolerance (15° from vertical)
            if forearm_angle_from_vertical > 25:  # Forearm not vertical enough
                issues.append({
                    'type': 'forearm_not_vertical',
                    'severity': 'medium',
                    'message': 'Forearms not vertical at bottom',
                    'recommendation': 'Keep forearms perpendicular to ground',
                    'angle': forearm_angle_from_vertical
                })
        
        # 3. Elbow lockout at top (stable phase)
        if phase == 'stable':
            # Calculate elbow angle (shoulder-elbow-wrist)
            elbow_angle = self.calculate_angle(shoulder_pt, elbow_pt, wrist_pt)
            
            if elbow_angle is not None:
                # Full lockout: elbow angle should be close to 180° (straight arm)
                if elbow_angle < 160:  # Soft elbows (not fully locked out)
                    issues.append({
                        'type': 'soft_elbow_lockout',
                        'severity': 'medium',
                        'message': 'Elbows not fully locked out',
                        'recommendation': 'Fully extend arms at top',
                        'angle': elbow_angle
                    })
                elif elbow_angle < 170:  # Slight bend
                    issues.append({
                        'type': 'slight_elbow_bend',
                        'severity': 'low',
                        'message': 'Slight elbow bend at top',
                        'recommendation': 'Complete full lockout',
                        'angle': elbow_angle
                    })
        
        return issues

    def analyze_bench_press_form(self, keypoints_with_scores, image_height, image_width):
        """Bench press form analysis with rotation handling and smoothing for lying position."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # First apply smoothing to reduce jitter
        smoothed_keypoints = self.smooth_keypoints(keypoints)
        
        # Rotate smoothed keypoints 90 degrees counter-clockwise (to make the lying person "stand up")
        rotated_keypoints = self.rotate_keypoints(smoothed_keypoints, image_height, image_width, 'ccw')
        
        # Detect bench press phase using the rotated keypoints
        phase, depth_metric = self.detect_bench_phase(rotated_keypoints, image_width, image_height)  # Note: swapped dimensions
        
        # Rotate keypoints back 90 degrees clockwise
        final_keypoints = self.rotate_keypoints(rotated_keypoints, image_width, image_height, 'cw')
        
        # Analyze form issues using the original (non-rotated) keypoints
        form_issues = []
        
        # Analyze arm and elbow mechanics
        arm_issues = self.analyze_arm_elbow_mechanics(keypoints, image_height, image_width, phase)
        form_issues.extend(arm_issues)
        
        # Return analysis with smoothed and rotated keypoints plus form issues
        analysis = {
            'phase': phase,
            'phase_frames': self.phase_frames,
            'depth_metric': depth_metric,
            'keypoints': final_keypoints,
            'issues': form_issues
        }
        
        return analysis
