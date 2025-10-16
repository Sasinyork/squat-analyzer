import cv2
import numpy as np
import math

class BenchPressFormAnalyzer:
    """Analyzes bench press form and provides keypoint detection for side view."""
    
    def __init__(self):
        self.prev_keypoints = None
        self.bench_phase = "stable"  # stable, lowering, pressing
        self.phase_frames = 0
        # Pending phase (debounce) state
        self.pending_phase = None
        self.pending_frames = 0
        self.phase_change_min_frames = 5  # default debounce frames for any phase change
        # Optional per-phase minima; can be tuned per orientation
        self.phase_min_frames = {
            'rest': 8,
            'lowering': 5,
            'bottom': 7,
            'pressing': 5,
            'pause': 8
        }
        
        # Phase detection parameters
        self.wrist_positions = []  # Store recent wrist positions for movement detection
        self.max_history = 10  # Number of frames to track for movement
        self.movement_threshold = 0.6  # Minimum pixel movement to detect direction
        self.rest_threshold = 50  # Wrist must be this much above chest to be "rest"
        self.bottom_threshold = 40  # Wrist must be close to chest level to be "bottom"
        self.lowering_threshold = 30  # Minimum distance from rest position to detect lowering
        self.last_rest_position = None  # Store the wrist position when last at rest
        
    # Video orientation settings
        self.video_orientation = "landscape"  # Default orientation
        self.needs_counter_rotation = False
        
        # Keypoint smoothing parameters
        self.keypoint_history = []  # Store recent keypoints for smoothing
        self.smoothing_window = 5  # Number of frames to average
        self.min_frames_for_smoothing = 3  # Minimum frames needed before smoothing
        
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
        self.depth_stable_frames = 0     # Count frames with negligible depth change
        self.stable_frames_threshold = 30  # ~1 second at 30 FPS
        self.depth_epsilon_ratio = 0.003   # ~0.3% of image height, minimum a couple pixels
        self.depth_direction_debounce = 2  # frames required to confirm direction change
        self._direction_pending = None
        self._direction_pending_frames = 0
    
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
            self.phase_min_frames.update({'rest': 9, 'bottom': 8, 'pause': 9, 'pressing': 6, 'lowering': 6})
        else:
            # Landscape (original) thresholds
            self.movement_threshold = 0.6
            self.rest_threshold = 50
            self.bottom_threshold = 40
            self.lowering_threshold = 30
            self.phase_change_min_frames = 6
            self.phase_min_frames.update({'rest': 8, 'bottom': 7, 'pause': 8, 'pressing': 5, 'lowering': 5})
        
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
            return "lowering"  # Wrist Y increasing means moving down (lowering)
        elif avg_movement < -movement_threshold:
            return "pressing"  # Wrist Y decreasing means moving up (pressing)
        else:
            return "stable"    # Minimal movement
    
    def detect_bench_phase(self, keypoints, image_height, image_width):
        """Detect bench phase using simple depth trend rules and expose relative depth.

        Rules:
        - If relative depth is decreasing -> phase = "lowering"
        - If relative depth is increasing -> phase = "pressing"
        - If depth change stays within epsilon for ~1 second -> phase = "stable" and reset baseline
        Returns current phase and the relative depth (baseline-adjusted).
        """
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        # Require wrists and shoulders to compute depth reliably
        required = [left_wrist, right_wrist, left_shoulder, right_shoulder]
        if not all(required):
            return self.bench_phase, 0.0

        # Average vertical positions (pixels)
        wrist_y = (left_wrist[1] + right_wrist[1]) / 2.0
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

        # Raw depth defined as vertical separation (wrist relative to shoulder)
        raw_depth = wrist_y - shoulder_y

        # Initialize baseline on first valid frame
        if self.baseline_depth is None:
            self.baseline_depth = raw_depth

        # Relative depth so that baseline is 0
        rel_depth = self.baseline_depth - raw_depth

        # Compute delta of relative depth to determine trend
        eps = max(2.0, self.depth_epsilon_ratio * float(image_height))
        if self.prev_rel_depth is not None:
            delta = rel_depth - self.prev_rel_depth
        else:
            delta = 0.0

        # Update stable counter
        if abs(delta) <= eps:
            self.depth_stable_frames += 1
        else:
            self.depth_stable_frames = 0

        # Determine instantaneous direction
        if delta > eps:
            direction = "lowering"   # relative depth increasing
        elif delta < -eps:
            direction = "pressing"   # relative depth decreasing
        else:
            direction = "stable"

        # Confirm direction with a tiny debounce to avoid flicker
        new_phase = self.bench_phase
        # Immediate stable if depth itself is effectively zero
        if abs(rel_depth) <= eps:
            new_phase = "stable"
            # Anchor baseline so the displayed depth remains at 0
            self.baseline_depth = raw_depth
            rel_depth = 0.0
            # Clear direction pending
            self._direction_pending = None
            self._direction_pending_frames = 0
            self.depth_stable_frames = max(self.depth_stable_frames, self.stable_frames_threshold)
        else:
            if direction in ("pressing", "lowering"):
                if self._direction_pending == direction:
                    self._direction_pending_frames += 1
                else:
                    self._direction_pending = direction
                    self._direction_pending_frames = 1
                if self._direction_pending_frames >= self.depth_direction_debounce:
                    new_phase = direction
            else:
                # Only call it truly stable if held for ~1 second
                if self.depth_stable_frames >= self.stable_frames_threshold:
                    new_phase = "stable"
                    # Reset depth baseline on stable to make depth start from 0 again
                    self.baseline_depth = raw_depth
                    rel_depth = 0.0
                    # Clear pending direction
                    self._direction_pending = None
                    self._direction_pending_frames = 0

        # Update phase frame counters
        if new_phase == self.bench_phase:
            self.phase_frames += 1
        else:
            self.bench_phase = new_phase
            self.phase_frames = 1

        # Persist for next frame
        self.prev_rel_depth = rel_depth

        return self.bench_phase, float(rel_depth)
    
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
        
        # Return analysis with smoothed and rotated keypoints
        analysis = {
            'phase': phase,
            'phase_frames': self.phase_frames,
            'depth_metric': depth_metric,
            'keypoints': final_keypoints
        }
        
        return analysis
