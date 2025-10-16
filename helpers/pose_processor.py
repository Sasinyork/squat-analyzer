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
    """Handles pose detection processing with improved stability and squat form analysis."""
    
    def __init__(self, movenet_model, input_size):
        self.movenet = movenet_model
        self.input_size = input_size
        self.feedback = PoseFeedback()
        
        # Pure spatio-temporal smoothing controls
        self.spatiotemporal_enabled = False
        self.temporal_alpha = 0.12    # weight toward previous (balanced for responsiveness)
        self.spatial_beta = 0.03      # minimal spatial (reduce cross-keypoint noise)
        self.conf_threshold = 0.12    # min confidence to trust current point
        self.spatial_min_conf = 0.15  # min confidence to include neighbor
        self._prev_smoothed = None    # (17,3) last smoothed
        # Build adjacency list from skeleton edges
        self._neighbors = self._build_neighbors()
        # Outlier guard history (store last N smoothed frames for robust stats)
        self._hist_len = 5  # Moderate history for noise detection without too much lag
        self._hist = []  # list of (17,2) arrays of smoothed yx
        self._mad_k = 3.5  # threshold multiplier (strict but not too strict)
        self._pixel_step_frac = 0.10  # max per-frame step for low-confidence points
        self.lead_gain = 0.0          # disable lookahead (causes overshoot with noisy data)
        self.last_frame_shape = None
        
        # Movement consistency tracking
        self._velocity_history = []  # Track velocity for consistency
        self._velocity_hist_len = 3

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
        conf_thr = float(self.conf_threshold)
        min_neigh_conf = float(self.spatial_min_conf)

        # Compute neighbor averages for spatial term
        neigh_avg = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            neigh = self._neighbors.get(i, [])
            pts = []
            for j in neigh:
                if conf[j] >= min_neigh_conf:
                    pts.append([kps[j, 0], kps[j, 1]])
            if pts:
                pts_arr = np.array(pts, dtype=np.float32)
                neigh_avg[i, 0] = float(np.mean(pts_arr[:, 0]))
                neigh_avg[i, 1] = float(np.mean(pts_arr[:, 1]))
            else:
                # Fallback to previous smoothed as neighbor proxy if none confident
                neigh_avg[i, 0] = prev[i, 0]
                neigh_avg[i, 1] = prev[i, 1]

        # Blend positions with confidence-aware weighting
        for i in range(17):
            conf_normalized = min(max(conf[i], 0.0), 1.0)
            
            # Calculate velocity (change from previous)
            if len(self._hist) >= 1:
                velocity = np.linalg.norm(kps[i, 0:2] - prev[i, 0:2])
            else:
                velocity = 0.0
            
            # Detect if this is likely noise vs real movement
            # Very permissive - only flag obvious noise
            is_likely_noise = (velocity > 0.05 and conf[i] < 0.25) or (velocity > 0.15 and conf[i] < 0.45)
            
            # REMOVED: Special torso handling that was causing lag
            # All keypoints now use the same smoothing logic
            
            # Standard handling for all keypoints
            if is_likely_noise:
                # Noise detected: use heavy temporal smoothing
                g = gammaC * 0.1
            elif conf[i] >= conf_thr:
                # Real movement: scale by confidence (very aggressive)
                if conf[i] > 0.4:
                    conf_scale = 6.0 + (conf_normalized - 0.4) / 0.6 * 14.0  # 6-20x for high confidence
                else:
                    conf_scale = 1.0 + (conf_normalized - conf_thr) / (0.4 - conf_thr) * 5.0
                g = gammaC * conf_scale
            else:
                g = 0.0
            
            total_weight = g + alphaT + betaS
            if total_weight > 0:
                g_norm = g / total_weight
                alpha_norm = alphaT / total_weight
                beta_norm = betaS / total_weight
            else:
                g_norm = 0.0
                alpha_norm = 1.0
                beta_norm = 0.0
            
            kps[i, 0] = g_norm * kps[i, 0] + alpha_norm * prev[i, 0] + beta_norm * neigh_avg[i, 0]
            kps[i, 1] = g_norm * kps[i, 1] + alpha_norm * prev[i, 1] + beta_norm * neigh_avg[i, 1]

        # Velocity lookahead disabled to prevent overshoot with noisy detections
        # (lead_gain = 0.0)
        
        # Robust outlier guard using short history and per-frame pixel clamp
        kps[:, 0:2] = self._apply_outlier_guard(kps[:, 0:2], prev[:, 0:2], keypoints_with_scores)

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

        # Apply global EMA smoothing to keypoints
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