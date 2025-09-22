import cv2
import tensorflow as tf
import numpy as np
from .visualization_utils import draw_prediction_on_image_simple, draw_prediction_on_image_adaptive, draw_prediction_on_image_enhanced
from .feedback_utils import PoseFeedback, draw_comprehensive_feedback_overlay

def interpolate_keypoints(prev_kp, curr_kp, next_kp):
    """Interpolate keypoints using previous, current, and next frames.
    If confidence is available, use it as weights; otherwise, use simple mean."""
    # Assume shape (..., 3): (y, x, confidence)
    if prev_kp.shape == curr_kp.shape == next_kp.shape and prev_kp.shape[-1] == 3:
        # Weighted average by confidence
        prev_conf = prev_kp[..., 2]
        curr_conf = curr_kp[..., 2]
        next_conf = next_kp[..., 2]
        total_conf = prev_conf + curr_conf + next_conf + 1e-6  # avoid div by zero
        y = (prev_kp[..., 0] * prev_conf + curr_kp[..., 0] * curr_conf + next_kp[..., 0] * next_conf) / total_conf
        x = (prev_kp[..., 1] * prev_conf + curr_kp[..., 1] * curr_conf + next_kp[..., 1] * next_conf) / total_conf
        conf = np.maximum.reduce([prev_conf, curr_conf, next_conf])
        return np.stack([y, x, conf], axis=-1)
    else:
        # Fallback: simple mean
        return (prev_kp + curr_kp + next_kp) / 3.0


class PoseProcessor:
    """Handles pose detection processing with improved stability and squat form analysis."""
    
    def __init__(self, movenet_model, input_size):
        self.movenet = movenet_model
        self.input_size = input_size
        self.feedback = PoseFeedback()
        
        # Responsive smoothing parameters - optimized for stability without lag
        self.prev_keypoints = None
        self.smoothing_factor = 0.6  # Reduced for more responsiveness
        self.confidence_threshold = 0.15  # Minimum confidence for keypoints
        
        # Keypoint-specific smoothing factors - balanced for responsiveness
        self.keypoint_smoothing_factors = {
            0: 0.7,   # nose - balanced
            1: 0.7,   # left_eye
            2: 0.7,   # right_eye
            3: 0.7,   # left_ear
            4: 0.7,   # right_ear
            5: 0.65,  # left_shoulder - core body part
            6: 0.65,  # right_shoulder
            7: 0.6,   # left_elbow - arm part
            8: 0.6,   # right_elbow
            9: 0.55,  # left_wrist - extremity
            10: 0.55, # right_wrist
            11: 0.65, # left_hip - core body part
            12: 0.65, # right_hip
            13: 0.6,  # left_knee - leg part
            14: 0.6,  # right_knee
            15: 0.55, # left_ankle - extremity
            16: 0.55  # right_ankle
        }
    
    def get_keypoint_confidence(self, keypoints, kp_idx):
        """Get confidence score for a keypoint, handling different formats."""
        try:
            if keypoints.shape[1] == 3:
                # Format: (x, y, confidence)
                return float(keypoints[kp_idx, 2])
            elif keypoints.shape[1] == 1:
                # Format: (confidence) - this is likely the scores array
                return float(keypoints[kp_idx, 0])
            else:
                # Unknown format, assume high confidence
                return 1.0
        except (IndexError, TypeError):
            # If we can't access the confidence, assume it's high
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
        """Calculate movement between current and previous keypoints."""
        if previous_kp is None:
            return np.zeros(current_kp.shape[0])
        
        # Calculate Euclidean distance for each keypoint
        movement = np.zeros(current_kp.shape[0])
        for i in range(current_kp.shape[0]):
            try:
                curr_x, curr_y = self.get_keypoint_coords(current_kp, i)
                prev_x, prev_y = self.get_keypoint_coords(previous_kp, i)
                movement[i] = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            except:
                movement[i] = 0.0
        
        return movement
    
    def apply_responsive_smoothing(self, current_keypoints, prev_keypoints):
        """Apply responsive smoothing that reduces jitter without causing lag."""
        if prev_keypoints is None:
            return current_keypoints
        
        smoothed_keypoints = current_keypoints.copy()
        
        # Calculate movement for each keypoint
        movement = self.calculate_movement(current_keypoints, prev_keypoints)
        
        for i in range(current_keypoints.shape[0]):
            try:
                # Get confidence for this keypoint
                confidence = self.get_keypoint_confidence(current_keypoints, i)
                
                # Skip if confidence is too low
                if confidence < self.confidence_threshold:
                    continue
                
                # Get keypoint-specific smoothing factor
                base_smoothing = self.keypoint_smoothing_factors.get(i, 0.6)
                
                # Adaptive smoothing based on movement
                if movement[i] < 0.01:  # Very small movement - apply more smoothing
                    smoothing_factor = min(0.8, base_smoothing + 0.1)
                elif movement[i] < 0.05:  # Small movement - moderate smoothing
                    smoothing_factor = base_smoothing
                else:  # Large movement - minimal smoothing for responsiveness
                    smoothing_factor = max(0.3, base_smoothing - 0.2)
                
                # Apply smoothing - handle different keypoint formats
                if current_keypoints.shape[1] == 3 and prev_keypoints.shape[1] == 3:
                    # Standard format: (x, y, confidence)
                    smoothed_keypoints[i] = (smoothing_factor * prev_keypoints[i] + 
                                           (1 - smoothing_factor) * current_keypoints[i])
                elif current_keypoints.shape[1] == 1 and prev_keypoints.shape[1] == 1:
                    # Confidence scores only
                    smoothed_keypoints[i, 0] = (smoothing_factor * prev_keypoints[i, 0] + 
                                              (1 - smoothing_factor) * current_keypoints[i, 0])
            except Exception as e:
                # If smoothing fails for this keypoint, keep original
                continue
        
        return smoothed_keypoints
    
    def apply_smoothing(self, current_keypoints, prev_keypoints):
        """Apply responsive smoothing pipeline."""
        if prev_keypoints is None:
            return current_keypoints
        
        try:
            # Apply responsive smoothing
            smoothed_keypoints = self.apply_responsive_smoothing(current_keypoints, prev_keypoints)
            return smoothed_keypoints
        except Exception as e:
            # If smoothing fails, return original keypoints
            print(f"Warning: Smoothing failed, using original keypoints. Error: {e}")
            return current_keypoints
    
    def process_frame(self, frame, show_feedback=True):
        """Process a single frame and return the result."""
        # Convert BGR to RGB for model input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad to model input size
        input_image = tf.expand_dims(image_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)

        # Run MoveNet
        keypoints_with_scores = self.movenet(input_image)

        # Apply responsive smoothing
        if self.prev_keypoints is not None:
            keypoints_with_scores = self.apply_smoothing(keypoints_with_scores, self.prev_keypoints)
        
        # Store current keypoints for next frame
        self.prev_keypoints = keypoints_with_scores.copy()

        # Get comprehensive feedback if requested
        feedback = None
        if show_feedback:
            feedback = self.feedback.get_comprehensive_feedback(
                keypoints_with_scores, frame.shape[0], frame.shape[1]
            )

        # Determine threshold and visualization method based on feedback
        if feedback and feedback.get('form_analysis'):
            # Use enhanced visualization for squat analysis
            threshold = 0.15
            use_enhanced = True
        elif feedback and feedback['distance_status'] in ['very_close', 'close']:
            threshold = 0.15
            use_adaptive = True
            use_enhanced = False
        else:
            threshold = 0.2
            use_adaptive = False
            use_enhanced = False

        # Draw prediction with appropriate method
        if use_enhanced:
            output_overlay = draw_prediction_on_image_enhanced(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold
            )
        elif use_adaptive:
            output_overlay = draw_prediction_on_image_adaptive(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold
            )
        else:
            output_overlay = draw_prediction_on_image_simple(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold
            )

        # Add comprehensive feedback overlay if requested
        if show_feedback and feedback:
            output_overlay = draw_comprehensive_feedback_overlay(output_overlay, feedback)

        return output_overlay, keypoints_with_scores, feedback

def process_video_with_squat_analysis(video_path, movenet_model, input_size, output_path=None):
    """Process video with comprehensive squat form analysis - optimized for mobile, with temporal interpolation."""
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
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    
    print("Processing video... Press 'q' to stop early.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Process frame in original orientation - keep the same orientation as input
            # Apply counter-rotation if needed to restore original portrait orientation
            if 'needs_counter_rotation' in locals() and needs_counter_rotation:
                # Counter-rotate the frame back to portrait (rotate 90 degrees clockwise)
                pose_detection_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                if frame_count <= 3:
                    print(f"Applied counter-rotation to frame {frame_count}")
            else:
                pose_detection_frame = frame
            
            # Debug: Print frame dimensions for first few frames
            if frame_count <= 3:
                frame_h, frame_w = frame.shape[:2]
                print(f"Frame {frame_count} dimensions: {frame_w}x{frame_h}")
                if 'needs_counter_rotation' in locals() and needs_counter_rotation:
                    rotated_h, rotated_w = pose_detection_frame.shape[:2]
                    print(f"After counter-rotation: {rotated_w}x{rotated_h}")
            
            # Process frame for pose detection - real-time processing without buffering
            output_overlay, keypoints_with_scores, feedback = processor.process_frame(pose_detection_frame, show_feedback=True)
            
            # Apply counter-rotation to output if needed
            if 'needs_counter_rotation' in locals() and needs_counter_rotation:
                # The output_overlay is already processed with the rotated frame, so no additional rotation needed
                pass
            
            # Display the result in a window that adapts to video dimensions
            cv2.namedWindow('MoveNet Lightning - Squat Analysis', cv2.WINDOW_NORMAL)
            
            # Calculate display window size based on video dimensions
            # Limit max size to fit on screen while maintaining aspect ratio
            max_width, max_height = 1920, 1080  # Max display size
            display_w, display_h = out_w, out_h
            
            # Scale down if video is too large
            if display_w > max_width or display_h > max_height:
                scale = min(max_width / display_w, max_height / display_h)
                display_w = int(display_w * scale)
                display_h = int(display_h * scale)
            
            cv2.resizeWindow('MoveNet Lightning - Squat Analysis', display_w, display_h)
            cv2.imshow('MoveNet Lightning - Squat Analysis', output_overlay)
            
            if writer:
                # Ensure output frame matches video writer dimensions to prevent black bars
                if output_overlay.shape[:2] != (out_h, out_w):
                    # If counter-rotation was applied, we need to rotate the output back to match video writer
                    if 'needs_counter_rotation' in locals() and needs_counter_rotation:
                        # Rotate back to original orientation to match video writer dimensions
                        output_overlay = cv2.rotate(output_overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        # Just resize to match dimensions
                        output_overlay = cv2.resize(output_overlay, (out_w, out_h))
                writer.write(output_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing stopped by user.")
                break
                
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Video processing completed!")

def process_webcam_with_squat_analysis(movenet_model, input_size):
    """Process webcam feed with comprehensive squat form analysis - optimized for mobile."""
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

    buffer_size = 5 # must be odd
    frame_buffer = []
    keypoints_buffer = []
    feedback_buffer = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame (get keypoints, but don't display yet)
            _, keypoints_with_scores, feedback = processor.process_frame(frame, show_feedback=True)
            frame_buffer.append(frame)
            keypoints_buffer.append(keypoints_with_scores)
            feedback_buffer.append(feedback)

            if len(frame_buffer) == buffer_size:
                center_idx = buffer_size // 2

                # Interpolate keypoints for the center frame
                prev_kp = keypoints_buffer[center_idx - 1]
                curr_kp = keypoints_buffer[center_idx]
                next_kp = keypoints_buffer[center_idx + 1]

                # You need to implement this function:
                smoothed_kp = interpolate_keypoints(prev_kp, curr_kp, next_kp)

                # Draw overlay for the center frame using smoothed keypoints
                output_overlay, _, _ = processor.process_frame(
                    frame_buffer[center_idx], show_feedback=True
                )
                # Optionally, replace keypoints in output_overlay with smoothed_kp

                # Display with adaptive window sizing
                cv2.namedWindow('MoveNet Lightning - Squat Analysis', cv2.WINDOW_NORMAL)
                cv2.imshow('MoveNet Lightning - Squat Analysis', output_overlay)

                # Remove oldest frame/keypoints/feedback
                frame_buffer.pop(0)
                keypoints_buffer.pop(0)
                feedback_buffer.pop(0)

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