#!/usr/bin/env python3
"""
Main.py with proper orientation handling and angle display.

OCCLUSION HANDLING USAGE:
-------------------------
The PoseProcessor now includes comprehensive occlusion handling with side mirroring
for side-view videos. The system can be toggled on/off:

# Enable occlusion handling with side mirroring (default: True)
processor.occlusion_handling_enabled = True
processor._enable_side_mirroring = True  # Mirror visible side to occluded side

# Disable occlusion handling
processor.occlusion_handling_enabled = False

# Disable just side mirroring (keep other occlusion handling)
processor._enable_side_mirroring = False

Occlusion handling includes:
- SIDE MIRRORING: Mirrors the visible side's movements to the occluded side
  (Perfect for side-view videos where half the body is not visible)
- Temporal confidence tracking (detects consistently low confidence keypoints)
- Kinematic estimation (uses body structure to predict occluded points)
- Spatial interpolation (uses neighboring keypoints)
- Velocity consistency checks (detects unrealistic jumps)

When enabled, it works seamlessly with existing spatio-temporal smoothing.
"""


import cv2
import sys
from helpers.utils.model_utils import load_movenet_model
from helpers.pose_processor import PoseProcessor
from helpers.analyzers.deadlift_analyzer import DeadliftFormAnalyzer

def analyze_video_with_orientation(video_path="data/bench/bench.mp4", exercise_mode="bench", max_frames=None, output_path=None):
    """Analyze exercise video with real-time feedback display and proper orientation handling."""
    
    print(f"Loading MoveNet model...")
    movenet, input_size = load_movenet_model("movenet_lightning")
    

    # Create processor and set exercise mode
    processor = PoseProcessor(movenet, input_size)
    processor.feedback.set_exercise_mode(exercise_mode)
    
    # OCCLUSION HANDLING TOGGLE (uncomment to disable)
    # processor.occlusion_handling_enabled = False
    # processor._enable_side_mirroring = False  # Disable only side mirroring
    print(f"Occlusion handling: {'ENABLED' if processor.occlusion_handling_enabled else 'DISABLED'}")
    print(f"Side mirroring: {'ENABLED' if processor._enable_side_mirroring else 'DISABLED'}")


    # Attach deadlift analyzer if needed
    if exercise_mode == "deadlift":
        if not hasattr(processor.feedback, "deadlift_analyzer") or processor.feedback.deadlift_analyzer is None:
            processor.feedback.deadlift_analyzer = DeadliftFormAnalyzer()
    
    # Open video with proper orientation handling
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Try to disable automatic rotation
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        print("✓ Disabled automatic rotation")
    except Exception as e:
        print(f"✗ Could not disable automatic rotation: {e}")
    
    # Get video properties and detect orientation
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    metadata_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try to get rotation metadata
    rotation_metadata = 0
    try:
        rotation_metadata = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        print(f"Rotation metadata: {rotation_metadata} degrees")
    except Exception as e:
        print(f"Could not read rotation metadata: {e}")
    
    # Read first frame to check orientation
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    actual_height, actual_width = test_frame.shape[:2]
    print(f"Video metadata: {metadata_width}x{metadata_height}")
    print(f"Actual frame: {actual_width}x{actual_height}")
    
    # Detect if this is a portrait video that was rotated by OpenCV
    needs_counter_rotation = False
    original_is_portrait = False
    rotation_angle = 0
    
    # Check rotation metadata (90 or 270 degrees means portrait video stored as landscape)
    if rotation_metadata in [90, 270]:
        print(f"DETECTED: Video has {rotation_metadata}° rotation metadata")
        needs_counter_rotation = True
        original_is_portrait = True
        rotation_angle = rotation_metadata
        # Swap dimensions for portrait
        display_width, display_height = actual_height, actual_width
    # Also check if metadata indicates portrait but actual frame is landscape (OpenCV rotated it)
    elif metadata_height > metadata_width and actual_width > actual_height:
        print("DETECTED: Portrait video rotated to landscape by OpenCV (dimension mismatch)")
        needs_counter_rotation = True
        original_is_portrait = True
        rotation_angle = 90  # Assume 90 degree rotation
        display_width, display_height = metadata_width, metadata_height
    else:
        display_width, display_height = actual_width, actual_height
        original_is_portrait = actual_height > actual_width
    
    # Store original dimensions for output video (preserve quality)
    output_width, output_height = display_width, display_height
    
    print(f"Video orientation: {'Portrait' if original_is_portrait else 'Landscape'}")
    print(f"Original dimensions: {output_width}x{output_height}")
    print(f"Counter-rotation needed: {needs_counter_rotation}")
    
    print("DEBUG: About to configure bench/deadlift analyzer...")
    print(f"DEBUG: exercise_mode = {exercise_mode}")
    print(f"DEBUG: original_is_portrait = {original_is_portrait}")
    if exercise_mode == "bench":
        orientation = "portrait" if original_is_portrait else "landscape"
        print(f"Configuring bench analyzer for {orientation} mode...")
        try:
            processor.feedback.bench_analyzer.set_video_orientation(orientation, needs_counter_rotation)
            print(f"Successfully configured orientation: {orientation}, counter_rotation: {needs_counter_rotation}")
        except Exception as e:
            print(f"Error setting orientation: {e}")
            import traceback
            traceback.print_exc()
    elif exercise_mode == "deadlift":
        print("Deadlift mode: no orientation-specific configuration needed.")
    else:
        print(f"DEBUG: Skipping bench/deadlift analyzer config because exercise_mode is not 'bench' or 'deadlift'")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Set up output video writer if output_path is provided
    # Use original dimensions to preserve quality
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        print(f"Output video will be saved to: {output_path} at {output_width}x{output_height}")
    
    print(f"Processing video in {exercise_mode.upper()} mode...")
    print("Press 'q' to quit, 's' for squat mode, 'b' for bench mode, 'd' for deadlift mode, 'p' to pause")
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                frame_count += 1
            
            # Process frame with feedback (shows angles and phase info)
            if not paused:
                # Apply counter-rotation if needed to restore original orientation
                if needs_counter_rotation:
                    # Counter-rotate based on detected rotation angle
                    if rotation_angle == 90:
                        # Video rotated 90° CCW, so rotate 90° CW to restore
                        pose_detection_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation_angle == 270:
                        # Video rotated 90° CW, so rotate 90° CCW to restore
                        pose_detection_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        pose_detection_frame = frame
                    if frame_count <= 3:
                        print(f"Applied {rotation_angle}° counter-rotation to frame {frame_count}")
                else:
                    pose_detection_frame = frame
                
                # Process the properly oriented frame
                output_overlay, keypoints_with_scores, feedback = processor.process_frame(pose_detection_frame, show_feedback=True)
                
                # Adjust text positioning based on orientation
                if original_is_portrait:
                    # Portrait mode - adjust text positions for taller display
                    info_x, info_y = 10, 50
                    mode_x, mode_y = 10, 100
                    font_scale = 0.8
                else:
                    # Landscape mode - use standard positions
                    info_x, info_y = 10, 30
                    mode_x, mode_y = 10, 60
                    font_scale = 0.7
                
                # Add frame info
                # cv2.putText(output_overlay, f"Frame: {frame_count}", (info_x, info_y), 
                #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                
                # # Add mode indicator
                # mode_text = f"Mode: {processor.feedback.exercise_mode.upper()}"
                # cv2.putText(output_overlay, mode_text, (mode_x, mode_y), 
                #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
                
                # Save to output video if enabled (at original resolution)
                if out:
                    # Ensure output frame matches the original dimensions
                    if output_overlay.shape[:2] != (output_height, output_width):
                        output_overlay_saved = cv2.resize(output_overlay, (output_width, output_height))
                    else:
                        output_overlay_saved = output_overlay
                    out.write(output_overlay_saved)
            
            # Display frame with proper window sizing (scaled for UI)
            window_name = 'Exercise Analysis'
            
            # Get the frame to display
            display_frame = output_overlay if not paused else frame
            
            # Resize frame to maintain aspect ratio for display window
            # This is separate from output video dimensions
            frame_height, frame_width = display_frame.shape[:2]
            
            if original_is_portrait:
                # Portrait video - larger display for better visibility
                max_width, max_height = 1080, 1920
            else:
                # Landscape video - standard sizing
                max_width, max_height = 1200, 800
            
            # Calculate scaling factor to fit within max dimensions for UI display
            scale = min(max_width / frame_width, max_height / frame_height)
            window_width = int(frame_width * scale)
            window_height = int(frame_height * scale)
            
            # Resize the display frame to exact window dimensions to avoid white space
            display_frame_resized = cv2.resize(display_frame, (window_width, window_height))
            
            # Use WINDOW_AUTOSIZE to prevent manual resizing and white space
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, display_frame_resized)
            
            # Handle key presses
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('b'):
                processor.feedback.set_exercise_mode("bench")
                print("Switched to BENCH PRESS mode")
            elif key == ord('s'):
                processor.feedback.set_exercise_mode("squat")
                print("Switched to SQUAT mode")
            elif key == ord('d'):
                processor.feedback.set_exercise_mode("deadlift")
                if not hasattr(processor.feedback, "deadlift_analyzer") or processor.feedback.deadlift_analyzer is None:
                    processor.feedback.deadlift_analyzer = DeadliftFormAnalyzer()
                print("Switched to DEADLIFT mode")
            elif key == ord('p'):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            
            # Print feedback every 30 frames
            if not paused and frame_count % 30 == 0 and feedback and feedback.get('form_analysis'):
                form = feedback['form_analysis']
                if processor.feedback.exercise_mode == "bench":
                    # Bench press specific feedback
                    phase = form.get('phase', 'unknown')
                    phase_frames = form.get('phase_frames', 0)
                    depth_metric = form.get('depth_metric', 0)
                    print(f"Frame {frame_count}: Phase: {phase} (frames: {phase_frames}) | Depth: {depth_metric:.1f}")
                else:
                    # Squat specific feedback
                    print(f"Frame {frame_count}: {form.get('back_message', 'N/A')} | {form.get('depth_message', 'N/A')}")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

def get_user_choices():
    """Get user choices for exercise type, video, and output saving."""
    import os
    
    # Automatically set to squat mode
    exercise_mode = "squat"
    print("\n=== Squat Mode (Auto-selected) ===")
    
    # Choose video based on exercise mode
    print(f"\n=== {exercise_mode.capitalize()} Video Selection ===")
    

    if exercise_mode == "squat":
        videos = []
        squat_dir = "data/squat"
        if os.path.exists(squat_dir):
            for file in os.listdir(squat_dir):
                if file.endswith('.mp4'):
                    videos.append(os.path.join(squat_dir, file))
    elif exercise_mode == "bench":
        videos = []
        bench_dir = "data/bench"
        if os.path.exists(bench_dir):
            for file in os.listdir(bench_dir):
                if file.endswith('.mp4'):
                    videos.append(os.path.join(bench_dir, file))
    elif exercise_mode == "deadlift":
        videos = []
        deadlift_dir = "data/deadlift"
        if os.path.exists(deadlift_dir):
            for file in os.listdir(deadlift_dir):
                if file.endswith('.mp4'):
                    videos.append(os.path.join(deadlift_dir, file))
    

    if not videos:
        print(f"No {exercise_mode} videos found!")
        return None, None, None
    
    # Display video options
    for i, video in enumerate(videos, 1):
        print(f"{i}. {os.path.basename(video)}")
    
    while True:
        try:
            choice = int(input(f"Choose video (1-{len(videos)}): ").strip())
            if 1 <= choice <= len(videos):
                video_path = videos[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(videos)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Ask about saving output
    print("\n=== Output Options ===")
    while True:
        save_choice = input("Do you want to save the processed video? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_base = f"output/{base_name}_{exercise_mode}_analysis"
            ext = ".mp4"
            output_path = output_base + ext
            import os
            os.makedirs("output", exist_ok=True)
            idx = 1
            while os.path.exists(output_path):
                output_path = f"{output_base}_{idx}{ext}"
                idx += 1
            break
        elif save_choice in ['n', 'no']:
            output_path = None
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")
    
    return video_path, exercise_mode, output_path

if __name__ == "__main__":
    # Check if command line arguments are provided (for backward compatibility)
    if len(sys.argv) > 1:
        # Use command line arguments
        video_path = sys.argv[1]
        exercise_mode = "bench"  # Default
        output_path = None
        max_frames = None
        
        # Auto-detect exercise mode from video path if not specified
        if "squat" in video_path.lower():
            exercise_mode = "squat"
        elif "bench" in video_path.lower():
            exercise_mode = "bench"
        
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            # Increment output_path if file exists
            if output_path:
                import os
                base, ext = os.path.splitext(output_path)
                candidate = output_path
                idx = 1
                while os.path.exists(candidate):
                    candidate = f"{base}_{idx}{ext}"
                    idx += 1
                output_path = candidate
        
        if len(sys.argv) > 3:
            exercise_mode = sys.argv[3]
            
        if len(sys.argv) > 4:
            max_frames = int(sys.argv[4])
    else:
        # Interactive mode
        result = get_user_choices()
        if result[0] is None:  # No videos found
            sys.exit(1)
        
        video_path, exercise_mode, output_path = result
        max_frames = None
    
    print(f"\n=== Configuration ===")
    print(f"Video: {video_path}")
    print(f"Exercise: {exercise_mode}")
    print(f"Output: {output_path if output_path else 'Display only'}")
    print("")
    
    analyze_video_with_orientation(video_path, exercise_mode, max_frames, output_path)
