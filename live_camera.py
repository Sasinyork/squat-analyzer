#!/usr/bin/env python3
"""
Live camera feed with squat exercise analysis and feedback.
Real-time pose detection with rep counting and form feedback.
"""

import cv2
import sys
import os
import time
from datetime import datetime
from helpers.utils.model_utils import load_movenet_model
from helpers.pose_processor import PoseProcessor


def run_live_camera_squat(save_output=False):
    """Run live camera feed with squat exercise analysis."""
    
    print("Loading MoveNet model...")
    movenet, input_size = load_movenet_model("movenet_lightning")
    
    # Create processor and set to squat mode
    processor = PoseProcessor(movenet, input_size)
    processor.feedback.set_exercise_mode("squat")
    
    # Occlusion handling configuration
    print(f"Occlusion handling: {'ENABLED' if processor.occlusion_handling_enabled else 'DISABLED'}")
    print(f"Side mirroring: {'ENABLED' if processor._enable_side_mirroring else 'DISABLED'}")
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties to portrait resolution (like phone camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Camera reported FPS: {actual_fps}")
    
    # Set up output video writer if saving is enabled
    # We'll calculate actual FPS after processing starts
    out = None
    output_path = None
    temp_frames = []  # Store frames temporarily until we calculate FPS
    if save_output:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/live_squat_{timestamp}.mp4"
    
    print("\nStarting live squat analysis...")
    print("Press 'q' to quit, 'p' to pause, 'r' to reset rep counter")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    fps_calc_frames = 30  # Calculate FPS after this many frames
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process frame with feedback
                output_overlay, keypoints_with_scores, feedback = processor.process_frame(
                    frame, 
                    show_feedback=True
                )
                
                # Add frame counter (optional)
                # cv2.putText(output_overlay, f"Frame: {frame_count}", (10, 30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame to output video if enabled
                if save_output:
                    if out is None:
                        # Calculate actual FPS after first batch of frames
                        if frame_count == fps_calc_frames:
                            elapsed_time = time.time() - start_time
                            calculated_fps = frame_count / elapsed_time
                            print(f"Calculated actual FPS: {calculated_fps:.2f}")
                            
                            # Initialize video writer with calculated FPS
                            # Use H.264 codec for better quality (avc1 for macOS)
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            out = cv2.VideoWriter(output_path, fourcc, calculated_fps, 
                                                (actual_width, actual_height))
                            print(f"Recording to: {output_path}")
                            
                            # Write buffered frames
                            for buffered_frame in temp_frames:
                                out.write(buffered_frame)
                            temp_frames.clear()
                        else:
                            # Buffer frames until we calculate FPS
                            temp_frames.append(output_overlay.copy())
                    else:
                        # Write frame directly once video writer is initialized
                        out.write(output_overlay)
                
                # Print detailed feedback every 30 frames
                if frame_count % 30 == 0 and feedback and feedback.get('form_analysis'):
                    form = feedback['form_analysis']
                    phase = form.get('phase', 'unknown')
                    rep_count = form.get('rep_count', 0)
                    correct_reps = form.get('correct_rep_count', 0)
                    incorrect_reps = form.get('incorrect_rep_count', 0)
                    print(f"Frame {frame_count}: Phase: {phase} | Reps: {rep_count} "
                          f"(Correct: {correct_reps}, Incorrect: {incorrect_reps})")
                
                display_frame = output_overlay
            else:
                # When paused, keep displaying the last frame
                pass
            
            # Display the frame
            window_name = 'Live Squat Analysis'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Set window size for portrait orientation (scaled down to fit screen better)
            cv2.resizeWindow(window_name, 720, 1280)
            cv2.imshow(window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
            elif key == ord('r'):
                # Reset rep counter
                if hasattr(processor.feedback, 'squat_analyzer'):
                    processor.feedback.squat_analyzer.rep_count = 0
                    processor.feedback.squat_analyzer.correct_rep_count = 0
                    processor.feedback.squat_analyzer.incorrect_rep_count = 0
                    print("\nRep counter reset!")
    
    finally:
        cap.release()
        if out:
            out.release()
            print(f"\nVideo saved to: {output_path}")
        cv2.destroyAllWindows()
        
        # Print final statistics
        if hasattr(processor.feedback, 'squat_analyzer'):
            analyzer = processor.feedback.squat_analyzer
            print(f"\n=== Session Summary ===")
            print(f"Total frames processed: {frame_count}")
            print(f"Total reps: {analyzer.rep_count}")
            print(f"Correct reps: {analyzer.correct_rep_count}")
            print(f"Incorrect reps: {analyzer.incorrect_rep_count}")
            if analyzer.rep_count > 0:
                accuracy = (analyzer.correct_rep_count / analyzer.rep_count) * 100
                print(f"Form accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    print("=" * 50)
    print("LIVE SQUAT ANALYSIS")
    print("=" * 50)
    
    # Ask if user wants to save the output
    print("\nDo you want to save the video output?")
    while True:
        save_choice = input("Save output? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            save_output = True
            break
        elif save_choice in ['n', 'no']:
            save_output = False
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")
    
    print("\nMake sure you are visible to the camera.")
    print("Position yourself so your full body is in frame.")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  'r' - Reset rep counter")
    print("\n" + "=" * 50 + "\n")
    
    try:
        run_live_camera_squat(save_output)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
