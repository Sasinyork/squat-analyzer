# main.py
import tensorflow as tf
import numpy as np
import cv2
import os
import helpers.visualization_utils as vis
from helpers.model_utils import load_movenet_model
from helpers.pose_processor import process_video_with_squat_analysis, process_webcam_with_squat_analysis

# Global variables for model
movenet = None
input_size = None

def get_unique_output_path(base_path, suffix="_pose_detected"):
    """Generate a unique output path by incrementing the filename if it already exists."""
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the original filename
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    # Start with the base name
    counter = 0
    while True:
        if counter == 0:
            # First try without number
            output_filename = f"{name}{suffix}{ext}"
        else:
            # Then try with incrementing numbers
            output_filename = f"{name}{suffix}_{counter}{ext}"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # If file doesn't exist, we can use this path
        if not os.path.exists(output_path):
            return output_path
        
        counter += 1

def main():
    """Main function to choose between webcam and video processing."""
    print("MoveNet Lightning - Squat Form Analysis")
    print("=" * 50)
    print("Advanced pose detection with squat form analysis")
    print("Detects: back rounding, knee alignment, depth, arm position")
    print("=" * 50)
    
    print("\nLoading MoveNet Lightning model...")
    global movenet, input_size
    movenet, input_size = load_movenet_model()
    print(f"Model loaded! Input size: {input_size}x{input_size}")
    
    print("\n" + "=" * 50)
    print("1. Squat Form Analysis (Webcam)")
    print("2. Squat Form Analysis (Video File)")
    print("=" * 50)
    
    while True:
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == "1":
            print("\nStarting Squat Form Analysis with Webcam...")
            print("Position yourself for squats and the system will analyze your form.")
            print("Features:")
            print("- Real-time form scoring")
            print("- Back rounding detection")
            print("- Knee alignment analysis")
            print("- Squat depth monitoring")
            print("- Arm position feedback")
            print("- 1080p resolution for maximum clarity")
            print("Press 'q' to quit.")
            
            try:
                process_webcam_with_squat_analysis(movenet, input_size)
            except KeyboardInterrupt:
                print("\nStopped by user.")
            break
            
        elif choice == "2":
            print("\nSquat Form Analysis for Video File")
            video_path = input("Enter the full path to your video file: ").strip()
            
            if not os.path.exists(video_path):
                print(f"Error: File '{video_path}' does not exist.")
                continue
            
            save_output = input("Save processed video? (y/n): ").strip().lower()
            output_path = None
            
            if save_output == 'y':
                output_path = get_unique_output_path(video_path, "_squat_analysis")
                print(f"Output will be saved to: {output_path}")
            
            try:
                process_video_with_squat_analysis(video_path, movenet, input_size, output_path)
            except KeyboardInterrupt:
                print("\nStopped by user.")
            break
            
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()


