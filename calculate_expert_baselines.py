"""
Calculate expert baselines for squat form analysis.

This script processes expert squat videos to determine:
1. Average back angle (nose-shoulder-hip) during bottom position
2. Average hip-knee depth at bottom position
3. Statistical measures (mean, std, min, max) for threshold calibration

Usage:
    python calculate_expert_baselines.py --video_dir data/expert_squats
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json
import tensorflow as tf
from helpers.utils.model_utils import load_movenet_model
from helpers.analyzers.squat_analyzer import SquatFormAnalyzer


class ExpertBaselineCalculator:
    """Calculate baseline metrics from expert squat videos."""
    
    def __init__(self):
        self.back_angles = []  # Back angles at bottom position
        self.hip_knee_depths = []  # Hip-knee depth at bottom position (pixels)
        self.hip_knee_depth_ratios = []  # Hip-knee depth normalized by image height
        
        # Load MoveNet model
        print("Loading MoveNet model...")
        self.movenet, self.input_size = load_movenet_model('model.tflite')
        
        # Use analyzer for phase detection and angle calculation
        self.analyzer = SquatFormAnalyzer()
    
    def run_inference(self, frame):
        """Run MoveNet inference on a frame."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad to model input size
        input_image = tf.expand_dims(image_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        
        # Run MoveNet
        keypoints_with_scores = self.movenet(input_image)
        return keypoints_with_scores
        
    def get_keypoint_coords(self, keypoints, index, image_height, image_width):
        """Get pixel coordinates for a keypoint."""
        if keypoints[index, 2] > 0.15:  # Confidence threshold
            x = int(keypoints[index, 1] * image_width)
            y = int(keypoints[index, 0] * image_height)
            return (x, y)
        return None
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)."""
        if point1 is None or point2 is None or point3 is None:
            return None
            
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def extract_back_angle(self, keypoints, image_height, image_width):
        """Extract back angle (nose-shoulder-hip) from keypoints."""
        nose = self.get_keypoint_coords(keypoints, self.analyzer.NOSE, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.analyzer.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.analyzer.RIGHT_SHOULDER, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.analyzer.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.analyzer.RIGHT_HIP, image_height, image_width)
        
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
        
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        )
        
        return self.calculate_angle(nose, shoulder_center, hip_center)
    
    def extract_hip_knee_depth(self, keypoints, image_height, image_width):
        """Extract hip-knee depth (vertical distance) from keypoints.
        
        Returns:
            tuple: (depth_pixels, depth_ratio)
                   depth_pixels: hip_y - knee_y (positive = hip below knee)
                   depth_ratio: normalized by image height
        """
        left_hip = self.get_keypoint_coords(keypoints, self.analyzer.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.analyzer.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.analyzer.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.analyzer.RIGHT_KNEE, image_height, image_width)
        
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return None, None
        
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        knee_center_y = (left_knee[1] + right_knee[1]) / 2
        
        # In image coords, y increases downward
        # Positive value = hip below knee (good depth)
        # Negative value = hip above knee (insufficient depth)
        depth_pixels = hip_center_y - knee_center_y
        depth_ratio = depth_pixels / image_height
        
        return depth_pixels, depth_ratio
    
    def process_video(self, video_path):
        """Process a single expert video and extract metrics at bottom position."""
        print(f"\nProcessing: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_count = 0
        bottom_frames = 0
        video_back_angles = []
        video_hip_knee_depths = []
        video_hip_knee_ratios = []
        
        # Reset analyzer for new video
        self.analyzer = SquatFormAnalyzer()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            image_height, image_width = frame.shape[:2]
            
            # Run inference
            keypoints_with_scores = self.run_inference(frame)
            keypoints = keypoints_with_scores[0, 0, :, :]
            
            # Detect phase
            phase, _ = self.analyzer.detect_squat_phase(keypoints, image_height, image_width)
            
            # Only collect metrics at bottom position
            if phase == "bottom":
                bottom_frames += 1
                
                # Extract back angle
                back_angle = self.extract_back_angle(keypoints, image_height, image_width)
                if back_angle is not None:
                    video_back_angles.append(back_angle)
                
                # Extract hip-knee depth
                depth_pixels, depth_ratio = self.extract_hip_knee_depth(keypoints, image_height, image_width)
                if depth_pixels is not None:
                    video_hip_knee_depths.append(depth_pixels)
                    video_hip_knee_ratios.append(depth_ratio)
        
        cap.release()
        
        # Store results
        if video_back_angles:
            self.back_angles.extend(video_back_angles)
            print(f"  Back angles at bottom: mean={np.mean(video_back_angles):.1f}°, "
                  f"std={np.std(video_back_angles):.1f}°, "
                  f"min={np.min(video_back_angles):.1f}°, "
                  f"max={np.max(video_back_angles):.1f}°")
        
        if video_hip_knee_depths:
            self.hip_knee_depths.extend(video_hip_knee_depths)
            self.hip_knee_depth_ratios.extend(video_hip_knee_ratios)
            print(f"  Hip-knee depth: mean={np.mean(video_hip_knee_depths):.1f}px, "
                  f"std={np.std(video_hip_knee_depths):.1f}px")
            print(f"  Hip-knee ratio: mean={np.mean(video_hip_knee_ratios):.4f}, "
                  f"std={np.std(video_hip_knee_ratios):.4f}")
        
        print(f"  Processed {frame_count} frames, {bottom_frames} at bottom position")
    
    def process_directory(self, video_dir):
        """Process all videos in a directory."""
        video_dir = Path(video_dir)
        
        if not video_dir.exists():
            print(f"Error: Directory {video_dir} does not exist")
            return
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
        
        if not video_files:
            print(f"No video files found in {video_dir}")
            return
        
        print(f"Found {len(video_files)} video(s) in {video_dir}")
        
        for video_file in sorted(video_files):
            self.process_video(video_file)
    
    def calculate_statistics(self):
        """Calculate and display final statistics."""
        print("\n" + "="*60)
        print("EXPERT BASELINE STATISTICS")
        print("="*60)
        
        if not self.back_angles:
            print("No back angle data collected")
        else:
            print(f"\nBack Angle at Bottom Position ({len(self.back_angles)} measurements):")
            print(f"  Mean:   {np.mean(self.back_angles):.2f}°")
            print(f"  Std:    {np.std(self.back_angles):.2f}°")
            print(f"  Min:    {np.min(self.back_angles):.2f}°")
            print(f"  Max:    {np.max(self.back_angles):.2f}°")
            print(f"  Median: {np.median(self.back_angles):.2f}°")
            
            # Recommended thresholds (mean - 1 std for more permissive)
            recommended_threshold = np.mean(self.back_angles) - np.std(self.back_angles)
            print(f"\n  Recommended threshold (mean - 1 std): {recommended_threshold:.2f}°")
            print(f"  (Back angles below this indicate rounding)")
        
        if not self.hip_knee_depths:
            print("\nNo hip-knee depth data collected")
        else:
            print(f"\nHip-Knee Depth at Bottom Position ({len(self.hip_knee_depths)} measurements):")
            print(f"  Mean:   {np.mean(self.hip_knee_depths):.2f} pixels")
            print(f"  Std:    {np.std(self.hip_knee_depths):.2f} pixels")
            print(f"  Min:    {np.min(self.hip_knee_depths):.2f} pixels")
            print(f"  Max:    {np.max(self.hip_knee_depths):.2f} pixels")
            print(f"  Median: {np.median(self.hip_knee_depths):.2f} pixels")
            
            print(f"\nHip-Knee Depth Ratio (normalized by image height):")
            print(f"  Mean:   {np.mean(self.hip_knee_depth_ratios):.4f}")
            print(f"  Std:    {np.std(self.hip_knee_depth_ratios):.4f}")
            print(f"  Min:    {np.min(self.hip_knee_depth_ratios):.4f}")
            print(f"  Max:    {np.max(self.hip_knee_depth_ratios):.4f}")
            print(f"  Median: {np.median(self.hip_knee_depth_ratios):.4f}")
            
            # Recommended depth allowance (mean - 1 std for acceptable range)
            recommended_depth_allowance = np.mean(self.hip_knee_depth_ratios) - np.std(self.hip_knee_depth_ratios)
            print(f"\n  Recommended depth allowance ratio: {recommended_depth_allowance:.4f}")
            print(f"  (Hip can be this ratio above knee and still be acceptable)")
            print(f"  (Negative ratio = hip below knee = excellent depth)")
    
    def save_results(self, output_file):
        """Save results to JSON file."""
        results = {
            'back_angle': {
                'measurements': len(self.back_angles),
                'mean': float(np.mean(self.back_angles)) if self.back_angles else None,
                'std': float(np.std(self.back_angles)) if self.back_angles else None,
                'min': float(np.min(self.back_angles)) if self.back_angles else None,
                'max': float(np.max(self.back_angles)) if self.back_angles else None,
                'median': float(np.median(self.back_angles)) if self.back_angles else None,
                'recommended_threshold': float(np.mean(self.back_angles) - np.std(self.back_angles)) if self.back_angles else None,
            },
            'hip_knee_depth_pixels': {
                'measurements': len(self.hip_knee_depths),
                'mean': float(np.mean(self.hip_knee_depths)) if self.hip_knee_depths else None,
                'std': float(np.std(self.hip_knee_depths)) if self.hip_knee_depths else None,
                'min': float(np.min(self.hip_knee_depths)) if self.hip_knee_depths else None,
                'max': float(np.max(self.hip_knee_depths)) if self.hip_knee_depths else None,
                'median': float(np.median(self.hip_knee_depths)) if self.hip_knee_depths else None,
            },
            'hip_knee_depth_ratio': {
                'measurements': len(self.hip_knee_depth_ratios),
                'mean': float(np.mean(self.hip_knee_depth_ratios)) if self.hip_knee_depth_ratios else None,
                'std': float(np.std(self.hip_knee_depth_ratios)) if self.hip_knee_depth_ratios else None,
                'min': float(np.min(self.hip_knee_depth_ratios)) if self.hip_knee_depth_ratios else None,
                'max': float(np.max(self.hip_knee_depth_ratios)) if self.hip_knee_depth_ratios else None,
                'median': float(np.median(self.hip_knee_depth_ratios)) if self.hip_knee_depth_ratios else None,
                'recommended_allowance': float(np.mean(self.hip_knee_depth_ratios) - np.std(self.hip_knee_depth_ratios)) if self.hip_knee_depth_ratios else None,
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate expert baselines for squat form analysis'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Directory containing expert squat videos'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='expert_baselines.json',
        help='Output file for baseline results (default: expert_baselines.json)'
    )
    
    args = parser.parse_args()
    
    calculator = ExpertBaselineCalculator()
    calculator.process_directory(args.video_dir)
    calculator.calculate_statistics()
    calculator.save_results(args.output)


if __name__ == '__main__':
    main()
