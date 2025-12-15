#!/usr/bin/env python3
"""
Batch Metrics Analysis for Multiple Videos
Calculates average metrics (jitter, stability, smoothness, etc.) for multiple videos.
No live preview for faster processing.
"""
import numpy as np
import cv2
import os
import sys
from helpers.utils.model_utils import load_movenet_model
from helpers.pose_processor import PoseProcessor


class VideoMetricsAnalyzer:
    def __init__(self):
        self.metrics = {
            'jitter': [],
            'temporal_consistency': [],
            'confidence': [],
            'stability': [],
            'smoothness': [],
            'responsiveness': [],
        }
        self.position_history = []
        self.velocity_history = []
    
    def calculate_jitter(self, positions):
        """Calculate jitter as mean absolute acceleration"""
        if len(positions) < 3:
            return 0.0
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        return float(np.mean(np.abs(accelerations)))
    
    def calculate_temporal_consistency(self, positions):
        """Standard deviation of positions over time"""
        if len(positions) < 2:
            return 0.0
        return float(np.std(positions, axis=0).mean())
    
    def calculate_stability(self, positions):
        """Calculate position stability using variance"""
        if len(positions) < 2:
            return 0.0
        return float(np.var(positions, axis=0).mean())
    
    def calculate_smoothness(self, positions):
        """Calculate smoothness using velocity changes (jerk)"""
        if len(positions) < 4:
            return 0.0
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerk = np.diff(accelerations, axis=0)  # Third derivative
        return float(np.mean(np.abs(jerk)))
    
    def calculate_responsiveness(self, velocities):
        """Calculate responsiveness as mean velocity magnitude"""
        if len(velocities) < 1:
            return 0.0
        velocity_magnitudes = np.linalg.norm(velocities, axis=2).mean(axis=1)
        return float(np.mean(velocity_magnitudes))
    
    def update(self, keypoints_with_scores):
        """Update metrics with new frame"""
        # Store positions (17x2 for y,x)
        self.position_history.append(keypoints_with_scores[:, :2].copy())
        
        # Calculate and store velocities
        if len(self.position_history) > 1:
            velocity = self.position_history[-1] - self.position_history[-2]
            self.velocity_history.append(velocity)
        
        # Keep last 30 frames for rolling metrics
        if len(self.position_history) > 30:
            self.position_history.pop(0)
        
        if len(self.velocity_history) > 30:
            self.velocity_history.pop(0)
        
        # Calculate metrics if we have enough history
        if len(self.position_history) >= 10:
            positions = np.array(self.position_history)
            velocities = np.array(self.velocity_history) if self.velocity_history else np.array([])
            
            # Jitter (acceleration-based)
            jitter = self.calculate_jitter(positions)
            self.metrics['jitter'].append(jitter)
            
            # Temporal consistency (position stability)
            consistency = self.calculate_temporal_consistency(positions)
            self.metrics['temporal_consistency'].append(consistency)
            
            # Stability (variance-based)
            stability = self.calculate_stability(positions)
            self.metrics['stability'].append(stability)
            
            # Smoothness (jerk-based)
            smoothness = self.calculate_smoothness(positions)
            self.metrics['smoothness'].append(smoothness)
            
            # Responsiveness (velocity-based)
            if len(velocities) > 0:
                responsiveness = self.calculate_responsiveness(velocities)
                self.metrics['responsiveness'].append(responsiveness)
            
            # Confidence (only for current frame)
            conf = float(keypoints_with_scores[:, 2].mean())
            self.metrics['confidence'].append(conf)
    
    def get_summary(self):
        """Get summary statistics"""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_min'] = np.min(values)
                summary[f'{metric_name}_max'] = np.max(values)
            else:
                summary[f'{metric_name}_mean'] = 0
                summary[f'{metric_name}_std'] = 0
                summary[f'{metric_name}_min'] = 0
                summary[f'{metric_name}_max'] = 0
        
        return summary


def analyze_video(video_path, movenet, input_size, use_smoothing=True):
    """Analyze a single video and return metrics"""
    mode = "SMOOTHED" if use_smoothing else "RAW"
    print(f"\nProcessing ({mode}): {os.path.basename(video_path)}")
    
    # Create processor with smoothing enabled/disabled
    processor = PoseProcessor(movenet, input_size)
    processor.spatiotemporal_enabled = use_smoothing
    if use_smoothing:
        processor._one_euro_enabled = True
    
    analyzer = VideoMetricsAnalyzer()
    
    # Open video with proper orientation handling
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"  Error: Could not open video file: {video_path}")
        return None
    
    # Try to disable automatic rotation
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except:
        pass
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame to check orientation
    ret, test_frame = cap.read()
    if not ret:
        print("  Error: Could not read first frame")
        cap.release()
        return None
    
    actual_height, actual_width = test_frame.shape[:2]
    
    # Detect if this is a portrait video that was rotated by OpenCV
    needs_counter_rotation = False
    if (metadata_width == 1920 and metadata_height == 1080 and 
        actual_width == 1920 and actual_height == 1080):
        needs_counter_rotation = True
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    print(f"  Total frames: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress indicator every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')
        
        # Apply counter-rotation if needed
        if needs_counter_rotation:
            pose_detection_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            pose_detection_frame = frame
        
        # Process frame
        _, keypoints_with_scores, _ = processor.process_frame(pose_detection_frame.copy(), show_feedback=False)
        
        # Update metrics
        analyzer.update(keypoints_with_scores[0, 0, :, :])
    
    cap.release()
    
    print(f"  Completed: {frame_count} frames processed")
    
    return analyzer.get_summary()


def select_videos():
    """Allow user to select multiple videos from squat directory"""
    squat_dir = "data/squat"
    
    if not os.path.exists(squat_dir):
        print(f"Error: Directory '{squat_dir}' not found!")
        return []
    
    # Get all video files
    all_videos = []
    for file in os.listdir(squat_dir):
        if file.endswith('.mp4'):
            all_videos.append(os.path.join(squat_dir, file))
    
    if not all_videos:
        print(f"No video files found in '{squat_dir}'")
        return []
    
    # Sort videos alphabetically
    all_videos.sort()
    
    print("\n=== Available Squat Videos ===")
    for i, video in enumerate(all_videos, 1):
        print(f"{i}. {os.path.basename(video)}")
    
    print("\nSelect videos to analyze:")
    print("  - Enter video numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter 'all' to analyze all videos")
    print("  - Enter a range with dash (e.g., 1-3)")
    
    while True:
        choice = input("\nYour selection: ").strip().lower()
        
        if choice == 'all':
            return all_videos
        
        try:
            selected_videos = []
            
            # Handle ranges and individual selections
            parts = choice.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Range
                    start, end = map(int, part.split('-'))
                    if 1 <= start <= len(all_videos) and 1 <= end <= len(all_videos):
                        for i in range(start, end + 1):
                            video_path = all_videos[i - 1]
                            if video_path not in selected_videos:
                                selected_videos.append(video_path)
                    else:
                        print(f"Invalid range: {part}")
                        selected_videos = []
                        break
                else:
                    # Individual number
                    num = int(part)
                    if 1 <= num <= len(all_videos):
                        video_path = all_videos[num - 1]
                        if video_path not in selected_videos:
                            selected_videos.append(video_path)
                    else:
                        print(f"Invalid number: {num}")
                        selected_videos = []
                        break
            
            if selected_videos:
                return selected_videos
            
        except ValueError:
            print("Invalid input. Please try again.")


def print_comparison_table(results):
    """Print a formatted comparison table with raw vs smoothed results"""
    print("\n" + "="*150)
    print("BATCH COMPARISON ANALYSIS: RAW vs SMOOTHED MODEL")
    print("="*150)
    
    # Summary table - Raw vs Smoothed side by side
    print(f"\n{'Video':<30} {'Model':<10} {'Jitter':<15} {'Stability':<15} {'Smoothness':<15} {'Resp.':<15} {'Conf.':<10}")
    print("-"*150)
    
    for video_path, comparison in results.items():
        video_name = os.path.basename(video_path)
        if len(video_name) > 28:
            video_name = video_name[:25] + "..."
        
        raw = comparison['raw']
        smoothed = comparison['smoothed']
        
        # Print raw metrics
        print(f"{video_name:<30} {'RAW':<10} "
              f"{raw['jitter_mean']:.6f}   "
              f"{raw['stability_mean']:.6f}   "
              f"{raw['smoothness_mean']:.6f}   "
              f"{raw['responsiveness_mean']:.6f}   "
              f"{raw['confidence_mean']:.3f}")
        
        # Print smoothed metrics
        print(f"{'':<30} {'SMOOTHED':<10} "
              f"{smoothed['jitter_mean']:.6f}   "
              f"{smoothed['stability_mean']:.6f}   "
              f"{smoothed['smoothness_mean']:.6f}   "
              f"{smoothed['responsiveness_mean']:.6f}   "
              f"{smoothed['confidence_mean']:.3f}")
        
        # Calculate improvements
        jitter_impr = ((raw['jitter_mean'] - smoothed['jitter_mean']) / raw['jitter_mean'] * 100) if raw['jitter_mean'] > 0 else 0
        stability_impr = ((raw['stability_mean'] - smoothed['stability_mean']) / raw['stability_mean'] * 100) if raw['stability_mean'] > 0 else 0
        smoothness_impr = ((raw['smoothness_mean'] - smoothed['smoothness_mean']) / raw['smoothness_mean'] * 100) if raw['smoothness_mean'] > 0 else 0
        resp_change = ((smoothed['responsiveness_mean'] - raw['responsiveness_mean']) / raw['responsiveness_mean'] * 100) if raw['responsiveness_mean'] > 0 else 0
        
        # Print improvements
        print(f"{'':<30} {'IMPROVE':<10} "
              f"{jitter_impr:+.1f}%         "
              f"{stability_impr:+.1f}%         "
              f"{smoothness_impr:+.1f}%         "
              f"{resp_change:+.1f}%         "
              f"{'—':<10}")
        print()
    
    print("-"*150)
    
    # Overall averages
    if len(results) > 1:
        avg_raw_jitter = np.mean([c['raw']['jitter_mean'] for c in results.values()])
        avg_raw_stability = np.mean([c['raw']['stability_mean'] for c in results.values()])
        avg_raw_smoothness = np.mean([c['raw']['smoothness_mean'] for c in results.values()])
        avg_raw_resp = np.mean([c['raw']['responsiveness_mean'] for c in results.values()])
        avg_raw_conf = np.mean([c['raw']['confidence_mean'] for c in results.values()])
        
        avg_smooth_jitter = np.mean([c['smoothed']['jitter_mean'] for c in results.values()])
        avg_smooth_stability = np.mean([c['smoothed']['stability_mean'] for c in results.values()])
        avg_smooth_smoothness = np.mean([c['smoothed']['smoothness_mean'] for c in results.values()])
        avg_smooth_resp = np.mean([c['smoothed']['responsiveness_mean'] for c in results.values()])
        avg_smooth_conf = np.mean([c['smoothed']['confidence_mean'] for c in results.values()])
        
        print(f"{'OVERALL AVERAGE':<30} {'RAW':<10} "
              f"{avg_raw_jitter:.6f}   "
              f"{avg_raw_stability:.6f}   "
              f"{avg_raw_smoothness:.6f}   "
              f"{avg_raw_resp:.6f}   "
              f"{avg_raw_conf:.3f}")
        
        print(f"{'':<30} {'SMOOTHED':<10} "
              f"{avg_smooth_jitter:.6f}   "
              f"{avg_smooth_stability:.6f}   "
              f"{avg_smooth_smoothness:.6f}   "
              f"{avg_smooth_resp:.6f}   "
              f"{avg_smooth_conf:.3f}")
        
        overall_jitter_impr = ((avg_raw_jitter - avg_smooth_jitter) / avg_raw_jitter * 100) if avg_raw_jitter > 0 else 0
        overall_stability_impr = ((avg_raw_stability - avg_smooth_stability) / avg_raw_stability * 100) if avg_raw_stability > 0 else 0
        overall_smoothness_impr = ((avg_raw_smoothness - avg_smooth_smoothness) / avg_raw_smoothness * 100) if avg_raw_smoothness > 0 else 0
        overall_resp_change = ((avg_smooth_resp - avg_raw_resp) / avg_raw_resp * 100) if avg_raw_resp > 0 else 0
        
        print(f"{'':<30} {'IMPROVE':<10} "
              f"{overall_jitter_impr:+.1f}%         "
              f"{overall_stability_impr:+.1f}%         "
              f"{overall_smoothness_impr:+.1f}%         "
              f"{overall_resp_change:+.1f}%         "
              f"{'—':<10}")
    
    print("="*150)
    
    # Detailed breakdown for each video
    print("\n" + "="*150)
    print("DETAILED COMPARISON BREAKDOWN")
    print("="*150)
    
    for video_path, comparison in results.items():
        video_name = os.path.basename(video_path)
        raw = comparison['raw']
        smoothed = comparison['smoothed']
        
        print(f"\n{video_name}:")
        print(f"\n  RAW MODEL:")
        print(f"    Jitter (Acceleration):        {raw['jitter_mean']:.6f} ± {raw['jitter_std']:.6f} (min: {raw['jitter_min']:.6f}, max: {raw['jitter_max']:.6f})")
        print(f"    Temporal Consistency:         {raw['temporal_consistency_mean']:.6f} ± {raw['temporal_consistency_std']:.6f}")
        print(f"    Stability (Variance):         {raw['stability_mean']:.6f} ± {raw['stability_std']:.6f}")
        print(f"    Smoothness (Jerk):            {raw['smoothness_mean']:.6f} ± {raw['smoothness_std']:.6f}")
        print(f"    Responsiveness (Velocity):    {raw['responsiveness_mean']:.6f} ± {raw['responsiveness_std']:.6f}")
        print(f"    Average Confidence:           {raw['confidence_mean']:.3f} ± {raw['confidence_std']:.3f}")
        
        print(f"\n  SMOOTHED MODEL:")
        print(f"    Jitter (Acceleration):        {smoothed['jitter_mean']:.6f} ± {smoothed['jitter_std']:.6f} (min: {smoothed['jitter_min']:.6f}, max: {smoothed['jitter_max']:.6f})")
        print(f"    Temporal Consistency:         {smoothed['temporal_consistency_mean']:.6f} ± {smoothed['temporal_consistency_std']:.6f}")
        print(f"    Stability (Variance):         {smoothed['stability_mean']:.6f} ± {smoothed['stability_std']:.6f}")
        print(f"    Smoothness (Jerk):            {smoothed['smoothness_mean']:.6f} ± {smoothed['smoothness_std']:.6f}")
        print(f"    Responsiveness (Velocity):    {smoothed['responsiveness_mean']:.6f} ± {smoothed['responsiveness_std']:.6f}")
        print(f"    Average Confidence:           {smoothed['confidence_mean']:.3f} ± {smoothed['confidence_std']:.3f}")
        
        # Calculate improvements
        jitter_impr = ((raw['jitter_mean'] - smoothed['jitter_mean']) / raw['jitter_mean'] * 100) if raw['jitter_mean'] > 0 else 0
        consistency_impr = ((raw['temporal_consistency_mean'] - smoothed['temporal_consistency_mean']) / raw['temporal_consistency_mean'] * 100) if raw['temporal_consistency_mean'] > 0 else 0
        stability_impr = ((raw['stability_mean'] - smoothed['stability_mean']) / raw['stability_mean'] * 100) if raw['stability_mean'] > 0 else 0
        smoothness_impr = ((raw['smoothness_mean'] - smoothed['smoothness_mean']) / raw['smoothness_mean'] * 100) if raw['smoothness_mean'] > 0 else 0
        resp_change = ((smoothed['responsiveness_mean'] - raw['responsiveness_mean']) / raw['responsiveness_mean'] * 100) if raw['responsiveness_mean'] > 0 else 0
        
        print(f"\n  IMPROVEMENTS:")
        print(f"    Jitter Reduction:        {jitter_impr:+.1f}%")
        print(f"    Consistency Improvement: {consistency_impr:+.1f}%")
        print(f"    Stability Improvement:   {stability_impr:+.1f}%")
        print(f"    Smoothness Improvement:  {smoothness_impr:+.1f}%")
        print(f"    Responsiveness Change:   {resp_change:+.1f}%")
    
    print("\n" + "="*150)
    print("\nMetric Definitions:")
    print("- Jitter: Mean absolute acceleration (lower = less shaky)")
    print("- Temporal Consistency: Position std deviation over time (lower = more stable)")
    print("- Stability: Position variance (lower = less wandering)")
    print("- Smoothness: Mean absolute jerk/3rd derivative (lower = smoother motion)")
    print("- Responsiveness: Mean velocity magnitude (measures tracking speed)")
    print("- Confidence: Average detection confidence (higher = better)")
    print("\nImprovement Interpretation:")
    print("- Positive % = smoothed model is better (for jitter, stability, smoothness)")
    print("- Responsiveness change shows impact on tracking speed (small change is ideal)")
    print("="*150)


def main():
    """Main function"""
    # Select videos
    selected_videos = select_videos()
    
    if not selected_videos:
        print("No videos selected. Exiting.")
        return
    
    print(f"\n=== Selected {len(selected_videos)} video(s) for analysis ===")
    for video in selected_videos:
        print(f"  - {os.path.basename(video)}")
    
    # Load model once
    print(f"\nLoading MoveNet Lightning model...")
    movenet, input_size = load_movenet_model("movenet_lightning")
    print("Model loaded successfully!")
    
    # Process each video (both raw and smoothed)
    results = {}
    for i, video_path in enumerate(selected_videos, 1):
        print(f"\n[{i}/{len(selected_videos)}] {os.path.basename(video_path)}")
        
        # Analyze with raw model
        raw_metrics = analyze_video(video_path, movenet, input_size, use_smoothing=False)
        
        # Analyze with smoothed model
        smoothed_metrics = analyze_video(video_path, movenet, input_size, use_smoothing=True)
        
        if raw_metrics and smoothed_metrics:
            results[video_path] = {
                'raw': raw_metrics,
                'smoothed': smoothed_metrics
            }
    
    # Print comparison summary
    if results:
        print_comparison_table(results)
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()
