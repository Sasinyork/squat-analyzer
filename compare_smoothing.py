#!/usr/bin/env python3
"""
Compare raw MoveNet Lightning vs Spatio-Temporal Enhanced version.
Provides quantitative metrics on smoothness, stability, and tracking quality.
"""
import numpy as np
import cv2
import sys
from helpers.utils.model_utils import load_movenet_model
from helpers.pose_processor import PoseProcessor


class SmoothingComparison:
    def __init__(self):
        self.metrics = {
            'raw': {
                'jitter': [],
                'temporal_consistency': [],
                'confidence': [],
                'outliers': [],
                'stability': [],
                'smoothness': [],
                'responsiveness': [],
            },
            'smoothed': {
                'jitter': [],
                'temporal_consistency': [],
                'confidence': [],
                'outliers': [],
                'stability': [],
                'smoothness': [],
                'responsiveness': [],
            }
        }
        self.position_history = {'raw': [], 'smoothed': []}
        self.velocity_history = {'raw': [], 'smoothed': []}
    
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
    
    def update(self, raw_kps, smoothed_kps):
        """Update metrics with new frame"""
        # Store positions (17x2 for y,x)
        self.position_history['raw'].append(raw_kps[:, :2].copy())
        self.position_history['smoothed'].append(smoothed_kps[:, :2].copy())
        
        # Calculate and store velocities
        if len(self.position_history['raw']) > 1:
            for key in ['raw', 'smoothed']:
                velocity = self.position_history[key][-1] - self.position_history[key][-2]
                self.velocity_history[key].append(velocity)
        
        # Keep last 30 frames
        if len(self.position_history['raw']) > 30:
            self.position_history['raw'].pop(0)
            self.position_history['smoothed'].pop(0)
        
        if len(self.velocity_history['raw']) > 30:
            self.velocity_history['raw'].pop(0)
            self.velocity_history['smoothed'].pop(0)
        
        # Calculate metrics if we have enough history
        if len(self.position_history['raw']) >= 10:
            for key in ['raw', 'smoothed']:
                positions = np.array(self.position_history[key])
                velocities = np.array(self.velocity_history[key]) if self.velocity_history[key] else np.array([])
                
                # Jitter (acceleration-based)
                jitter = self.calculate_jitter(positions)
                self.metrics[key]['jitter'].append(jitter)
                
                # Temporal consistency (position stability)
                consistency = self.calculate_temporal_consistency(positions)
                self.metrics[key]['temporal_consistency'].append(consistency)
                
                # Stability (variance-based)
                stability = self.calculate_stability(positions)
                self.metrics[key]['stability'].append(stability)
                
                # Smoothness (jerk-based)
                smoothness = self.calculate_smoothness(positions)
                self.metrics[key]['smoothness'].append(smoothness)
                
                # Responsiveness (velocity-based)
                if len(velocities) > 0:
                    responsiveness = self.calculate_responsiveness(velocities)
                    self.metrics[key]['responsiveness'].append(responsiveness)
                
                # Confidence (only for current frame)
                kps = smoothed_kps if key == 'smoothed' else raw_kps
                conf = float(kps[:, 2].mean())
                self.metrics[key]['confidence'].append(conf)
    
    def get_summary(self):
        """Get summary statistics"""
        summary = {}
        for key in ['raw', 'smoothed']:
            summary[key] = {
                'jitter_mean': np.mean(self.metrics[key]['jitter']) if self.metrics[key]['jitter'] else 0,
                'jitter_std': np.std(self.metrics[key]['jitter']) if self.metrics[key]['jitter'] else 0,
                'consistency_mean': np.mean(self.metrics[key]['temporal_consistency']) if self.metrics[key]['temporal_consistency'] else 0,
                'stability_mean': np.mean(self.metrics[key]['stability']) if self.metrics[key]['stability'] else 0,
                'smoothness_mean': np.mean(self.metrics[key]['smoothness']) if self.metrics[key]['smoothness'] else 0,
                'responsiveness_mean': np.mean(self.metrics[key]['responsiveness']) if self.metrics[key]['responsiveness'] else 0,
                'confidence_mean': np.mean(self.metrics[key]['confidence']) if self.metrics[key]['confidence'] else 0,
            }
        
        # Improvement percentages (lower is better for jitter, consistency, stability, smoothness)
        improvements = {}
        if summary['raw']['jitter_mean'] > 0:
            improvements['jitter_reduction'] = (1 - summary['smoothed']['jitter_mean'] / summary['raw']['jitter_mean']) * 100
        else:
            improvements['jitter_reduction'] = 0
            
        if summary['raw']['consistency_mean'] > 0:
            improvements['consistency_improvement'] = (1 - summary['smoothed']['consistency_mean'] / summary['raw']['consistency_mean']) * 100
        else:
            improvements['consistency_improvement'] = 0
            
        if summary['raw']['stability_mean'] > 0:
            improvements['stability_improvement'] = (1 - summary['smoothed']['stability_mean'] / summary['raw']['stability_mean']) * 100
        else:
            improvements['stability_improvement'] = 0
            
        if summary['raw']['smoothness_mean'] > 0:
            improvements['smoothness_improvement'] = (1 - summary['smoothed']['smoothness_mean'] / summary['raw']['smoothness_mean']) * 100
        else:
            improvements['smoothness_improvement'] = 0
        
        # Responsiveness change (tracking how much smoothing affects movement speed)
        if summary['raw']['responsiveness_mean'] > 0:
            improvements['responsiveness_change'] = ((summary['smoothed']['responsiveness_mean'] / summary['raw']['responsiveness_mean']) - 1) * 100
        else:
            improvements['responsiveness_change'] = 0
        
        summary['improvements'] = {k: f"{v:.1f}%" for k, v in improvements.items()}
        
        return summary


def draw_live_metrics_overlay(frame, metrics, frame_count):
    """Draw live comparison metrics on the frame"""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Color palette from feedback_utils (BGR format)
    # Muted, natural colors matching the feedback system
    COLOR_PALETTE = {
        'good': (90, 170, 90),          # Muted green - good performance
        'warning': (60, 160, 255),      # Soft amber/orange - needs attention
        'error': (75, 74, 179),         # Muted red - poor performance
        'neutral': (180, 180, 180),     # Light gray - neutral/stable
        'info': (200, 180, 180),        # Light blue-gray - informational
        'title': (60, 220, 255),        # Soft yellow - title/highlight
    }
    
    # Get current metrics
    if len(metrics['raw']['jitter']) > 0:
        raw_jitter = metrics['raw']['jitter'][-1]
        smooth_jitter = metrics['smoothed']['jitter'][-1]
        raw_conf = metrics['raw']['confidence'][-1]
        smooth_conf = metrics['smoothed']['confidence'][-1]
        
        # Get additional metrics if available
        raw_stability = metrics['raw']['stability'][-1] if metrics['raw']['stability'] else 0
        smooth_stability = metrics['smoothed']['stability'][-1] if metrics['smoothed']['stability'] else 0
        raw_smoothness = metrics['raw']['smoothness'][-1] if metrics['raw']['smoothness'] else 0
        smooth_smoothness = metrics['smoothed']['smoothness'][-1] if metrics['smoothed']['smoothness'] else 0
        raw_responsiveness = metrics['raw']['responsiveness'][-1] if metrics['raw']['responsiveness'] else 0
        smooth_responsiveness = metrics['smoothed']['responsiveness'][-1] if metrics['smoothed']['responsiveness'] else 0
        
        # Calculate improvements
        jitter_improvement = ((raw_jitter - smooth_jitter) / raw_jitter * 100) if raw_jitter > 0 else 0
        stability_improvement = ((raw_stability - smooth_stability) / raw_stability * 100) if raw_stability > 0 else 0
        smoothness_improvement = ((raw_smoothness - smooth_smoothness) / raw_smoothness * 100) if raw_smoothness > 0 else 0
        responsiveness_change = ((smooth_responsiveness - raw_responsiveness) / raw_responsiveness * 100) if raw_responsiveness > 0 else 0
        
        # Metrics box dimensions (scaled up for full resolution and more metrics)
        box_height = 420
        box_width = 1600
        box_x1 = (width - box_width) // 2
        box_x2 = box_x1 + box_width
        box_y1 = height - box_height - 40
        box_y2 = height - 40
        
        # Semi-transparent background for metrics at bottom center (dark blue-ish like feedback)
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (80, 86, 106), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title centered - using soft yellow from palette
        title = "LIVE COMPARISON: Raw vs Spatio-Temporal Smoothing"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        title_x = (width - title_size[0]) // 2
        cv2.putText(frame, title, 
                    (title_x, box_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_PALETTE['title'], 3)
        
        # Left column - RAW MODEL (using error color - muted red)
        left_x = box_x1 + 40
        y_offset = box_y1 + 110
        line_height = 45
        
        cv2.putText(frame, "RAW MODEL:", 
                    (left_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_PALETTE['error'], 3)
        y_offset += line_height
        cv2.putText(frame, f"Jitter: {raw_jitter:.6f}", 
                    (left_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Stability: {raw_stability:.6f}", 
                    (left_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Smoothness: {raw_smoothness:.6f}", 
                    (left_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Responsiveness: {raw_responsiveness:.6f}", 
                    (left_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Confidence: {raw_conf:.3f}", 
                    (left_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        # Middle column - SMOOTHED MODEL (using good color - muted green)
        mid_x = box_x1 + 560
        y_offset = box_y1 + 110
        
        cv2.putText(frame, "SMOOTHED MODEL:", 
                    (mid_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_PALETTE['good'], 3)
        y_offset += line_height
        cv2.putText(frame, f"Jitter: {smooth_jitter:.6f}", 
                    (mid_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Stability: {smooth_stability:.6f}", 
                    (mid_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Smoothness: {smooth_smoothness:.6f}", 
                    (mid_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Responsiveness: {smooth_responsiveness:.6f}", 
                    (mid_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y_offset += line_height
        cv2.putText(frame, f"Confidence: {smooth_conf:.3f}", 
                    (mid_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        # Right column - Improvements
        right_x = box_x1 + 1120
        y_offset = box_y1 + 110
        
        cv2.putText(frame, "IMPROVEMENTS:", 
                    (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_PALETTE['info'], 3)
        y_offset += line_height
        
        # Jitter improvement
        jitter_color = COLOR_PALETTE['good'] if jitter_improvement > 0 else COLOR_PALETTE['error']
        cv2.putText(frame, f"Jitter: {jitter_improvement:.1f}%", 
                    (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, jitter_color, 2)
        y_offset += line_height
        
        # Stability improvement
        stability_color = COLOR_PALETTE['good'] if stability_improvement > 0 else COLOR_PALETTE['error']
        cv2.putText(frame, f"Stability: {stability_improvement:.1f}%", 
                    (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, stability_color, 2)
        y_offset += line_height
        
        # Smoothness improvement
        smoothness_color = COLOR_PALETTE['good'] if smoothness_improvement > 0 else COLOR_PALETTE['error']
        cv2.putText(frame, f"Smoothness: {smoothness_improvement:.1f}%", 
                    (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, smoothness_color, 2)
        y_offset += line_height
        
        # Responsiveness change (green if positive/near zero, red if negative)
        resp_color = COLOR_PALETTE['good'] if responsiveness_change >= 0 else COLOR_PALETTE['error']
        cv2.putText(frame, f"Resp: {responsiveness_change:+.1f}%", 
                    (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, resp_color, 2)
        y_offset += line_height + 5
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", 
                    (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PALETTE['neutral'], 2)
    
    return frame


def compare_on_video(video_path, show_live=True, output_path=None):
    """Compare raw vs smoothed on a video with optional live display"""
    print(f"Loading MoveNet Lightning model...")
    movenet, input_size = load_movenet_model("movenet_lightning")
    
    # Two processors: one with smoothing, one without
    # Important: Both share the same model but have independent smoothing state
    processor_raw = PoseProcessor(movenet, input_size)
    processor_raw.spatiotemporal_enabled = False
    # Don't touch other settings - use defaults
    
    processor_smoothed = PoseProcessor(movenet, input_size)
    processor_smoothed.spatiotemporal_enabled = True
    # Enable One Euro Filter for adaptive smoothing
    processor_smoothed._one_euro_enabled = True
    processor_smoothed._one_euro_min_cutoff = 0.5
    processor_smoothed._one_euro_beta = 5  # Higher beta for better visual smoothness
    print(f"DEBUG: Smoothed processor spatiotemporal: {processor_smoothed.spatiotemporal_enabled}")
    print(f"DEBUG: Smoothed processor One Euro Filter: {processor_smoothed._one_euro_enabled}")
    print(f"DEBUG: One Euro min_cutoff: {processor_smoothed._one_euro_min_cutoff}")
    print(f"DEBUG: One Euro beta: {processor_smoothed._one_euro_beta}")
    print(f"DEBUG: Smoothed processor temporal_alpha: {processor_smoothed.temporal_alpha}")
    print(f"DEBUG: Smoothed processor spatial_beta: {processor_smoothed.spatial_beta}")
    print(f"DEBUG: Smoothed processor temporal_alpha: {processor_smoothed.temporal_alpha}")
    print(f"DEBUG: Smoothed processor spatial_beta: {processor_smoothed.spatial_beta}")
    
    comparison = SmoothingComparison()
    
    # Open video with proper orientation handling (same as main.py)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Try to disable automatic rotation
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except:
        pass
    
    # Get video properties and detect orientation
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    metadata_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame to check orientation
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    actual_height, actual_width = test_frame.shape[:2]
    
    # Detect if this is a portrait video that was rotated by OpenCV
    needs_counter_rotation = False
    original_is_portrait = False
    
    if (metadata_width == 1920 and metadata_height == 1080 and 
        actual_width == 1920 and actual_height == 1080):
        print("DETECTED: Portrait video rotated to landscape by OpenCV")
        needs_counter_rotation = True
        original_is_portrait = True
        display_width, display_height = 1080, 1920
    else:
        display_width, display_height = actual_width, actual_height
        original_is_portrait = actual_height > actual_width
    
    print(f"Video orientation: {'Portrait' if original_is_portrait else 'Landscape'}")
    print(f"Display dimensions: {display_width}x{display_height}")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # VideoWriter will be initialized after we know the actual combined frame size
    out = None
    output_width = None
    output_height = None
    
    frame_count = 0
    
    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}")
    if show_live:
        print("Live display enabled - Press 'q' to quit, 'p' to pause/resume")
    print("-" * 60)
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
        
        if not show_live and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        if not paused:
            # Apply counter-rotation if needed to restore original orientation
            if needs_counter_rotation:
                # Counter-rotate from landscape back to portrait (rotate 90 degrees clockwise)
                pose_detection_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                pose_detection_frame = frame
            
            # Process with raw model
            frame_raw, kps_raw, _ = processor_raw.process_frame(pose_detection_frame.copy(), show_feedback=False)
            
            # Process with smoothed model
            frame_smoothed, kps_smoothed, _ = processor_smoothed.process_frame(pose_detection_frame.copy(), show_feedback=False)
            
            # Update comparison
            comparison.update(
                kps_raw[0, 0, :, :],
                kps_smoothed[0, 0, :, :]
            )
            
            # Create side-by-side comparison (for display and/or output)
            h, w = frame_raw.shape[:2]
            
            # Adjust max width based on orientation (only for display)
            if show_live:
                if original_is_portrait:
                    # Portrait videos - each frame is narrower, can fit side by side
                    max_width = 540  # Each frame will be ~540px wide
                else:
                    # Landscape videos
                    max_width = 640
                
                # Resize if too large for display
                if w > max_width:
                    scale = max_width / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_frame_raw = cv2.resize(frame_raw, (new_w, new_h))
                    display_frame_smoothed = cv2.resize(frame_smoothed, (new_w, new_h))
                else:
                    display_frame_raw = frame_raw.copy()
                    display_frame_smoothed = frame_smoothed.copy()
            
            # For output video, use full resolution (no resizing)
            output_frame_raw = frame_raw.copy()
            output_frame_smoothed = frame_smoothed.copy()
            
            # Add labels
            label_scale = 0.7 if original_is_portrait else 1.0
            label_thickness = 1 if original_is_portrait else 2
            label_color = (120, 220, 120)  # Lighter green
            
            # Larger labels for output video
            output_label_scale = 1.8
            output_label_thickness = 4
            
            if show_live:
                cv2.putText(display_frame_raw, "RAW MODEL", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_color, label_thickness)
                cv2.putText(display_frame_smoothed, "SMOOTHED MODEL", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_color, label_thickness)
            
            if output_path:
                cv2.putText(output_frame_raw, "RAW MODEL", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, output_label_scale, label_color, output_label_thickness)
                cv2.putText(output_frame_smoothed, "SMOOTHED MODEL", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, output_label_scale, label_color, output_label_thickness)
            
            # Create combined frames
            if output_path:
                combined_output = np.hstack([output_frame_raw, output_frame_smoothed])
                combined_output = draw_live_metrics_overlay(combined_output, comparison.metrics, frame_count)
                
                # Initialize video writer on first frame
                if out is None:
                    combined_h, combined_w = combined_output.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))
                    if out.isOpened():
                        print(f"Output video initialized: {combined_w}x{combined_h}")
                        print(f"Output will be saved to: {output_path}")
                    else:
                        print("Error: Could not initialize video writer")
                        out = None
                
                # Write to output video
                if out and out.isOpened():
                    out.write(combined_output)
        
        if show_live and not paused:
            # Stack horizontally for display
            combined_display = np.hstack([display_frame_raw, display_frame_smoothed])
            combined_display = draw_live_metrics_overlay(combined_display, comparison.metrics, frame_count)
            
            # Display
            cv2.imshow("Comparison: Raw vs Spatio-Temporal Smoothing", combined_display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopping comparison...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
    
    cap.release()
    if out and out.isOpened():
        out.release()
        print(f"\nVideo saved to: {output_path}")
    if show_live:
        cv2.destroyAllWindows()
    
    # Print summary
    summary = comparison.get_summary()
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\nRAW MODEL (Base MoveNet Lightning):")
    print(f"  Jitter (Acceleration):     {summary['raw']['jitter_mean']:.6f} ± {summary['raw']['jitter_std']:.6f}")
    print(f"  Temporal Consistency:      {summary['raw']['consistency_mean']:.6f}")
    print(f"  Stability (Variance):      {summary['raw']['stability_mean']:.6f}")
    print(f"  Smoothness (Jerk):         {summary['raw']['smoothness_mean']:.6f}")
    print(f"  Responsiveness (Velocity): {summary['raw']['responsiveness_mean']:.6f}")
    print(f"  Average Confidence:        {summary['raw']['confidence_mean']:.3f}")
    
    print("\nSMOOTHED MODEL (Spatio-Temporal Enhanced):")
    print(f"  Jitter (Acceleration):     {summary['smoothed']['jitter_mean']:.6f} ± {summary['smoothed']['jitter_std']:.6f}")
    print(f"  Temporal Consistency:      {summary['smoothed']['consistency_mean']:.6f}")
    print(f"  Stability (Variance):      {summary['smoothed']['stability_mean']:.6f}")
    print(f"  Smoothness (Jerk):         {summary['smoothed']['smoothness_mean']:.6f}")
    print(f"  Responsiveness (Velocity): {summary['smoothed']['responsiveness_mean']:.6f}")
    print(f"  Average Confidence:        {summary['smoothed']['confidence_mean']:.3f}")
    
    print("\nIMPROVEMENTS:")
    print(f"  Jitter Reduction:        {summary['improvements']['jitter_reduction']}")
    print(f"  Consistency Improvement: {summary['improvements']['consistency_improvement']}")
    print(f"  Stability Improvement:   {summary['improvements']['stability_improvement']}")
    print(f"  Smoothness Improvement:  {summary['improvements']['smoothness_improvement']}")
    print(f"  Responsiveness Change:   {summary['improvements']['responsiveness_change']}")
    print("="*60)
    
    print("\nMetric Definitions:")
    print("- Jitter: Mean absolute acceleration (lower = less shaky)")
    print("- Temporal Consistency: Position std deviation over time (lower = more stable)")
    print("- Stability: Position variance (lower = less wandering)")
    print("- Smoothness: Mean absolute jerk/3rd derivative (lower = smoother motion)")
    print("- Responsiveness: Mean velocity magnitude (measures tracking speed)")
    print("- Confidence: Average detection confidence (higher = better)")
    print("="*60)


def get_user_choices():
    """Get user choices for video and output saving (squat mode auto-selected)."""
    import os
    
    # Automatically set to squat mode
    print("\n=== Squat Mode (Auto-selected) ===")
    
    # Choose video from squat directory
    print("\n=== Squat Video Selection ===")
    
    videos = []
    squat_dir = "data/squat"
    if os.path.exists(squat_dir):
        for file in os.listdir(squat_dir):
            if file.endswith('.mp4'):
                videos.append(os.path.join(squat_dir, file))
    
    if not videos:
        print("No squat videos found in data/squat directory!")
        return None, None
    
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
        save_choice = input("Do you want to save the comparison video? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_base = f"output/{base_name}_comparison"
            ext = ".mp4"
            output_path = output_base + ext
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
    
    return video_path, output_path


if __name__ == "__main__":
    # Check if command line arguments are provided (for backward compatibility)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Use command line arguments
        video_path = sys.argv[1]
        show_live = "--no-live" not in sys.argv
        
        # Parse output path
        output_path = None
        if "--output" in sys.argv:
            try:
                output_idx = sys.argv.index("--output")
                if output_idx + 1 < len(sys.argv):
                    output_path = sys.argv[output_idx + 1]
                    # Create output directory if it doesn't exist
                    import os
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Auto-increment filename if it exists
                    if os.path.exists(output_path):
                        base, ext = os.path.splitext(output_path)
                        idx = 1
                        while os.path.exists(f"{base}_{idx}{ext}"):
                            idx += 1
                        output_path = f"{base}_{idx}{ext}"
            except (ValueError, IndexError):
                print("Error: --output requires a file path")
                sys.exit(1)
    else:
        # Interactive mode
        result = get_user_choices()
        if result[0] is None:  # No videos found
            sys.exit(1)
        
        video_path, output_path = result
        show_live = True  # Default to showing live display in interactive mode
    
    print(f"\n=== Configuration ===")
    print(f"Video: {video_path}")
    print(f"Mode: Squat (Auto-selected)")
    print(f"Display: {'Live view enabled' if show_live else 'No live view'}")
    print(f"Output: {output_path if output_path else 'Display only'}")
    print("")
    
    compare_on_video(video_path, show_live=show_live, output_path=output_path)
