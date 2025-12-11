import cv2
import numpy as np
from PIL import Image, ImageDraw
from helpers.analyzers.squat_analyzer import SquatFormAnalyzer
from helpers.analyzers.bench_analyzer import BenchPressFormAnalyzer


def calculate_ui_scale(height, width):
    """Calculate scale factor for UI elements based on video resolution.
    
    Uses 1080x1920 (portrait) as the baseline reference resolution.
    Returns a scale factor for text sizes, padding, margins, etc.
    """
    # Reference resolution (1080 width x 1920 height for portrait videos)
    reference_width = 1080
    reference_height = 1920
    
    # Calculate scale based on the smaller dimension
    if height > width:
        # Portrait orientation
        scale = width / reference_width
    else:
        # Landscape orientation - use height as reference
        scale = height / reference_height
    
    return max(0.5, min(scale, 3.0))  # Clamp between 0.5x and 3x


class PoseFeedback:
    """Handles real-time feedback for pose detection positioning and exercise form."""
    
    def __init__(self):
        self.stable_frames = 0
        self.last_person_center = None
        self.last_person_size = None
        self.feedback_history = []
        self.last_distance_feedback = None
        self.frames_since_movement = 0
        self.movement_threshold = 0.05  # Reduced sensitivity
        self.stability_frames_needed = 15  # Need to hold still for 15 frames (0.5 seconds at 30fps)
        self.feedback_persistence_frames = 45  # Keep feedback for 1.5 seconds (45 frames at 30fps)

        # For persistent actionable feedback
        self.last_actionable_feedback = []
        self.last_actionable_frame = 0

        # Initialize exercise analyzers
        self.squat_analyzer = SquatFormAnalyzer()
        self.bench_analyzer = BenchPressFormAnalyzer()
        try:
            from helpers.analyzers.deadlift_analyzer import DeadliftFormAnalyzer
            self.deadlift_analyzer = DeadliftFormAnalyzer()
        except Exception:
            self.deadlift_analyzer = None
        self.form_analysis_enabled = True

        # Exercise mode settings
        self.exercise_mode = "squat"  # Default to squat mode
    
    def set_exercise_mode(self, mode):
        """Set the exercise mode for analysis."""
        if mode in ["squat", "bench", "deadlift"]:
            self.exercise_mode = mode
            print(f"Exercise mode set to: {mode.upper()}")
            if mode == "deadlift" and (not hasattr(self, "deadlift_analyzer") or self.deadlift_analyzer is None):
                try:
                    from .deadlift_analyzer import DeadliftFormAnalyzer
                    self.deadlift_analyzer = DeadliftFormAnalyzer()
                except Exception:
                    self.deadlift_analyzer = None
        else:
            print(f"Unknown exercise mode: {mode}. Using default: squat")
            self.exercise_mode = "squat"
    
    def calculate_person_center_and_size(self, keypoints_with_scores, image_height, image_width):
        """Calculate the center and size of the person more reliably."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # Use only core body keypoints for more stable detection
        core_keypoints = [5, 6, 11, 12]  # Left/right shoulders and hips
        valid_core_points = []
        
        for i in core_keypoints:
            if keypoints[i, 2] > 0.2:  # Higher threshold for core points
                x = keypoints[i, 1] * image_width
                y = keypoints[i, 0] * image_height
                valid_core_points.append((x, y))
        
        if len(valid_core_points) < 2:
            return None, None, 0, 0
        
        # Calculate center from core body points
        x_coords = [p[0] for p in valid_core_points]
        y_coords = [p[1] for p in valid_core_points]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Calculate size based on core body spread
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Add some padding to account for full body
        estimated_width = width * 1.5
        estimated_height = height * 2.5  # Account for head and legs
        
        return (center_x, center_y), (estimated_width, estimated_height), len(valid_core_points), len(core_keypoints)
    
    def detect_distance_movement(self, current_center, current_size):
        """Detect if the person is moving closer or farther (not just any movement)."""
        if self.last_person_center is None or self.last_person_size is None:
            self.last_person_center = current_center
            self.last_person_size = current_size
            return False, "initializing"
        
        # Calculate change in center position (sideways movement)
        center_change = np.sqrt(
            (current_center[0] - self.last_person_center[0])**2 + 
            (current_center[1] - self.last_person_center[1])**2
        )
        
        # Calculate change in size (indicates distance change)
        size_change = abs(current_size[0] - self.last_person_size[0]) / self.last_person_size[0]
        
        # Update last values
        self.last_person_center = current_center
        self.last_person_size = current_size
        
        # Only consider it movement if there's significant size change (distance change)
        # Ignore small movements and arm movements
        if size_change > 0.1:  # 10% change in size indicates distance movement
            self.frames_since_movement = 0
            return True, "distance_changing"
        elif center_change > self.movement_threshold:
            # Sideways movement, don't reset distance feedback
            return False, "sideways_movement"
        else:
            # No significant movement
            self.frames_since_movement += 1
            return False, "stable"
    
    def get_comprehensive_feedback(self, keypoints_with_scores, image_height, image_width, frame_idx=None):
        """Get comprehensive feedback including positioning and squat form."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # Calculate person center and size
        center, size, valid_core, total_core = self.calculate_person_center_and_size(
            keypoints_with_scores, image_height, image_width
        )
        
        if center is None or valid_core < 2:
            return {
                'message': "Person not clearly detected",
                'color': (128, 128, 128),
                'distance_status': 'not_detected',
                'recommendation': 'Move into frame and face camera',
                'person_percentage': 0,
                'visibility_percentage': 0,
                'is_stable': False,
                'form_analysis': None
            }
        
        # Calculate person area percentage
        person_area = size[0] * size[1]
        image_area = image_height * image_width
        person_percentage = (person_area / image_area) * 100
        
        # Count visible keypoints
        visible_keypoints = sum(1 for i in range(17) if keypoints[i, 2] > 0.15)
        visibility_percentage = (visible_keypoints / 17) * 100
        
        # --- DISABLE DISTANCE FEEDBACK ---
        # is_moving, movement_type = self.detect_distance_movement(center, size)
        # if movement_type == "initializing": ...
        # if movement_type == "distance_changing": ...
        # if movement_type == "sideways_movement": ...
        # if movement_type == "stable": ...
        # Instead, always show form feedback below:
        
        # Get exercise form analysis if enabled
        form_analysis = None
        if self.form_analysis_enabled and visibility_percentage > 70:  # Only analyze if good visibility
            if self.exercise_mode == "squat":
                form_analysis = self.squat_analyzer.analyze_squat_form(keypoints_with_scores, image_height, image_width)
            elif self.exercise_mode == "bench":
                form_analysis = self.bench_analyzer.analyze_bench_press_form(keypoints_with_scores, image_height, image_width)
            elif self.exercise_mode == "deadlift" and hasattr(self, "deadlift_analyzer") and self.deadlift_analyzer is not None:
                form_analysis = self.deadlift_analyzer.analyze_deadlift_form(keypoints_with_scores, image_height, image_width)
        
        feedback = {
            'message': '',
            'color': (90, 170, 90),
            'distance_status': None,
            'recommendation': '',
            'person_percentage': person_percentage,
            'visibility_percentage': visibility_percentage,
            'is_stable': True,
            'form_analysis': form_analysis,
            'exercise_mode': self.exercise_mode
        }
        
        # Only show form feedback
        if form_analysis:
            if self.exercise_mode == "squat":
                if form_analysis.get('depth_message'):
                    feedback['message'] = ''
                    if form_analysis.get('depth_status') == 'good':
                        feedback['color'] = (90, 170, 90)
                    elif form_analysis.get('depth_status') == 'needs_improvement':
                        feedback['color'] = (60, 160, 255)
                    else:
                        feedback['color'] = (220, 220, 220)
                if form_analysis.get('recommendations'):
                    feedback['recommendation'] = form_analysis['recommendations'][0]
            elif self.exercise_mode == "bench":
                # Use unified phase pill; do not set main message/color from phase here
                pass
            elif self.exercise_mode == "deadlift":
                # Use unified phase pill; keep actionable feedback below
                # Show persistent actionable feedback for deadlift
                actionable = form_analysis.get('recommendations', [])
                if frame_idx is not None:
                    if actionable and actionable != self.last_actionable_feedback:
                        self.last_actionable_feedback = actionable
                        self.last_actionable_frame = frame_idx
                    elif (frame_idx - self.last_actionable_frame) < self.feedback_persistence_frames:
                        actionable = self.last_actionable_feedback
                    else:
                        self.last_actionable_feedback = []
                        actionable = []
                if actionable:
                    feedback['recommendation'] = actionable[0]
                else:
                    feedback['recommendation'] = ''
        else:
            feedback['message'] = "Hold still for form feedback"
            feedback['color'] = (60, 160, 255)
            feedback['recommendation'] = "Stay in place for form feedback"
        
        return feedback

def draw_rounded_rectangle(image, x1, y1, x2, y2, color, radius=20, thickness=-1):
    """Draw a rounded rectangle on the image."""
    # Create a mask for the rounded rectangle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw the rounded rectangle on the mask
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
    
    # Draw the corner circles
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)
    
    # Apply the mask to the image
    image[mask == 255] = color

def draw_rounded_rectangle_simple(image, x1, y1, x2, y2, color, radius=15):
    """Draw a simple rounded rectangle."""
    # Draw the main rectangle
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    # Draw the corner circles
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)

def draw_rounded_rectangle_fast(image, x1, y1, x2, y2, color, radius=10):
    """Draw a rounded rectangle with solid color - maximum performance."""
    # Draw the main rectangle
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    # Draw the corner circles
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)

def draw_rounded_rectangle_with_alpha(image, x1, y1, x2, y2, color, alpha=0.7, radius=10):
    """Draw a rounded rectangle with alpha transparency using PIL for better performance."""
    # Convert OpenCV image to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw rounded rectangle with alpha
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=(color[0], color[1], color[2], int(255 * alpha)))
    
    # Composite the overlay onto the image
    result = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)

def draw_comprehensive_feedback_overlay(image, feedback):
    """Draw comprehensive feedback on the image with mobile-optimized layout."""
    height, width, _ = image.shape
    
    # Calculate UI scale factor based on video resolution
    ui_scale = calculate_ui_scale(height, width)

    # Calculate adaptive background height based on content - increased for larger text
    base_height = int(100 * ui_scale)  # Base height accommodates title area
    extra_height = 0

    # Add extra height for phase pill if form_analysis has a phase (all modes)
    show_reps = False
    rep_text = ""
    phase_text = ""
    phase_bg_color = None
    phase_text_color = (255, 255, 255)
    if feedback.get('form_analysis'):
        fa = feedback['form_analysis']
        if 'phase' in fa and fa.get('phase'):
            exercise_mode = feedback.get('exercise_mode', 'squat')
            current_phase = str(fa.get('phase', '')).lower()
            phase_text = f"Phase: {current_phase.capitalize()}"
            # Natural, muted palette per phase (BGR)
            common_palette = {
                'ascending': (90, 170, 90),     # muted green
                'pressing': (90, 170, 90),      # muted green
                'descending': (60, 160, 255),   # soft amber/orange
                'lowering': (60, 160, 255),     # soft amber/orange
                'bottom': (60, 220, 255),       # soft yellow
                'pause': (180, 180, 180),       # light gray
                'rest': (200, 200, 200),        # light gray
                'standing': (200, 200, 200),    # light gray
                'stable': (200, 200, 200),      # light gray
            }
            # Use the common palette for all modes
            phase_bg_color = common_palette.get(current_phase, (180, 180, 180))
            # Choose readable text color based on brightness
            phase_text_color = (0, 0, 0) if sum(phase_bg_color) > 500 else (255, 255, 255)
            extra_height += int(28 * ui_scale)
        # Rep counters (if present in form analysis for squat/deadlift)
        if any(k in fa for k in ['rep_count', 'correct_rep_count', 'incorrect_rep_count']):
            show_reps = True
            rep_text = f"Reps: {fa.get('rep_count',0)}  Correct: {fa.get('correct_rep_count',0)}  Incorrect: {fa.get('incorrect_rep_count',0)}"
            # Don't add extra height here, we'll move it to the top

    # Add extra height dynamically based on number of status lines (back/depth) and recommendation lines
    status_lines = 0
    if feedback.get('form_analysis'):
        form_analysis = feedback['form_analysis']
        # Deadlift: show hip extension cue if present
        if feedback.get('exercise_mode') == 'deadlift' and form_analysis.get('hip_extension_message'):
            status_lines += 1
            extra_height += int((24 + 10) * ui_scale)
        else:
            if form_analysis.get('back_message') and form_analysis.get('back_status') is not None:
                status_lines += 1
            if form_analysis.get('depth_message') and form_analysis.get('depth_status') is not None:
                status_lines += 1
            if status_lines > 1:
                extra_height += int((status_lines - 1) * 26 * ui_scale)
            if form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0:
                recommendation_text = form_analysis['recommendations'][0]
                max_chars_per_line = 65
                if len(recommendation_text) > max_chars_per_line:
                    estimated_lines = max(1, len(recommendation_text) // max_chars_per_line + 1)
                    extra_height += int((estimated_lines * 24 + 10) * ui_scale)
                else:
                    extra_height += int((24 + 10) * ui_scale)
            if status_lines > 0 or (form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0):
                pill_font = cv2.FONT_HERSHEY_SIMPLEX
                pill_scale = 0.9 * ui_scale
                pill_thickness = max(1, int(2 * ui_scale))
                label = 'CORRECT FORM' if (form_analysis.get('depth_status') == 'good' and form_analysis.get('back_status') == 'good') else 'INCORRECT FORM'
                text_size = cv2.getTextSize(label, pill_font, pill_scale, pill_thickness)[0]
                vertical_padding = int(8 * ui_scale)
                pill_height = text_size[1] + vertical_padding * 2
                extra_height += int((pill_height + 16) * ui_scale)

    # Bench mode: allocate extra space for form feedback issues
    if feedback.get('exercise_mode') == 'bench' and feedback.get('form_analysis'):
        issues_list = feedback['form_analysis'].get('issues', [])
        if issues_list:
            # Each issue can have message + recommendation which may wrap to multiple lines
            # Estimate based on number of issues and average line length
            max_chars_per_line = 50  # Bench feedback tends to be longer
            for issue in issues_list:
                message = issue.get('message', '')
                recommendation = issue.get('recommendation', '')
                full_text = f"{message} - {recommendation}" if recommendation else message
                estimated_lines = max(1, len(full_text) // max_chars_per_line + 1)
                extra_height += int((estimated_lines * 26 + 8) * ui_scale)  # 26px per line + 8px spacing
        else:
            # "Form looks good!" message
            extra_height += int(30 * ui_scale)
    
    text_bg_height = base_height + extra_height
    
    # Ensure we don't exceed image bounds and leave some margin
    max_height = min(height * 0.35, int(220 * ui_scale))  # Increased max height for bench press (was 0.3, 180)
    # For deadlift, keep a fixed overlay height to prevent jitter when messages appear/disappear
    if feedback.get('exercise_mode') == 'deadlift':
        text_bg_height = min(int(160 * ui_scale), int(max_height))
    elif feedback.get('exercise_mode') == 'bench':
        # For bench, allow more height for form feedback
        text_bg_height = min(int(text_bg_height), int(max_height))
    elif text_bg_height > max_height:
        text_bg_height = int(max_height)
    
    # Position overlay with margin from bottom and sides
    margin = int(5 * ui_scale)  # Base bottom margin
    side_margin = 0
    bottom_offset = min(int(height * 0.06), int(64 * ui_scale))
    bg_y_start = max(0, height - text_bg_height - margin - bottom_offset)

    # Ensure valid rectangle coordinates
    x1 = side_margin
    x2 = width - side_margin
    y1 = bg_y_start
    y2 = min(height - margin - bottom_offset + (height - (bg_y_start + text_bg_height + margin)), height - margin)

    # Draw rep counter as a full-width top bar with three columns (Total Reps, Correct, Incorrect) only if rep counting is active
    # Get rep counts from form_analysis for all exercise modes
    rep_count = 0
    correct_count = 0
    incorrect_count = 0
    # Show rep counter for squat, deadlift, and bench modes
    if feedback.get('exercise_mode') in ['squat', 'deadlift', 'bench'] and feedback.get('form_analysis'):
        fa = feedback['form_analysis']
        rep_count = fa.get('rep_count', 0)
        correct_count = fa.get('correct_rep_count', 0)
        incorrect_count = fa.get('incorrect_rep_count', 0)
    else:
        rep_count = feedback.get('rep_count', 0)
        correct_count = feedback.get('correct_rep_count', 0)
        incorrect_count = feedback.get('incorrect_rep_count', 0)
    # Always show rep counter for all three exercise modes
    rep_counter_active = feedback.get('exercise_mode') in ['squat', 'deadlift', 'bench']
    if rep_counter_active:
        # Bar background color (dark blue-ish)
        bar_color = (80, 86, 106)
        bar_alpha = 0.95
        bar_height = int(120 * ui_scale)
        bar_y1 = 0
        bar_y2 = bar_y1 + bar_height
        # Draw the full-width bar
        image = draw_rounded_rectangle_with_alpha(image, 0, bar_y1, width, bar_y2, bar_color, alpha=bar_alpha, radius=0)

        # Column positions
        col_width = width // 3
        col_x = [0, col_width, col_width * 2, width]
        # Colors for text
        correct_color = (100, 139, 51)
        incorrect_color = (75, 74, 179)
        reps_label_color = (200, 180, 180)
        # Labels
        labels = ["REPS", "CORRECT", "INCORRECT"]
        values = [rep_count, correct_count, incorrect_count]
        label_colors = [reps_label_color, correct_color, incorrect_color]

        # Draw divider lines between columns
        divider_thickness = max(1, int(4 * ui_scale))
        for i in [1, 2]:
            x = col_x[i]
            cv2.line(image, (x, int(bar_y1 + 12 * ui_scale)), (x, int(bar_y2 - 12 * ui_scale)), (120, 120, 120), divider_thickness)

        # Draw each column: label (top, centered), value (below, centered)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = 0.9 * ui_scale
        label_thickness = max(1, int(4 * ui_scale))
        value_font_scale = 1.2 * ui_scale
        value_thickness = max(1, int(6 * ui_scale))
        for i in range(3):
            # Center of this column
            center_x = (col_x[i] + col_x[i+1]) // 2
            # Draw label (allow for two lines for TOTAL REPS)
            label = labels[i]
            label_lines = label.split("\n")
            # Draw each line of label
            for j, line in enumerate(label_lines):
                label_size = cv2.getTextSize(line, font, label_font_scale, label_thickness)[0]
                label_x = center_x - label_size[0] // 2
                label_y = int(bar_y1 + 42 * ui_scale + j * (label_size[1] + 2))
                cv2.putText(image, line, (label_x, label_y), font, label_font_scale, label_colors[i], label_thickness, cv2.LINE_AA)
            # Draw value (below label, centered)
            value_str = str(values[i])
            value_size = cv2.getTextSize(value_str, font, value_font_scale, value_thickness)[0]
            value_x = center_x - value_size[0] // 2
            value_y = int(bar_y1 + 78 * ui_scale + value_size[1] // 2)
            cv2.putText(image, value_str, (value_x, value_y), font, value_font_scale, (255, 255, 255), value_thickness, cv2.LINE_AA)
    
    # Only draw if we have a valid rectangle
    if x1 < x2 and y1 < y2:
        # Draw rounded rectangle background with alpha transparency
        image = draw_rounded_rectangle_with_alpha(image, x1, y1, x2, y2, (80, 86, 106), alpha=0.7, radius=int(10 * ui_scale))
        

        # Draw rep counter if enabled (at the top of overlay)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.95 * ui_scale
        thickness = max(1, int(2 * ui_scale))
        y_offset = int(20 * ui_scale)
        # Draw phase only as a feedback line with background (handled below)
        # Remove duplicate phase display here

    # Draw main feedback message
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Only draw if we have a valid rectangle
    if x1 < x2 and y1 < y2:
        # Draw rounded rectangle background with alpha transparency
        image = draw_rounded_rectangle_with_alpha(image, x1, y1, x2, y2, (80, 86, 106), alpha=0.7, radius=int(10 * ui_scale))

        # Prepare all feedback lines to display
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_main = 1.1 * ui_scale  # Increased font size for main message
        font_scale_secondary = 1.1 * ui_scale  # For secondary lines
        font_scale_small = 0.95 * ui_scale
        thickness_main = max(1, int(2 * ui_scale))
        thickness_secondary = max(1, int(2 * ui_scale))
        thickness_small = max(1, int(2 * ui_scale))
    y_offset = int(36 * ui_scale)
    line_spacing = int(18 * ui_scale)
    block_spacing = int(34 * ui_scale)  # Increased gap between feedback lines
    lines = []
    bg_colors = []
    text_colors = []

    # Phase line (if present)
    if phase_text:
        lines.append(phase_text)
        if phase_bg_color is None:
            # Fallback if not set: use a neutral dark background
            bg_colors.append((40, 40, 40))
            text_colors.append((255, 255, 255))
        else:
            bg_colors.append(phase_bg_color)
            text_colors.append(phase_text_color)
    # Main feedback message
    if feedback['message']:
        lines.append(feedback['message'])
        bg_colors.append(feedback['color'])
        text_colors.append((0,0,0) if sum(feedback['color']) > 500 else (255,255,255))

    # Form analysis lines
    if feedback.get('form_analysis'):
        form_analysis = feedback['form_analysis']
        # For squat and deadlift, only show the referenced feedback (back_message/depth_message), not the recommendations list
        def wrap_text(text, font, scale, thickness, max_width):
            words = text.split()
            lines = []
            current_line = ''
            for word in words:
                test_line = f'{current_line} {word}'.strip()
                size = cv2.getTextSize(test_line, font, scale, thickness)[0]
                if size[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            return lines

        max_text_width = x2 - x1 - 64  # leave some margin inside the box
        if feedback.get('exercise_mode') == 'squat':
            # Helper: map severity string to BGR color
            def _severity_to_color(sev):
                # low -> soft yellow, medium -> soft amber, high -> muted red
                if sev == 'low':
                    return (60, 220, 255)
                if sev == 'medium':
                    return (60, 160, 255)
                if sev == 'high':
                    return (0, 0, 200)
                # default/fallback
                return (220, 220, 220)

            # Helper: pick highest severity from a set of issue types
            def _highest_severity(issues, types):
                rank = {'low': 1, 'medium': 2, 'high': 3}
                best = None
                best_r = 0
                for it in issues or []:
                    if it.get('type') in types:
                        sev = it.get('severity')
                        r = rank.get(sev, 0)
                        if r > best_r:
                            best_r = r
                            best = sev
                return best

            issues_list = form_analysis.get('issues', [])

            # Back message with severity-based color (fallback to status if no specific issue found)
            if form_analysis.get('back_message') and form_analysis.get('back_status'):
                back_sev = _highest_severity(issues_list, {'back_rounding', 'forward_lean_top', 'top_posture'})
                if back_sev:
                    back_color = _severity_to_color(back_sev)
                else:
                    back_color = (90, 170, 90) if form_analysis['back_status'] == 'good' else (60, 160, 255)
                wrapped = wrap_text(form_analysis['back_message'], font, font_scale_secondary, thickness_secondary, max_text_width)
                for wline in wrapped:
                    lines.append(wline)
                    bg_colors.append((40,40,40))
                    text_colors.append(back_color)

            # Depth message with severity-based color (fallback to status if no depth issue found)
            if form_analysis.get('depth_message') and form_analysis.get('depth_status'):
                depth_sev = _highest_severity(issues_list, {'insufficient_depth'})
                if depth_sev:
                    depth_color = _severity_to_color(depth_sev)
                else:
                    if form_analysis['depth_status'] == 'good':
                        depth_color = (90, 170, 90)  # Green for good depth
                    elif form_analysis['depth_status'] == 'needs_improvement':
                        depth_color = (60, 160, 255)  # Orange for form issue
                    elif form_analysis['depth_status'] == 'guidance':
                        depth_color = (180, 200, 255)  # Light blue for guidance (not a form issue)
                    else:
                        depth_color = (220, 220, 220)
                wrapped = wrap_text(form_analysis['depth_message'], font, font_scale_secondary, thickness_secondary, max_text_width)
                for wline in wrapped:
                    lines.append(wline)
                    bg_colors.append((40,40,40))
                    text_colors.append(depth_color)
            
            # If no back or depth messages shown, display positive recommendations if available
            if not form_analysis.get('back_message') and not form_analysis.get('depth_message'):
                recs = form_analysis.get('recommendations', [])
                for rec in recs:
                    # Check if it's a positive recommendation
                    if any(word in rec.lower() for word in ['good', 'great', 'strong', 'controlled', 'ready']):
                        rec_color = (90, 170, 90)  # muted green for positive feedback
                    else:
                        rec_color = (60, 160, 255)  # amber for improvement
                    wrapped = wrap_text(rec, font, font_scale_secondary, thickness_secondary, max_text_width)
                    for wline in wrapped:
                        lines.append(wline)
                        bg_colors.append((40,40,40))
                        text_colors.append(rec_color)
        elif feedback.get('exercise_mode') == 'deadlift':
            # Helper: map severity string to BGR color
            def _severity_to_color(sev):
                # good -> muted green, low -> soft yellow, medium -> soft amber, high -> muted red
                if sev == 'good':
                    return (90, 170, 90)  # muted green for positive feedback
                if sev == 'low':
                    return (60, 220, 255)
                if sev == 'medium':
                    return (60, 160, 255)
                if sev == 'high':
                    return (0, 0, 200)
                # default/fallback
                return (220, 220, 220)

            # If hip extension is explicitly good, show a green status line first
            if form_analysis.get('hip_extension_message') and form_analysis.get('hip_extension_status') == 'good':
                wrapped = wrap_text(form_analysis['hip_extension_message'], font, font_scale_secondary, thickness_secondary, max_text_width)
                for wline in wrapped:
                    lines.append(wline)
                    bg_colors.append((40,40,40))
                    text_colors.append((90, 170, 90))  # muted green

            # Prefer structured recommendations with severity
            recs_detailed = form_analysis.get('recommendations_detailed')
            if recs_detailed:
                for entry in recs_detailed:
                    rec_text = entry.get('text') or ''
                    sev = entry.get('severity', 'low')
                    rec_color = _severity_to_color(sev)
                    wrapped = wrap_text(rec_text, font, font_scale_secondary, thickness_secondary, max_text_width)
                    for wline in wrapped:
                        lines.append(wline)
                        bg_colors.append((40,40,40))
                        text_colors.append(rec_color)
            else:
                # Fallback to existing recommendations list and issue matching
                recs = form_analysis.get('recommendations', [])
                all_issues = (form_analysis.get('hip_extension_issues', []) +
                              form_analysis.get('knee_issues', []) + 
                              form_analysis.get('spine_issues', []) + 
                              form_analysis.get('balance_issues', []))
                for rec in recs:
                    rec_color = (220, 220, 220)  # default light gray
                    for issue in all_issues:
                        if issue.get('recommendation') == rec:
                            severity = issue.get('severity', 'low')
                            rec_color = _severity_to_color(severity)
                            break
                    wrapped = wrap_text(rec, font, font_scale_secondary, thickness_secondary, max_text_width)
                    for wline in wrapped:
                        lines.append(wline)
                        bg_colors.append((40,40,40))
                        text_colors.append(rec_color)
        elif feedback.get('exercise_mode') == 'bench':
            # Bench press form feedback with severity-based colors
            def _severity_to_color(sev):
                # low -> soft yellow, medium -> soft amber, high -> muted red
                if sev == 'low':
                    return (60, 220, 255)
                if sev == 'medium':
                    return (60, 160, 255)
                if sev == 'high':
                    return (0, 0, 200)
                # default/fallback
                return (220, 220, 220)
            
            # Get issues from form analysis
            issues_list = form_analysis.get('issues', [])
            
            # Display each issue with its severity-based color
            for issue in issues_list:
                message = issue.get('message', '')
                recommendation = issue.get('recommendation', '')
                severity = issue.get('severity', 'low')
                issue_color = _severity_to_color(severity)
                
                # Display message and recommendation together
                full_text = f"{message} - {recommendation}" if recommendation else message
                wrapped = wrap_text(full_text, font, font_scale_secondary, thickness_secondary, max_text_width)
                for wline in wrapped:
                    lines.append(wline)
                    bg_colors.append((40,40,40))
                    text_colors.append(issue_color)
            
            # If no issues, show a positive feedback message based on phase
            if not issues_list:
                phase = form_analysis.get('phase', '').lower()
                if phase == 'rest':
                    positive_msg = "Form looks good!"
                elif phase == 'lowering':
                    positive_msg = "Controlled descent!"
                elif phase == 'pause':
                    positive_msg = "Good pause at chest!"
                elif phase == 'pressing':
                    positive_msg = "Strong press!"
                else:
                    positive_msg = "Form looks good!"
                lines.append(positive_msg)
                bg_colors.append((40,40,40))
                text_colors.append((90, 170, 90))  # muted green
        else:
            # For other exercises, keep existing logic if needed
            pass

    # Optionally cap number of lines for deadlift to keep inside fixed box
    if feedback.get('exercise_mode') == 'deadlift':
        max_deadlift_lines = 3  # Phase + 2 lines of feedback
        if len(lines) > max_deadlift_lines:
            lines = lines[:max_deadlift_lines]
            bg_colors = bg_colors[:max_deadlift_lines]
            text_colors = text_colors[:max_deadlift_lines]

    # Draw each feedback line with its own background, evenly spaced
    current_y = bg_y_start + y_offset
    for i, (text, bg_color, text_color) in enumerate(zip(lines, bg_colors, text_colors)):
        # Choose font size for each line
        if i == 0 and phase_text:
            scale = font_scale_secondary
            thick = thickness_secondary
        elif i == 0 or (i == 1 and phase_text):
            scale = font_scale_main
            thick = thickness_main
        else:
            scale = font_scale_secondary
            thick = thickness_secondary
        text_size = cv2.getTextSize(text, font, scale, thick)[0]
        pad_x = int(32 * ui_scale)
        pad_y = int(16 * ui_scale)  # Slightly more vertical padding
        rect_x1 = int(x1 + 16 * ui_scale)
        rect_x2 = int(x2 - 16 * ui_scale)
        rect_y1 = current_y - text_size[1] - pad_y//2
        rect_y2 = int(current_y + pad_y + 4 * ui_scale)
        # Draw background for this line
        image = draw_rounded_rectangle_with_alpha(image, rect_x1, rect_y1, rect_x2, rect_y2, bg_color, alpha=0.85, radius=int(12 * ui_scale))
        # Draw text centered
        text_x = (rect_x1 + rect_x2 - text_size[0]) // 2
        text_y = current_y + pad_y//2
        cv2.putText(image, text, (text_x, text_y), font, scale, text_color, thick)
        current_y = rect_y2 + block_spacing

        # (Removed: correct/incorrect form indicator at the bottom)

    return image