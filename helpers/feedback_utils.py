import cv2
import numpy as np
from PIL import Image, ImageDraw
from helpers.analyzers.squat_analyzer import SquatFormAnalyzer
from helpers.analyzers.bench_analyzer import BenchPressFormAnalyzer

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
            'color': (0, 255, 0),
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
                        feedback['color'] = (0, 255, 0)
                    elif form_analysis.get('depth_status') == 'needs_improvement':
                        feedback['color'] = (0, 165, 255)
                    else:
                        feedback['color'] = (255, 255, 255)
                if form_analysis.get('recommendations'):
                    feedback['recommendation'] = form_analysis['recommendations'][0]
            elif self.exercise_mode == "bench":
                phase = form_analysis.get('phase')
                if phase:
                    phase_title = phase.capitalize()
                    feedback['message'] = f"Bench: {phase_title}"
                    color_map = {
                        'pressing': (0, 255, 0),
                        'lowering': (0, 165, 255),
                        'bottom': (0, 255, 255),
                        'rest': (255, 255, 255),
                        'pause': (255, 255, 255),
                        'stable': (255, 255, 255)
                    }
                    feedback['color'] = color_map.get(phase, (255, 255, 255))
            elif self.exercise_mode == "deadlift":
                phase = form_analysis.get('phase')
                if phase:
                    phase_title = phase.capitalize()
                    feedback['message'] = f"Deadlift: {phase_title}"
                    color_map = {
                        'ascending': (0, 255, 0),
                        'lowering': (0, 165, 255),
                        'bottom': (0, 255, 255),
                        'standing': (255, 255, 255)
                    }
                    feedback['color'] = color_map.get(phase, (255, 255, 255))
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
            feedback['color'] = (0, 165, 255)
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
    
    # Calculate adaptive background height based on content - increased for larger text
    base_height = 100  # Base height accommodates title area
    extra_height = 0

    # Add extra height dynamically based on number of status lines (back/depth) and recommendation lines
    status_lines = 0
    if feedback.get('form_analysis'):
        form_analysis = feedback['form_analysis']
        # Deadlift: show hip extension cue if present
        if feedback.get('exercise_mode') == 'deadlift' and form_analysis.get('hip_extension_message'):
            status_lines += 1
            extra_height += 24 + 10
        else:
            if form_analysis.get('back_message') and form_analysis.get('back_status') is not None:
                status_lines += 1
            if form_analysis.get('depth_message') and form_analysis.get('depth_status') is not None:
                status_lines += 1
            if status_lines > 1:
                extra_height += (status_lines - 1) * 26
            if form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0:
                recommendation_text = form_analysis['recommendations'][0]
                max_chars_per_line = 65
                if len(recommendation_text) > max_chars_per_line:
                    estimated_lines = max(1, len(recommendation_text) // max_chars_per_line + 1)
                    extra_height += (estimated_lines * 24) + 10
                else:
                    extra_height += 24 + 10
            if status_lines > 0 or (form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0):
                pill_font = cv2.FONT_HERSHEY_SIMPLEX
                pill_scale = 0.9
                pill_thickness = 2
                label = 'GOOD FORM' if (form_analysis.get('depth_status') == 'good' and form_analysis.get('back_status') == 'good') else 'INCORRECT FORM'
                text_size = cv2.getTextSize(label, pill_font, pill_scale, pill_thickness)[0]
                vertical_padding = 8
                pill_height = text_size[1] + vertical_padding * 2
                extra_height += pill_height + 16

    # Bench mode: reserve space for phase and depth lines
    if feedback.get('exercise_mode') == 'bench' and feedback.get('form_analysis'):
        extra_height += 26 * 2  # Phase + Depth lines
    
    text_bg_height = base_height + extra_height
    
    # Ensure we don't exceed image bounds and leave some margin
    max_height = min(height * 0.3, 180)  # Increased max height for larger text
    if text_bg_height > max_height:
        text_bg_height = int(max_height)
    
    # Position overlay with margin from bottom and sides
    margin = 5  # Base bottom margin
    side_margin = min(200, width // 10)  # Much smaller side margins for wider feedback background
    # Raise the whole overlay a bit from the bottom so it doesn't sit on top of content
    bottom_offset = min(int(height * 0.06), 64)  # ~6% of height, capped
    bg_y_start = max(0, height - text_bg_height - margin - bottom_offset)
    
    # Ensure valid rectangle coordinates
    x1 = side_margin
    x2 = width - side_margin
    y1 = bg_y_start
    y2 = min(height - margin - bottom_offset + (height - (bg_y_start + text_bg_height + margin)), height - margin)
    
    # Only draw if we have a valid rectangle
    if x1 < x2 and y1 < y2:
        # Draw rounded rectangle background with alpha transparency
        image = draw_rounded_rectangle_with_alpha(image, x1, y1, x2, y2, (0, 0, 0), alpha=0.7, radius=10)
        
        # Draw main feedback message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1  # Increased font size for better visibility
        thickness = 3
        
        # Main message - adjust for narrower width
        text_size = cv2.getTextSize(feedback['message'], font, font_scale, thickness)[0]
        text_x = ((width - side_margin * 2) - text_size[0]) // 2 + side_margin
        text_y = bg_y_start + 20
        cv2.putText(image, feedback['message'], (text_x, text_y), font, font_scale, feedback['color'], thickness)
        
        # Draw form analysis if available (compact)
        if feedback.get('form_analysis'):
            form_analysis = feedback['form_analysis']
            if feedback.get('exercise_mode') == 'deadlift':
                # Collect all actionable recommendations for deadlift (hip, knee, spine)
                recs = []
                # Hip extension
                if form_analysis.get('hip_extension_status') == 'needs_improvement' and form_analysis.get('hip_extension_message'):
                    recs.append((form_analysis['hip_extension_message'], (0, 165, 255)))
                # Knee issues
                for issue in form_analysis.get('knee_issues', []):
                    if 'recommendation' in issue:
                        color = (0, 165, 255) if issue['severity'] in ('medium', 'high') else (0, 255, 0)
                        recs.append((issue['recommendation'], color))
                # Spine/torso issues
                for issue in form_analysis.get('spine_issues', []):
                    if 'recommendation' in issue:
                        color = (0, 165, 255) if issue['severity'] in ('medium', 'high') else (0, 255, 0)
                        recs.append((issue['recommendation'], color))
                # Draw each recommendation line
                for i, (text, color) in enumerate(recs):
                    if len(text) > 60:
                        text = text[:57] + "..."
                    size = cv2.getTextSize(text, font, 0.9, 2)[0]
                    x = ((width - side_margin * 2) - size[0]) // 2 + side_margin
                    y = bg_y_start + 45 + i * 26
                    cv2.putText(image, text, (x, y), font, 0.9, color, 2)
            else:
                # Show both back and depth messages, back first if present (squat)
                lines_to_show = []
                if form_analysis.get('back_message') and form_analysis.get('back_status'):
                    back_color = (0, 255, 0) if form_analysis['back_status'] == 'good' else (0, 165, 255)
                    lines_to_show.append((form_analysis['back_message'], back_color))
                if form_analysis.get('depth_message') and form_analysis.get('depth_status'):
                    if form_analysis['depth_status'] == 'good':
                        depth_color = (0, 255, 0)
                    elif form_analysis['depth_status'] == 'needs_improvement':
                        depth_color = (0, 165, 255)
                    else:
                        depth_color = (255, 255, 255)
                    lines_to_show.append((form_analysis['depth_message'], depth_color))
                for i, (text, color) in enumerate(lines_to_show[:2]):
                    if len(text) > 45:
                        text = text[:42] + "..."
                    size = cv2.getTextSize(text, font, 0.9, 2)[0]
                    x = ((width - side_margin * 2) - size[0]) // 2 + side_margin
                    y = bg_y_start + 45 + i * 26
                    cv2.putText(image, text, (x, y), font, 0.9, color, 2)
                # Show recommendation below depth message if available
                if form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0:
                    recommendation_text = form_analysis['recommendations'][0]
                    max_chars_per_line = 65
                    if len(recommendation_text) > max_chars_per_line:
                        lines = []
                        words = recommendation_text.split()
                        current_line = ""
                        for word in words:
                            if len(current_line + " " + word) <= max_chars_per_line:
                                current_line += (" " + word) if current_line else word
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                        if current_line:
                            lines.append(current_line)
                    else:
                        lines = [recommendation_text]
                    line_height = 24
                    start_y = bg_y_start + 45 + (len(lines_to_show[:2]) * 26) + 20
                    for i, line in enumerate(lines):
                        line_size = cv2.getTextSize(line, font, 0.75, 2)[0]
                        line_x = ((width - side_margin * 2) - line_size[0]) // 2 + side_margin
                        line_y = start_y + (i * line_height)
                        recommendation_color = (255, 255, 0)
                        cv2.putText(image, line, (line_x, line_y), font, 0.75, recommendation_color, 2)
            
            # Add good/bad form indicator only for squat at bottom
            if feedback.get('exercise_mode') != 'bench':
                # Good/bad indicator is GOOD only when depth is good AND back is good
                depth_good = form_analysis.get('depth_status') == 'good'
                back_good = form_analysis.get('back_status') == 'good'
                has_any_status = form_analysis.get('depth_status') is not None or form_analysis.get('back_status') is not None
                # Only show the GOOD/INCORRECT pill at the bottom of the squat
                phase_is_bottom = form_analysis.get('phase') == 'bottom'
                if phase_is_bottom and has_any_status:
                    is_good = depth_good and back_good
                    pill_color = (0, 255, 0) if is_good else (255, 0, 0)
                    label = 'GOOD FORM' if is_good else 'INCORRECT FORM'

                    # Compute pill dimensions based on text size
                    pill_font = cv2.FONT_HERSHEY_SIMPLEX
                    pill_scale = 0.9
                    pill_thickness = 2
                    text_size = cv2.getTextSize(label, pill_font, pill_scale, pill_thickness)[0]
                    horizontal_padding = 22
                    vertical_padding = 8
                    pill_width = text_size[0] + horizontal_padding * 2
                    pill_height = text_size[1] + vertical_padding * 2

                    # Position centered at the very bottom inside the overlay box
                    available_width = x2 - x1
                    pill_x1 = x1 + max(10, (available_width - pill_width) // 2)
                    pill_x2 = pill_x1 + pill_width
                    pill_y2 = y2 - 10
                    pill_y1 = pill_y2 - pill_height

                    # Draw rounded pill inside the box
                    image = draw_rounded_rectangle_with_alpha(image, pill_x1, pill_y1, pill_x2, pill_y2, pill_color, alpha=0.95, radius=12)

                    # Draw label centered in pill
                    text_x = pill_x1 + (pill_width - text_size[0]) // 2
                    text_y = pill_y1 + (pill_height + text_size[1]) // 2
                    text_color = (0, 0, 0) if is_good else (255, 255, 255)
                    cv2.putText(image, label, (text_x, text_y), pill_font, pill_scale, text_color, pill_thickness)
    
    return image

# Keep the old function for backward compatibility
def draw_feedback_overlay(image, feedback):
    """Draw positioning feedback on the image (legacy function)."""
    return draw_comprehensive_feedback_overlay(image, feedback) 