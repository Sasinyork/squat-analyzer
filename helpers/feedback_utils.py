import cv2
import numpy as np
from PIL import Image, ImageDraw
from .squat_analyzer import SquatFormAnalyzer

class PoseFeedback:
    """Handles real-time feedback for pose detection positioning and squat form."""
    
    def __init__(self):
        self.stable_frames = 0
        self.last_person_center = None
        self.last_person_size = None
        self.feedback_history = []
        self.last_distance_feedback = None
        self.frames_since_movement = 0
        self.movement_threshold = 0.05  # Reduced sensitivity
        self.stability_frames_needed = 15  # Need to hold still for 15 frames (0.5 seconds at 30fps)
        self.feedback_persistence_frames = 90  # Keep feedback for 3 seconds (90 frames at 30fps)
        
        # Initialize squat form analyzer
        self.squat_analyzer = SquatFormAnalyzer()
        self.form_analysis_enabled = True
    
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
    
    def get_comprehensive_feedback(self, keypoints_with_scores, image_height, image_width):
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
        
        # Get squat form analysis if enabled
        form_analysis = None
        if self.form_analysis_enabled and visibility_percentage > 70:  # Only analyze if good visibility
            form_analysis = self.squat_analyzer.analyze_squat_form(keypoints_with_scores, image_height, image_width)
        
        feedback = {
            'message': '',
            'color': (0, 255, 0),
            'distance_status': None,
            'recommendation': '',
            'person_percentage': person_percentage,
            'visibility_percentage': visibility_percentage,
            'is_stable': True,
            'form_analysis': form_analysis
        }
        
        # Only show form feedback
        if form_analysis:
            if form_analysis.get('depth_message'):
                # Only show depth message in overlay, not as main message
                feedback['message'] = ''
                if form_analysis.get('depth_status') == 'good':
                    feedback['color'] = (0, 255, 0)
                elif form_analysis.get('depth_status') == 'needs_improvement':
                    feedback['color'] = (0, 165, 255)
                else:
                    feedback['color'] = (255, 255, 255)
            if form_analysis.get('recommendations'):
                feedback['recommendation'] = form_analysis['recommendations'][0]
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
    base_height = 100  # Increased to accommodate larger text
    extra_height = 0
    
    # Add extra height for form analysis
    if feedback.get('form_analysis'):
        form_analysis = feedback['form_analysis']
        if form_analysis.get('depth_message') and form_analysis.get('depth_status'):
            extra_height += 25  # Increased for larger text
        # Add extra height for recommendations
        if form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0:
            recommendation_text = form_analysis['recommendations'][0]
            # Calculate how many lines the recommendation will need
            max_chars_per_line = 65
            if len(recommendation_text) > max_chars_per_line:
                # Estimate number of lines (rough calculation)
                estimated_lines = max(1, len(recommendation_text) // max_chars_per_line + 1)
                extra_height += (estimated_lines * 15)  # 15 pixels per line
            else:
                extra_height += 20  # Space for single line recommendation text
    
    text_bg_height = base_height + extra_height
    
    # Ensure we don't exceed image bounds and leave some margin
    max_height = min(height * 0.3, 180)  # Increased max height for larger text
    if text_bg_height > max_height:
        text_bg_height = int(max_height)
    
    # Position overlay with margin from bottom and sides
    margin = 5  # Reduced margin to give more space
    side_margin = min(200, width // 8)  # Much smaller side margins for wider feedback background
    bg_y_start = height - text_bg_height - margin
    
    # Ensure valid rectangle coordinates
    x1 = side_margin
    x2 = width - side_margin
    y1 = bg_y_start
    y2 = height - margin
    
    # Only draw if we have a valid rectangle
    if x1 < x2 and y1 < y2:
        # Draw rounded rectangle background with alpha transparency
        image = draw_rounded_rectangle_with_alpha(image, x1, y1, x2, y2, (0, 0, 0), alpha=0.7, radius=10)
        
        # Draw main feedback message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased font size for better visibility
        thickness = 3
        
        # Main message - adjust for narrower width
        text_size = cv2.getTextSize(feedback['message'], font, font_scale, thickness)[0]
        text_x = ((width - side_margin * 2) - text_size[0]) // 2 + side_margin
        text_y = bg_y_start + 20
        cv2.putText(image, feedback['message'], (text_x, text_y), font, font_scale, feedback['color'], thickness)
        
        # Draw form analysis if available (compact)
        if feedback.get('form_analysis'):
            form_analysis = feedback['form_analysis']
            
            # Show depth status message only if it exists
            if form_analysis.get('depth_message') and form_analysis.get('depth_status'):
                depth_text = form_analysis['depth_message']
                # Truncate if too long
                if len(depth_text) > 45:  # Reduced from 50 for mobile
                    depth_text = depth_text[:42] + "..."
                depth_size = cv2.getTextSize(depth_text, font, 0.7, 2)[0]  # Increased font size
                depth_x = ((width - side_margin * 2) - depth_size[0]) // 2 + side_margin
                depth_y = bg_y_start + 45
                
                # Use different colors based on depth status
                if form_analysis.get('depth_status') == 'good':
                    depth_color = (0, 255, 0)  # Green for good depth
                elif form_analysis.get('depth_status') == 'needs_improvement':
                    depth_color = (0, 165, 255)  # Orange for needs improvement
                else:
                    depth_color = (255, 255, 255)  # White for other statuses
                
                cv2.putText(image, depth_text, (depth_x, depth_y), font, 0.7, depth_color, 2)
                
                # Show recommendation below depth message if available
                if form_analysis.get('recommendations') and len(form_analysis['recommendations']) > 0:
                    recommendation_text = form_analysis['recommendations'][0]
                    
                    # Break long text into multiple lines instead of truncating
                    max_chars_per_line = 65
                    if len(recommendation_text) > max_chars_per_line:
                        # Split text into lines
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
                    
                    # Draw each line
                    line_height = 20  # Increased space between lines
                    for i, line in enumerate(lines):
                        line_size = cv2.getTextSize(line, font, 0.6, 2)[0]  # Increased font size
                        line_x = ((width - side_margin * 2) - line_size[0]) // 2 + side_margin
                        line_y = bg_y_start + 65 + (i * line_height)  # Below depth message, with spacing
                        
                        # Use a different color for recommendations (cyan)
                        recommendation_color = (255, 255, 0)  # Cyan for recommendations
                        cv2.putText(image, line, (line_x, line_y), font, 0.6, recommendation_color, 2)
    
    return image

# Keep the old function for backward compatibility
def draw_feedback_overlay(image, feedback):
    """Draw positioning feedback on the image (legacy function)."""
    return draw_comprehensive_feedback_overlay(image, feedback) 