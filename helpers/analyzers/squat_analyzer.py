import cv2
import numpy as np
import math

class SquatFormAnalyzer:
    """Analyzes squat form and provides feedback on common issues."""
    
    def __init__(self):
        self.prev_keypoints = None
        self.squat_phase = "standing"  # standing, descending, bottom, ascending
        self.phase_frames = 0
        self.form_issues = []
        self.feedback_history = []

        # Rep counting state
        self.rep_count = 0
        self.correct_rep_count = 0
        self.incorrect_rep_count = 0
        self._rep_state = {
            'in_rep': False,
            'bottom_reached': False,
            'issues_at_bottom': [],
        }
        self._last_bottom_issues = []  # Store issues at last bottom phase for rep correctness
        # Rep counting by phase sequence
        self._phase_queue = []  # Track last N phases
        self._max_phase_queue = 5
        self._last_phase = None

        # Phase detection parameters
        self.hip_positions = []  # Store recent hip positions for movement detection
        self.max_history = 10  # Number of frames to track for movement
        self.movement_threshold = 5  # Minimum pixel movement to detect direction
        self.standing_threshold = -50  # Hip must be this much above knee to be "standing" (negative in image coords)
        self.bottom_threshold = -50  # Hip must be close to knee level to be "bottom" (more inclusive for depth analysis)
        self._bottom_cooldown = 0  # Prevent repeated bottom phase triggers
        self._bottom_cooldown_frames = 8  # Number of frames to wait before allowing bottom again

        # Keypoint indices for MoveNet
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
        
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
            
        # Calculate vectors
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def detect_movement_direction(self, current_hip_y):
        """Detect if the person is moving up or down based on hip position history."""
        if len(self.hip_positions) < 3:
            return "unknown"
        
        # Calculate recent movement trend
        recent_positions = self.hip_positions[-5:]  # Last 5 frames
        if len(recent_positions) < 3:
            return "unknown"
        
        # Calculate average movement over recent frames
        total_movement = 0
        for i in range(1, len(recent_positions)):
            movement = recent_positions[i] - recent_positions[i-1]
            total_movement += movement
        
        avg_movement = total_movement / (len(recent_positions) - 1)
        
        # Determine direction based on movement
        if avg_movement > self.movement_threshold:
            return "descending"  # Hip moving down (y increasing)
        elif avg_movement < -self.movement_threshold:
            return "ascending"   # Hip moving up (y decreasing)
        else:
            return "stable"       # Minimal movement
    
    def detect_squat_phase(self, keypoints, image_height, image_width):
        """Detect the current phase of the squat movement with improved logic."""
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        
        if left_hip and right_hip and left_knee and right_knee:
            hip_y = (left_hip[1] + right_hip[1]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2
            hip_knee_diff = hip_y - knee_y

            self.hip_positions.append(hip_y)
            if len(self.hip_positions) > self.max_history:
                self.hip_positions.pop(0)

            movement_direction = self.detect_movement_direction(hip_y)

            # FSM logic without setup state
            prev_phase = self.squat_phase
            new_phase = prev_phase

            min_descend_frames = 5
            min_ascend_frames = 10

            if hip_knee_diff < self.standing_threshold and movement_direction == "stable":
                # If user is clearly standing, force phase to standing
                new_phase = "standing"
            elif prev_phase == "standing":
                # Only start descent if hip drops and movement is clear
                if movement_direction == "descending" and hip_knee_diff > self.standing_threshold + 10:
                    new_phase = "descending"
            elif prev_phase == "descending":
                if self.phase_frames < min_descend_frames:
                    new_phase = "descending"
                else:
                    if hip_knee_diff > self.bottom_threshold and movement_direction in ["stable", "ascending"]:
                        new_phase = "bottom"
                    elif movement_direction == "ascending":
                        new_phase = "ascending"
                    else:
                        new_phase = "descending"
            elif prev_phase == "bottom":
                if movement_direction == "ascending" and hip_knee_diff < self.bottom_threshold:
                    new_phase = "ascending"
            elif prev_phase == "ascending":
                if self.phase_frames < min_ascend_frames:
                    new_phase = "ascending"
                else:
                    if hip_knee_diff < self.standing_threshold and movement_direction == "stable":
                        new_phase = "standing"
                    elif hip_knee_diff > self.bottom_threshold and movement_direction == "descending":
                        new_phase = "descending"
                    else:
                        new_phase = "ascending"

            if new_phase == prev_phase:
                if movement_direction == "descending" and prev_phase != "descending":
                    new_phase = "descending"
                elif movement_direction == "ascending" and prev_phase != "ascending":
                    new_phase = "ascending"

            if new_phase == prev_phase:
                self.phase_frames += 1
            else:
                self.squat_phase = new_phase
                self.phase_frames = 0

            return new_phase, hip_knee_diff
        return self.squat_phase, 0
    
    def detect_view_angle(self, keypoints, image_height, image_width):
        """Detect if the person is viewed from front, side, or angle."""
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Calculate shoulder and hip widths
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            hip_width = abs(left_hip[0] - right_hip[0])
            
            # If shoulders/hips are very close together, likely side view
            if shoulder_width < 50 and hip_width < 50:
                return "side"
            # If shoulders/hips are far apart, likely front view
            elif shoulder_width > 100 and hip_width > 100:
                return "front"
            # Otherwise, likely angled view
            else:
                return "angled"
        
        return "unknown"
    
    def analyze_back_form(self, keypoints, image_height, image_width):
        """Analyze back form for rounding issues."""
        issues = []
        
        # Detect view angle
        view_angle = self.detect_view_angle(keypoints, image_height, image_width)
        
        # Get keypoints for back analysis
        nose = self.get_keypoint_coords(keypoints, self.NOSE, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Calculate shoulder and hip centers
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            # Calculate back angle (should be relatively straight)
            if nose:
                # Calculate angle between nose-shoulder-hip
                back_angle = self.calculate_angle(nose, shoulder_center, hip_center)
                
                if back_angle:
                    # Adjust thresholds based on view angle
                    if view_angle == "side":
                        # For side view, be more lenient with back angle
                        angle_threshold = 140  # More lenient for side view
                    else:
                        # For front/angled view, use stricter threshold
                        angle_threshold = 150
                    
                    # Check for excessive forward lean (back rounding)
                    if back_angle < angle_threshold:
                        issues.append({
                            'type': 'back_rounding',
                            'severity': 'high' if back_angle < (angle_threshold - 10) else 'medium',
                            'recommendation': 'Keep chest up and back straight'
                        })
            
            # Only check for lateral tilt in front/angled views
            if view_angle != "side":
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                if shoulder_diff > 15:  # More than 15 pixels difference
                    issues.append({
                        'type': 'shoulder_tilt',
                        'severity': 'medium',
                        'recommendation': 'Keep shoulders level and square'
                    })
        
        return issues
    
    def analyze_knee_form(self, keypoints, image_height, image_width):
        """Analyze knee position and alignment."""
        issues = []
        
        # Detect view angle
        view_angle = self.detect_view_angle(keypoints, image_height, image_width)
        
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        left_ankle = self.get_keypoint_coords(keypoints, self.LEFT_ANKLE, image_height, image_width)
        right_ankle = self.get_keypoint_coords(keypoints, self.RIGHT_ANKLE, image_height, image_width)
        
        if left_knee and right_knee and left_ankle and right_ankle:
            # Calculate stance center for better knee tracking analysis
            stance_center_x = (left_ankle[0] + right_ankle[0]) / 2
            stance_width = abs(left_ankle[0] - right_ankle[0])
            
            # Check knee tracking relative to stance center (more accurate than ankle alignment)
            left_knee_tracking = abs(left_knee[0] - stance_center_x)
            right_knee_tracking = abs(right_knee[0] - stance_center_x)
            
            # Knees should track within the stance width (not too far outside)
            # Allow some flexibility but not excessive tracking outside
            max_tracking_distance = stance_width * 0.4  # 40% of stance width
            
            if left_knee_tracking > max_tracking_distance or right_knee_tracking > max_tracking_distance:
                issues.append({
                    'type': 'knee_tracking',
                    'severity': 'medium',
                    'recommendation': 'Keep knees within your stance width'
                })
            
            # Check for excessive knee movement forward (common squat issue)
            # Compare knee position to ankle position in X-axis
            left_knee_forward = left_knee[0] - left_ankle[0]
            right_knee_forward = right_knee[0] - right_ankle[0]
            
            # Knees should not go too far forward past ankles
            max_forward_distance = 50  # pixels - adjust based on testing
            
            if left_knee_forward > max_forward_distance or right_knee_forward > max_forward_distance:
                issues.append({
                    'type': 'knee_forward',
                    'severity': 'medium',
                    'recommendation': 'Keep knees behind toes, sit back more'
                })
            
    
        return issues
    
    def analyze_depth(self, keypoints, image_height, image_width):
        """Analyze squat depth only."""
        issues = []
        
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        
        if left_hip and right_hip and left_knee and right_knee:
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            knee_center = ((left_knee[0] + right_knee[0]) / 2, 
                           (left_knee[1] + right_knee[1]) / 2)
            
            # Calculate depth (in image coordinates, y increases downward)
            # depth_ratio < 0: hip above knee (insufficient)
            # depth_ratio ≈ 0: hip aligned with knee (good)
            # depth_ratio > 0: hip below knee (excessive)
            depth_ratio = (hip_center[1] - knee_center[1]) / image_height
            
            # Analyze depth during bottom phase or when descending and close to knee level
            should_analyze = False
            if self.squat_phase == "bottom":
                should_analyze = True
            elif self.squat_phase == "descending":
                # Check if hip is close to knee level (within 50 pixels) or if we've been descending for a while
                hip_knee_diff = hip_center[1] - knee_center[1]
                if abs(hip_knee_diff) < 300:  # Within 50 pixels of knee level
                    should_analyze = True
                elif self.phase_frames > 10:  # Been descending for more than 10 frames
                    should_analyze = True
            
            if should_analyze:
                good_depth_tolerance = 0.05  # 5% of image height tolerance
                # Allow hip to be slightly above knee (more lenient depth requirement)
                depth_allowance = 0.02 * image_height  # 2% of image height above knee level
                if hip_center[1] < knee_center[1] - depth_allowance:
                    # Hip too far above knee - insufficient depth
                    issues.append({
                        'type': 'insufficient_depth',
                        'severity': 'medium',
                        'recommendation': 'Go deeper - hips should be at least parallel with knees'
                    })
                # good depth is when hip is aligned with knee level
                elif abs(hip_center[1] - knee_center[1]) <= good_depth_tolerance * image_height:
                    # Hip aligned with knee - good depth
                    # No issues to append
                    pass
                else:
                    # Hip below knee - deep squat (considered good form)
                    # No issues to append - deep squats are generally better
                    pass
        
        return issues
    
    
    def analyze_arm_position(self, keypoints, image_height, image_width):
        """Analyze arm position during squat."""
        issues = []
        
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_elbow = self.get_keypoint_coords(keypoints, self.LEFT_ELBOW, image_height, image_width)
        right_elbow = self.get_keypoint_coords(keypoints, self.RIGHT_ELBOW, image_height, image_width)
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        
        if left_shoulder and right_shoulder and left_elbow and right_elbow:
            # Check if arms are raised (common in bodyweight squats)
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            elbow_center = ((left_elbow[0] + right_elbow[0]) / 2, 
                           (left_elbow[1] + right_elbow[1]) / 2)
            
            # Arms should be roughly at shoulder level or extended forward
            arm_angle = self.calculate_angle(shoulder_center, elbow_center, 
                                           (elbow_center[0], elbow_center[1] - 50))
            
            if arm_angle and arm_angle < 45:  # Arms too low
                issues.append({
                    'type': 'arm_position',
                    'severity': 'low',
                    'recommendation': 'Extend arms forward or raise them higher'
                })
        
        return issues
    
    def analyze_squat_form(self, keypoints_with_scores, image_height, image_width):
        """Squat form analysis including depth and back rounding."""

        keypoints = keypoints_with_scores[0, 0, :, :]
        # Detect squat phase
        phase, depth_metric = self.detect_squat_phase(keypoints, image_height, image_width)
        # Debug: print only when phase changes
        # if not hasattr(self, '_last_debug_phase') or self._last_debug_phase != phase:
            # print(f"[Squat Debug] Current phase: {phase} | Last 5 phases: {self._phase_queue[-5:]}")
            # self._last_debug_phase = phase

        # Only analyze depth during bottom phase or if hip is near knee level
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)

        # Show depth feedback during bottom phase or when we detect depth issues during descending
        show_depth_feedback = False
        if phase == "bottom":
            show_depth_feedback = True

        # Always run analyze_depth/back, but only show depth feedback if show_depth_feedback is True
        depth_issues = self.analyze_depth(keypoints, image_height, image_width)
        back_issues = self.analyze_back_form(keypoints, image_height, image_width)

        # If we're descending and have depth issues, show feedback
        if phase == "descending" and depth_issues:
            show_depth_feedback = True

        # Merge all detected issues for scoring and recommendations
        all_issues = []
        all_issues.extend(depth_issues)
        all_issues.extend(back_issues)



        # --- REP COUNTING LOGIC: robust phase sequence method ---
        # Track last N phases
        if not self._phase_queue or self._phase_queue[-1] != phase:
            self._phase_queue.append(phase)
            if len(self._phase_queue) > self._max_phase_queue:
                self._phase_queue.pop(0)

        # Store issues at bottom phase for rep correctness
        if phase == "bottom":
            self._last_bottom_issues = all_issues.copy()

        # More robust rep detection: allow a single 'stable' or 'unknown' phase in the sequence
        def matches_expected(seq, expected):
            if len(seq) != len(expected):
                return False
            mismatches = 0
            for s, e in zip(seq, expected):
                if s == e:
                    continue
                if s in ("stable", "unknown"):
                    mismatches += 1
                else:
                    return False
            return mismatches <= 1

        expected_seq = ["standing", "descending", "bottom", "ascending", "standing"]
        if len(self._phase_queue) >= 5 and matches_expected(self._phase_queue[-5:], expected_seq):
            self.rep_count += 1
            # Use issues at bottom for correctness
            is_correct = not any(i.get('severity', 'low') in ('medium', 'high') for i in self._last_bottom_issues)
            if is_correct:
                self.correct_rep_count += 1
            else:
                self.incorrect_rep_count += 1
            self._phase_queue = []  # Reset to avoid double-counting

        self._last_phase = phase

        feedback = {
            'phase': phase,
            'phase_frames': self.phase_frames,
            'issues': all_issues,
            'depth_metric': depth_metric,
            'overall_score': self.calculate_form_score(all_issues),
            'primary_issue': self.get_primary_issue(all_issues),
            'recommendations': self.get_recommendations(all_issues),
            'rep_count': self.rep_count,
            'correct_rep_count': self.correct_rep_count,
            'incorrect_rep_count': self.incorrect_rep_count
        }

        if show_depth_feedback:
            if not depth_issues:
                feedback['depth_status'] = 'good'
                feedback['depth_message'] = 'Good depth - hips aligned with knees'
                # Add recommendation for good depth
                feedback['recommendations'] = ['Great! Knee level is the minimum, but going deeper is even better - especially for building strength and muscle.']
            else:
                feedback['depth_status'] = 'needs_improvement'
                # Use the first recommendation for depth issues, or a generic message
                feedback['depth_message'] = depth_issues[0]['recommendation'] if 'recommendation' in depth_issues[0] else 'Needs improvement'
        else:
            feedback['depth_status'] = None
            feedback['depth_message'] = None

        # Back rounding feedback (always show if detected)
        if back_issues:
            feedback['back_status'] = 'needs_improvement'
            # Use the first recommendation for back issues, or a generic message
            feedback['back_message'] = next((i['recommendation'] for i in back_issues if i['type'] == 'back_rounding' and 'recommendation' in i), back_issues[0]['recommendation'] if 'recommendation' in back_issues[0] else 'Needs improvement')
        else:
            feedback['back_status'] = 'good'
            feedback['back_message'] = 'Good back position - neutral spine maintained'

        self.feedback_history.append(feedback)
        if len(self.feedback_history) > 10:
            self.feedback_history.pop(0)
        return feedback
    
    def calculate_form_score(self, issues):
        """Calculate overall form score (0-100)."""
        if not issues:
            return 100
        
        # Weight issues by severity
        severity_weights = {'low': 5, 'medium': 15, 'high': 25}
        total_penalty = sum(severity_weights.get(issue['severity'], 10) for issue in issues)
        
        return max(0, 100 - total_penalty)
    
    def get_primary_issue(self, issues):
        """Get the most important issue to address."""
        if not issues:
            return None
        
        # Prioritize by severity and type
        priority_order = ['back_rounding', 'knee_alignment', 
                         'insufficient_depth', 'shoulder_tilt', 'arm_position']
        
        for priority_type in priority_order:
            for issue in issues:
                if issue['type'] == priority_type:
                    return issue
        
        return issues[0]  # Return first issue if no priority match
    
    def get_recommendations(self, issues):
        """Get actionable recommendations."""
        if not issues:
            return ["Good depth! Keep hips aligned with knees"]
        
        recommendations = []
        for issue in issues:
            if issue['recommendation'] not in recommendations:
                recommendations.append(issue['recommendation'])
        
        return recommendations[:3]  # Limit to top 3 recommendations 