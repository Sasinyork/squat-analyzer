import numpy as np

class DeadliftFormAnalyzer:
    def analyze_weight_distribution_and_balance(self, keypoints, image_height, image_width, phase):
        """
        Analyze weight distribution and balance for deadlift using available keypoints.
        - Forward weight shift: Heels lifting, weight on toes (ankle rises or moves forward relative to hip/shoulder)
        - Backward lean: Excessive lean back at lockout (hips forward of ankles, torso leans back past vertical)
        """
        issues = []
        left_ankle = self.get_keypoint_coords(keypoints, self.LEFT_ANKLE, image_height, image_width)
        right_ankle = self.get_keypoint_coords(keypoints, self.RIGHT_ANKLE, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        
        # Use average positions for symmetry
        if all([left_ankle, right_ankle, left_hip, right_hip, left_shoulder, right_shoulder]):
            avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            avg_ankle_x = (left_ankle[0] + right_ankle[0]) / 2
            avg_hip_x = (left_hip[0] + right_hip[0]) / 2
            avg_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2

            # Forward weight shift: ankle rises (y decreases) or moves forward (x increases) relative to hip/shoulder during ascent
            if phase == 'ascending':
                # If ankle y is much higher than hip/shoulder y (i.e., foot is coming off ground)
                if avg_ankle_y < min(avg_hip_y, avg_shoulder_y) - 30:  # 30px threshold, tune as needed
                    issues.append({
                        'type': 'forward_weight_shift',
                        'severity': 'medium',
                        'message': 'Possible forward weight shift (heels lifting)',
                        'recommendation': 'Keep weight balanced over midfoot; avoid lifting heels.'
                    })
                # If ankle x moves much forward of hip/shoulder x (for side view)
                if abs(avg_ankle_x - avg_hip_x) > 50 and avg_ankle_x > avg_hip_x:
                    issues.append({
                        'type': 'forward_weight_shift',
                        'severity': 'medium',
                        'message': 'Possible forward weight shift (ankles ahead of hips)',
                        'recommendation': 'Keep weight balanced; avoid letting ankles move far ahead of hips.'
                    })

            # Backward lean at lockout: hips forward of ankles, torso leans back
            if phase == 'standing':
                # Hips significantly forward of ankles (x axis, for side view)
                if avg_hip_x - avg_ankle_x > 40:
                    issues.append({
                        'type': 'backward_lean',
                        'severity': 'medium',
                        'message': 'Excessive backward lean at lockout',
                        'recommendation': 'Finish tall and neutral; avoid leaning back at lockout.'
                    })
                # Shoulders behind hips (x axis, for side view)
                if avg_shoulder_x < avg_hip_x - 20:
                    issues.append({
                        'type': 'backward_lean',
                        'severity': 'low',
                        'message': 'Torso leans back past vertical at lockout',
                        'recommendation': 'Keep torso stacked over hips at lockout.'
                    })
        return issues
    def analyze_spine_torso_alignment(self, keypoints, image_height, image_width, phase):
        """Analyze spine and torso alignment for deadlift."""
        issues = []
        # Get keypoints
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        nose = self.get_keypoint_coords(keypoints, self.NOSE, image_height, image_width)
        if not all([left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee]):
            return issues
        # Helper: angle between three points (at b)
        def angle(a, b, c):
            ba = np.array([a[0] - b[0], a[1] - b[1]])
            bc = np.array([c[0] - b[0], c[1] - b[1]])
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        # Lower back rounding: hip-shoulder-knee angle (should be ~180, lower = rounding)
        left_lb_angle = angle(left_hip, left_shoulder, left_knee)
        right_lb_angle = angle(right_hip, right_shoulder, right_knee)
        avg_lb_angle = (left_lb_angle + right_lb_angle) / 2
        if avg_lb_angle < 145:
            issues.append({
                'type': 'lumbar_flexion',
                'severity': 'high',
                'message': 'Lower back rounding detected',
                'recommendation': 'Maintain a neutral lower back; avoid rounding.'
            })
        # Upper back rounding: nose-shoulder-hip angle (should be ~180, lower = rounding)
        if nose:
            left_ub_angle = angle(nose, left_shoulder, left_hip)
            right_ub_angle = angle(nose, right_shoulder, right_hip)
            avg_ub_angle = (left_ub_angle + right_ub_angle) / 2
            if avg_ub_angle < 145:
                issues.append({
                    'type': 'thoracic_flexion',
                    'severity': 'medium',
                    'message': 'Upper back rounding detected',
                    'recommendation': 'Keep chest up and avoid excessive upper back rounding.'
                })
        # Hyperextension: at lockout (standing), if hip-shoulder-knee angle > 195
        if phase == 'standing' and avg_lb_angle > 205:
            issues.append({
                'type': 'hyperextension',
                'severity': 'medium',
                'message': 'Excessive arching at lockout (hyperextension)',
                'recommendation': 'Do not over-arch your back at the top; finish tall and neutral.'
            })
        # Torso angle consistency: track torso angle (shoulder-hip to vertical) across movement
        # Save torso angle history for consistency check
        torso_vec = np.array([(left_shoulder[0] + right_shoulder[0]) / 2 - (left_hip[0] + right_hip[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2 - (left_hip[1] + right_hip[1]) / 2])
        torso_angle = np.degrees(np.arctan2(torso_vec[0], torso_vec[1]))  # Angle from vertical
        if not hasattr(self, '_torso_angle_history'):
            self._torso_angle_history = []
        self._torso_angle_history.append(torso_angle)
        if len(self._torso_angle_history) > self.max_history:
            self._torso_angle_history.pop(0)
        if phase == 'descending' and len(self._torso_angle_history) >= 5:
            # Check std deviation of last 5 angles
            recent = self._torso_angle_history[-5:]
            if np.std(recent) > 14:
                issues.append({
                    'type': 'torso_angle_inconsistent',
                    'severity': 'low',
                    'message': 'Torso angle is changing too much during descent',
                    'recommendation': 'Maintain a consistent torso angle throughout the movement.'
                })
        return issues
    def analyze_knee_tracking(self, keypoints, image_height, image_width, phase):
        """Analyze knee tracking for deadlift: stability, timing, lockout."""
        issues = []
        # Get keypoints
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        left_ankle = self.get_keypoint_coords(keypoints, self.LEFT_ANKLE, image_height, image_width)
        right_ankle = self.get_keypoint_coords(keypoints, self.RIGHT_ANKLE, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        if not all([left_knee, right_knee, left_ankle, right_ankle, left_hip, right_hip]):
            return issues
        # 1. Knee stability: excessive forward travel (knees past toes)
        # Use average for both sides
        knee_x = (left_knee[0] + right_knee[0]) / 2
        ankle_x = (left_ankle[0] + right_ankle[0]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        # Forward travel: knee y should not be much ahead of ankle y (in image, y increases downward)
        forward_travel = knee_y - ankle_y
        if forward_travel > image_height * 0.08:  # 8% of image height
            issues.append({
                'type': 'knee_forward',
                'severity': 'medium',
                'message': 'Knees are traveling too far forward',
                'recommendation': 'Keep shins more vertical; avoid knees past toes.'
            })
        # Collapse/valgus: knees caving in (x distance between knees < ankles)
        knee_dist = abs(left_knee[0] - right_knee[0])
        ankle_dist = abs(left_ankle[0] - right_ankle[0])
        if knee_dist < ankle_dist * 0.7:
            issues.append({
                'type': 'knee_valgus',
                'severity': 'high',
                'message': 'Knees are caving inward (valgus)',
                'recommendation': 'Push knees outward and keep them aligned with feet.'
            })
        # 2. Knee bend timing: knees bend too early in RDL descent (should hinge hips first)
        # If phase is descending and knees bend before hips move back, flag it
        # Use hip and knee angles: if knee angle decreases before hip moves back, it's early knee bend
        # For simplicity, compare vertical position of hip vs knee at start of descent
        if phase == 'descending':
            # Save initial positions at start of descent
            if not hasattr(self, '_descent_start_hip_y') or self._descent_start_hip_y is None:
                self._descent_start_hip_y = (left_hip[1] + right_hip[1]) / 2
                self._descent_start_knee_y = (left_knee[1] + right_knee[1]) / 2
                self._descent_start_frame = 0
            self._descent_start_frame += 1
            # After a few frames, check if knees have bent a lot but hips haven't moved back much
            if self._descent_start_frame == 4:
                hip_delta = ((left_hip[1] + right_hip[1]) / 2) - self._descent_start_hip_y
                knee_delta = ((left_knee[1] + right_knee[1]) / 2) - self._descent_start_knee_y
                if knee_delta > hip_delta * 1.2 and knee_delta > 8:  # Knee bends more than hip moves
                    issues.append({
                        'type': 'knee_bend_timing',
                        'severity': 'medium',
                        'message': 'Knees are bending too early in descent',
                        'recommendation': 'Initiate descent by hinging hips back before bending knees.'
                    })
        else:
            self._descent_start_hip_y = None
            self._descent_start_knee_y = None
            self._descent_start_frame = 0
        # 3. Knee lockout: soft knees at top (should be nearly straight at standing)
        if phase == 'standing':
            # Angle at knee: if not close to 180, knees are soft
            def angle(a, b, c):
                # Angle at b
                ba = np.array([a[0] - b[0], a[1] - b[1]])
                bc = np.array([c[0] - b[0], c[1] - b[1]])
                cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            # Use hip-knee-ankle angle
            left_angle = angle(left_hip, left_knee, left_ankle)
            right_angle = angle(right_hip, right_knee, right_ankle)
            avg_knee_angle = (left_angle + right_angle) / 2
            if avg_knee_angle < 170:  # Not fully locked out
                issues.append({
                    'type': 'knee_lockout',
                    'severity': 'low',
                    'message': 'Knees are not fully locked out at the top',
                    'recommendation': 'Stand tall and fully extend knees at the top.'
                })
        return issues
    def calculate_hip_backward_extension(self, keypoints, image_height, image_width):
        """Calculate normalized hip backward extension (hip behind ankle, normalized by leg length)."""
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_ankle = self.get_keypoint_coords(keypoints, self.LEFT_ANKLE, image_height, image_width)
        right_ankle = self.get_keypoint_coords(keypoints, self.RIGHT_ANKLE, image_height, image_width)
        if left_hip and right_hip and left_ankle and right_ankle:
            hip_x = (left_hip[0] + right_hip[0]) / 2
            ankle_x = (left_ankle[0] + right_ankle[0]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            extension = ankle_x - hip_x  # Positive if hip is behind ankle
            leg_length = np.sqrt((ankle_x - hip_x) ** 2 + (ankle_y - hip_y) ** 2)
            if leg_length > 1e-3:
                normalized_extension = extension / leg_length
            else:
                normalized_extension = 0.0
            return normalized_extension, extension
        return None, None
    """Analyzes deadlift form and detects movement phases: standing, descending, ascending. Also stubs for hip hinge and premature hip rise."""
    def __init__(self):
        self.prev_keypoints = None
        self.deadlift_phase = "standing"  # standing, descending, ascending
        self.phase_frames = 0
        self.hip_positions = []  # Track hip y positions for movement
        self.hip_x_positions = []  # Track hip x positions for hinge
        self.shoulder_y_positions = []  # For premature hip rise
        self.max_history = 10
        self.movement_threshold = 5  # Minimum pixel movement to detect direction
        self.hip_hinge_threshold = 8  # Minimum horizontal movement to count as hinge
        self.standing_vertical_threshold = 30  # px, for vertical alignment

        # MoveNet keypoint indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16

    def get_keypoint_coords(self, keypoints, index, image_height, image_width):
        # Returns (x, y) in pixel coordinates if confidence > 0.15
        if keypoints[index, 2] > 0.15:
            y = int(keypoints[index, 0] * image_height)
            x = int(keypoints[index, 1] * image_width)
            return (x, y)
        return None

    def is_standing(self, keypoints, image_height, image_width):
        # Standing: shoulders above hips, hips above knees, knees above ankles, and all nearly vertically aligned
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        left_ankle = self.get_keypoint_coords(keypoints, self.LEFT_ANKLE, image_height, image_width)
        right_ankle = self.get_keypoint_coords(keypoints, self.RIGHT_ANKLE, image_height, image_width)

        if not all([left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            return False

        # Use average x/y for each joint
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_x = (left_hip[0] + right_hip[0]) / 2
        knee_x = (left_knee[0] + right_knee[0]) / 2
        ankle_x = (left_ankle[0] + right_ankle[0]) / 2

        # Check vertical order
        if not (shoulder_y < hip_y < knee_y < ankle_y):
            return False

        # Check vertical alignment (x positions close)
        if (abs(shoulder_x - hip_x) > self.standing_vertical_threshold or
            abs(hip_x - knee_x) > self.standing_vertical_threshold or
            abs(knee_x - ankle_x) > self.standing_vertical_threshold):
            return False

        return True

    def detect_movement_direction(self, current_hip_y, current_hip_x, current_shoulder_x):
        # Returns 'ascending', 'descending', or None based on hip y and x movement relative to shoulders
        self.hip_positions.append(current_hip_y)
        self.hip_x_positions.append(current_hip_x)
        if len(self.hip_positions) > self.max_history:
            self.hip_positions.pop(0)
        if len(self.hip_x_positions) > self.max_history:
            self.hip_x_positions.pop(0)
        if len(self.hip_positions) < 3 or len(self.hip_x_positions) < 3:
            return None, 0
        recent_y = self.hip_positions[-5:]
        recent_x = self.hip_x_positions[-5:]
        if len(recent_y) < 3 or len(recent_x) < 3:
            return None, 0
        total_movement_y = 0
        total_movement_x = 0
        for i in range(1, len(recent_y)):
            total_movement_y += recent_y[i] - recent_y[i-1]
            total_movement_x += recent_x[i] - recent_x[i-1]
        avg_movement_y = total_movement_y / (len(recent_y) - 1)
        avg_movement_x = total_movement_x / (len(recent_x) - 1)

        # Use hip relative to shoulder for hinge: if hips move further back (x decreases relative to shoulders), it's descending
        # Save shoulder x history for relative comparison
        if not hasattr(self, 'shoulder_x_history'):
            self.shoulder_x_history = []
        self.shoulder_x_history.append(current_shoulder_x)
        if len(self.shoulder_x_history) > self.max_history:
            self.shoulder_x_history.pop(0)
        if len(self.shoulder_x_history) < 3:
            return None, 0
        recent_shoulder_x = self.shoulder_x_history[-5:]
        avg_shoulder_x = sum(recent_shoulder_x) / len(recent_shoulder_x)
        avg_hip_x = sum(recent_x) / len(recent_x)
        hip_to_shoulder_x = avg_hip_x - avg_shoulder_x
        # Compare change in hip_to_shoulder_x over time
        hip_to_shoulder_x_start = recent_x[0] - recent_shoulder_x[0]
        hip_to_shoulder_x_end = recent_x[-1] - recent_shoulder_x[-1]
        hip_to_shoulder_delta = hip_to_shoulder_x_end - hip_to_shoulder_x_start

        # If hips move back relative to shoulders (delta negative), it's descending
        if hip_to_shoulder_delta < -self.hip_hinge_threshold:
            return "descending", hip_to_shoulder_delta
        elif hip_to_shoulder_delta > self.hip_hinge_threshold:
            return "ascending", hip_to_shoulder_delta
        elif abs(hip_to_shoulder_delta) < self.hip_hinge_threshold:
            return "standing", hip_to_shoulder_delta
        else:
            return None, hip_to_shoulder_delta

    def detect_deadlift_phase(self, keypoints, image_height, image_width):
        """Detect the current phase of the deadlift movement: standing, descending, ascending, using hip and shoulder mechanics."""
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)

        if left_hip and right_hip and left_knee and right_knee and left_shoulder and right_shoulder:
            # Use average hip and knee y/x
            hip_y = (left_hip[1] + right_hip[1]) / 2
            hip_x = (left_hip[0] + right_hip[0]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2
            shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            # Check for standing posture
            if self.is_standing(keypoints, image_height, image_width):
                phase = "standing"
            else:
                direction, hip_to_shoulder_delta = self.detect_movement_direction(hip_y, hip_x, shoulder_x)
                if direction == "descending":
                    phase = "descending"
                elif direction == "ascending":
                    phase = "ascending"
                else:
                    phase = self.deadlift_phase  # Hold last phase if uncertain
            self.deadlift_phase = phase
            return phase
        return None

    def analyze_deadlift_form(self, keypoints_with_scores, image_height, image_width):
        """Returns phase and pose info for deadlift, including hip backward extension feedback and weight distribution/balance."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        phase = self.detect_deadlift_phase(keypoints, image_height, image_width)
        # Hip backward extension analysis
        hip_ext_norm, hip_ext_px = self.calculate_hip_backward_extension(keypoints, image_height, image_width)
        hip_extension_issue = None
        hip_extension_threshold = 0.10  # Minimum normalized extension (tune as needed)
        hip_extension_status = None
        hip_extension_message = None
        recommendations = []
        if hip_ext_norm is not None and phase in ("descending", "bottom"):
            if hip_ext_norm < hip_extension_threshold:
                hip_extension_status = 'needs_improvement'
                hip_extension_message = 'Push hips back'
                recommendations.append('Push hips back')
            else:
                hip_extension_status = 'good'
                hip_extension_message = 'Good hip extension'

        # Knee tracking analysis
        knee_issues = self.analyze_knee_tracking(keypoints, image_height, image_width, phase)
        for issue in knee_issues:
            # Shorten knee feedback
            short_map = {
                'Keep shins more vertical; avoid knees past toes.': 'Shins more vertical',
                'Push knees outward and keep them aligned with feet.': 'Push knees out',
                'Initiate descent by hinging hips back before bending knees.': 'Hinge hips first',
                'Stand tall and fully extend knees at the top.': 'Lock knees at top'
            }
            rec = short_map.get(issue['recommendation'], issue['recommendation'])
            recommendations.append(rec)
        # Spine & torso alignment analysis
        spine_issues = self.analyze_spine_torso_alignment(keypoints, image_height, image_width, phase)
        for issue in spine_issues:
            # Shorten spine/torso feedback
            short_map = {
                'Maintain a neutral lower back; avoid rounding.': 'Neutral lower back',
                'Keep chest up and avoid excessive upper back rounding.': 'Chest up',
                'Do not over-arch your back at the top; finish tall and neutral.': 'No over-arch at top',
                'Maintain a consistent torso angle throughout the movement.': 'Torso angle consistent'
            }
            rec = short_map.get(issue['recommendation'], issue['recommendation'])
            recommendations.append(rec)
        # Weight distribution & balance analysis
        balance_issues = self.analyze_weight_distribution_and_balance(keypoints, image_height, image_width, phase)
        for issue in balance_issues:
            # Shorten balance feedback
            short_map = {
                'Keep weight balanced over midfoot; avoid lifting heels.': 'Weight midfoot',
                'Keep weight balanced; avoid letting ankles move far ahead of hips.': 'Weight midfoot',
                'Finish tall and neutral; avoid leaning back at lockout.': 'Stand tall at top',
                'Keep torso stacked over hips at lockout.': 'Torso over hips at top'
            }
            rec = short_map.get(issue['recommendation'], issue['recommendation'])
            recommendations.append(rec)
        return {
            "phase": phase,
            "hip_extension_norm": hip_ext_norm,
            "hip_extension_px": hip_ext_px,
            "hip_extension_status": hip_extension_status,
            "hip_extension_message": hip_extension_message,
            "knee_issues": knee_issues,
            "spine_issues": spine_issues,
            "balance_issues": balance_issues,
            "recommendations": recommendations
        }
