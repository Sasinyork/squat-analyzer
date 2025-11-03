# MoveNet Lightning – Multi‑Exercise Form Analysis

A real-time pose detection and form analysis app powered by Google’s MoveNet Lightning model. The project provides unified, color‑coded feedback overlays for Squat, Bench Press, and Deadlift, including phase detection, rep counting, issue detection, and actionable recommendations.

## 🚀 Features

### Supported Exercises (with phase detection)

- Squat: standing → descending → bottom → ascending
- Bench Press: stable → descending → bottom → ascending
- Deadlift: standing → descending → bottom → ascending

### Core Analysis Capabilities

- Real-time phase detection with debouncing to reduce flicker
- Rep counting with robust phase sequence validation and double-count prevention
- Issue detection with severity (low/medium/high) and color-coded emphasis
- Actionable recommendations with short persistence so you can read them

### Pose Detection & Processing

- Real-time webcam and video file processing
- MoveNet Lightning TFLite model loading from local file
- Rotation handling for portrait videos with automatic counter-rotation on display
- Keypoint smoothing (bench) and rotated-keypoint analysis for lying positions

### Unified, Mobile-Friendly Overlay UI

- Bottom-aligned, rounded, semi-transparent overlay with per-line “pills”
- Unified Phase pill across all exercises with a natural, muted color palette
- Natural, muted color palette for feedback messages and severity levels
- Fixed overlay height for Squat and Deadlift to prevent UI “jumping”
- Top bar rep counter (REPS, CORRECT, INCORRECT) across modes
- Compact per-line backgrounds with automatic readable text color

### Exercise-Specific Details

- Squat

  - Back rounding/forward-lean detection and posture checks
  - Knee alignment/tracking and valgus (knees caving in) detection
  - Depth monitoring with targeted feedback (e.g., “Go deeper”)
  - Rep counting and basic correctness tracking
  - Severity-aware coloring for issues and status lines

- Bench Press

  - Rotated-keypoint analysis for lying position (consistent “depth” trend)
  - Smoothed keypoints and phase detection (stable, descending, bottom, ascending)
  - Unified Phase pill and rep counter UI

- Deadlift
  - Phase detection with robust transitions and state holding
  - Hip extension/hinge cueing and torso alignment checks
  - Knee tracking and basic balance/weight distribution feedback
  - Persistent actionable recommendations (limited-time display)
  - Fixed-height overlay like squat to avoid jitter when messages appear

## 📋 Requirements

- Python 3.8+
- macOS (tested on macOS 24.5.0)
- Webcam access (for real-time mode)
- Sufficient storage for video processing

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd movenet-lightning
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Grant camera permissions** (macOS):
   - Go to **System Settings** > **Privacy & Security** > **Camera**
   - Enable camera access for Terminal (or your terminal app)
   - Restart Terminal after granting permissions

## 🎯 Usage

### Quick Start

Run the application:

```bash
python3 main.py
```

### Interactive Flow

1. Select Exercise Mode:

   ```
   1. Squat
   2. Bench Press
   3. Deadlift
   ```

2. Choose a sample video to process or run live (if implemented), and whether to save the output.

3. Perform the selected exercise while the overlay shows phase, rep counts, and feedback. Hold still briefly to allow stable analysis when needed.

### Understanding the Feedback

- Phase pill: current phase with a muted color (green for up/press, amber for down/lowering, yellow for bottom, gray for rest/standing)
- Rep bar (top): total reps, correct, and incorrect (when supported by analyzer)
- Feedback “pills”: per-line messages for back/depth/knees/balance/etc., severity-colored
- Recommendations: concise, actionable suggestions; may persist briefly for readability

## 🎮 Controls

While processing:

- `q` – Quit
- `s` – Switch to Squat mode
- `b` – Switch to Bench mode
- `d` – Switch to Deadlift mode
- `p` – Pause/Resume

## 📊 Form Analysis Features

### Back Analysis

- **Back Rounding Detection**: Measures spine angle and alerts for excessive forward lean
- **Shoulder Level**: Ensures shoulders remain level during movement
- **Chest Position**: Monitors chest up position for proper form

### Knee Analysis

- **Knee-to-Toe Alignment**: Ensures knees track over toes
- **Valgus Detection**: Alerts when knees cave inward
- **Knee Stability**: Monitors knee position throughout movement

### Depth Analysis

- **Squat Depth**: Ensures proper depth (thighs parallel to ground)
- **Depth Consistency**: Monitors consistent depth across repetitions
- **Excessive Depth**: Warns if going too deep (if uncomfortable)

### Arm Position

- **Arm Extension**: Analyzes arm position for balance
- **Arm Height**: Ensures arms are at appropriate height
- **Arm Symmetry**: Checks for balanced arm positioning

## 🎨 Visualization Features

### Enhanced Keypoint Colors

- **🟡 Yellow**: Head keypoints (nose, eyes, ears)
- **🔴 Red**: Shoulders (critical for back analysis)
- **🟣 Magenta**: Arms (elbows, wrists)
- **🟢 Green**: Hips (critical for squat analysis)
- **🟠 Orange**: Knees (critical for squat analysis)
- **🟣 Purple**: Ankles

### Visual Elements

- Enhanced skeleton and connection styling
- Confidence indicators for high-confidence keypoints
- Bottom overlay with rounded, semi-transparent backgrounds
- Unified Phase pill and natural color palette across exercises
- Fixed overlay height for consistent UI (squat and deadlift)

## 📁 Project Structure

```
movenet-lightning/
├── main.py                    # Main runner (exercise/video selection, processing)
├── main_old.py, main_backup.py# Legacy/backup runners
├── model.tflite               # MoveNet Lightning model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── helpers/
│   ├── feedback_utils.py      # Unified overlay UI + feedback assembly
│   ├── pose_processor.py      # Pose processing pipeline + overlay draw
│   ├── analyzers/
│   │   ├── squat_analyzer.py  # Squat form analysis + phases + reps
│   │   ├── bench_analyzer.py  # Bench press analysis (rotation + smoothing)
│   │   └── deadlift_analyzer.py # Deadlift analysis (hip hinge, balance, phases)
│   └── utils/
│       ├── model_utils.py     # Model loading helpers
│       └── visualization_utils.py # Drawing helpers for skeleton/overlays
├── data/
│   ├── squat/                 # Sample squat videos
│   ├── bench/                 # Sample bench press videos
│   └── deadlift/              # Sample deadlift videos
└── output/                    # Saved processed videos (optional)
```

## 🔧 Configuration

### Adjustable Parameters

In the analyzers, you can modify:

- Angle/Depth thresholds (squat), posture/hinge checks (deadlift), movement thresholds (bench)
- Phase debounce frame counts and min-frames per phase
- Keypoint confidence thresholds and smoothing parameters
- Color palette (muted greens/ambers/grays) for phases and messages in `feedback_utils.py`

### Performance Tips

1. **For Best Form Analysis**:

   - Use side view for squats (camera perpendicular to movement)
   - Ensure full body is visible in frame
   - Maintain good lighting conditions
   - Wear contrasting clothing for better detection

2. **For Real-time Performance**:

   - Use 640x480 resolution for webcam
   - Close other applications
   - Ensure adequate lighting

3. **For Video Analysis**:

   - Use high-quality video input
   - Ensure stable camera position
   - Record from side angle for best analysis

## 🐛 Troubleshooting

### Common Issues

1. **"Cannot open webcam"**:

   - Check camera permissions in System Settings
   - Restart Terminal after granting permissions
   - Ensure no other app is using the camera

2. **Poor form detection**:

   - Improve lighting conditions
   - Ensure full body is visible
   - Wear contrasting clothing
   - Position camera at side view for squats

3. **Inaccurate form analysis**:

   - Ensure camera is perpendicular to movement
   - Maintain consistent distance from camera
   - Perform squats in center of frame
   - Check that all keypoints are visible

4. **Slow performance**:

   - Reduce camera resolution
   - Close other applications
   - Use basic pose detection for faster processing

### Testing the System

If you have tests, run them to verify components work:

```bash
# example
# python3 tests/test_squat_analysis.py
```

This typically covers analyzer logic, feedback aggregation, and visualization helpers.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with `python3 test_squat_analysis.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google's MoveNet team for the excellent pose detection models
- TensorFlow and TensorFlow Hub for model hosting
- Fitness community for form analysis insights

## 🎯 Future Enhancements

- Set tracking and workout summaries
- Form history and progress tracking
- Mobile app version
- Integration with fitness apps
