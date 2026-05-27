# MoveNet Lightning – Squat Form Analysis

A real-time pose detection and form analysis app powered by Google's MoveNet Lightning model. The project provides color-coded feedback overlays for Squat form, including phase detection, rep counting, issue detection, and actionable recommendations.

## Features

### Squat Phase Detection

- standing → descending → bottom → ascending

### Core Analysis Capabilities

- Real-time phase detection with debouncing to reduce flicker
- Rep counting with robust phase sequence validation and double-count prevention
- Issue detection with severity (low/medium/high) and color-coded emphasis
- Actionable recommendations with short persistence so you can read them

### Pose Detection & Processing

- Real-time webcam and video file processing
- MoveNet Lightning TFLite model loading from local file
- Rotation handling for portrait videos with automatic counter-rotation on display

### Unified, Mobile-Friendly Overlay UI

- Bottom-aligned, rounded, semi-transparent overlay with per-line "pills"
- Unified Phase pill with a natural, muted color palette
- Fixed overlay height to prevent UI "jumping"
- Top bar rep counter (REPS, CORRECT, INCORRECT)
- Compact per-line backgrounds with automatic readable text color

### Squat-Specific Analysis

- Back rounding/forward-lean detection and posture checks
- Knee alignment/tracking and valgus (knees caving in) detection
- Depth monitoring with targeted feedback (e.g., "Go deeper")
- Rep counting and basic correctness tracking
- Severity-aware coloring for issues and status lines

## Requirements

- Python 3.8+
- macOS (tested on macOS 24.5.0)
- Webcam access (for real-time mode)
- Sufficient storage for video processing

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Sasinyork/squat-analyzer.git
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

## Usage

### Quick Start

Run the application:

```bash
python3 main.py
```

### Interactive Flow

1. Choose a sample video to process or run live, and whether to save the output.

2. Perform squats while the overlay shows phase, rep counts, and feedback. Hold still briefly to allow stable analysis when needed.

### Understanding the Feedback

- **Phase pill**: current phase with a muted color (green for ascending, amber for descending, yellow for bottom, gray for standing)
- **Rep bar (top)**: total reps, correct, and incorrect
- **Feedback "pills"**: per-line messages for back/depth/knees, severity-colored
- **Recommendations**: concise, actionable suggestions; may persist briefly for readability

## Controls

While processing:

- `q` – Quit
- `p` – Pause/Resume

## Form Analysis Features

### Back Analysis

- **Back Rounding Detection**: Measures spine angle and alerts for excessive forward lean
- **Shoulder Level**: Ensures shoulders remain level during movement
- **Chest Position**: Monitors chest-up position for proper form

### Knee Analysis

- **Knee-to-Toe Alignment**: Ensures knees track over toes
- **Valgus Detection**: Alerts when knees cave inward
- **Knee Stability**: Monitors knee position throughout movement

### Depth Analysis

- **Squat Depth**: Ensures proper depth (thighs parallel to ground)
- **Depth Consistency**: Monitors consistent depth across repetitions
- **Excessive Depth**: Warns if going too deep (if uncomfortable)

## Visualization Features

### Keypoint Colors

- **Yellow**: Head keypoints (nose, eyes, ears)
- **Red**: Shoulders (critical for back analysis)
- **Magenta**: Arms (elbows, wrists)
- **Green**: Hips (critical for squat analysis)
- **Orange**: Knees (critical for squat analysis)
- **Purple**: Ankles

### Visual Elements

- Enhanced skeleton and connection styling
- Confidence indicators for high-confidence keypoints
- Bottom overlay with rounded, semi-transparent backgrounds
- Unified Phase pill and natural color palette
- Fixed overlay height for consistent UI

## Project Structure

```
movenet-lightning/
├── main.py                    # Main runner (video selection, processing)
├── model.tflite               # MoveNet Lightning model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── helpers/
│   ├── feedback_utils.py      # Unified overlay UI + feedback assembly
│   ├── pose_processor.py      # Pose processing pipeline + overlay draw
│   ├── analyzers/
│   │   └── squat_analyzer.py  # Squat form analysis + phases + reps
│   └── utils/
│       ├── model_utils.py     # Model loading helpers
│       └── visualization_utils.py # Drawing helpers for skeleton/overlays
├── data/
│   └── squat/                 # Sample squat videos
└── output/                    # Saved processed videos (optional)
```

## Configuration

### Adjustable Parameters

In `squat_analyzer.py`, you can modify:

- Angle/depth thresholds for form checks
- Phase debounce frame counts and min-frames per phase
- Keypoint confidence thresholds
- Color palette (muted greens/ambers/grays) for phases and messages in `feedback_utils.py`

### Performance Tips

1. **For Best Form Analysis**:

   - Use side view (camera perpendicular to movement)
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

## Troubleshooting

### Common Issues

1. **"Cannot open webcam"**:

   - Check camera permissions in System Settings
   - Restart Terminal after granting permissions
   - Ensure no other app is using the camera

2. **Poor form detection**:

   - Improve lighting conditions
   - Ensure full body is visible
   - Wear contrasting clothing
   - Position camera at side view

3. **Inaccurate form analysis**:

   - Ensure camera is perpendicular to movement
   - Maintain consistent distance from camera
   - Perform squats in center of frame
   - Check that all keypoints are visible

4. **Slow performance**:

   - Reduce camera resolution
   - Close other applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with `python3 test_squat_analysis.py`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google's MoveNet team for the excellent pose detection models
- TensorFlow and TensorFlow Hub for model hosting
- Fitness community for form analysis insights

## Future Enhancements

- Set tracking and workout summaries
- Form history and progress tracking
- Mobile app version
- Integration with fitness apps
