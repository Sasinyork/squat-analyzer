# MoveNet Lightning - Squat Form Analysis

A powerful real-time pose detection and squat form analysis application using Google's MoveNet Lightning model. This project provides comprehensive feedback on squat form, detecting common issues like back rounding, knee alignment, depth problems, and arm positioning.

## 🚀 Features

### Advanced Squat Form Analysis

- **Back Rounding Detection**: Analyzes spine alignment and detects excessive forward lean
- **Knee Alignment Analysis**: Monitors knee position relative to toes and detects valgus (knees caving in)
- **Squat Depth Monitoring**: Ensures proper depth (thighs parallel to ground)
- **Arm Position Feedback**: Analyzes arm positioning for optimal balance
- **Real-time Form Scoring**: Provides 0-100 form score with color-coded feedback

### Pose Detection Features

- **Real-time Webcam Processing**: Live pose detection with form analysis
- **Video File Processing**: Analyze pre-recorded videos with comprehensive feedback
- **Enhanced Keypoint Visualization**: Improved skeleton rendering with color-coded body parts
- **Adaptive Thresholds**: Different sensitivity levels for different body parts

### User Interface Improvements

- **Bottom-aligned Feedback**: Form analysis displayed at bottom of screen
- **Comprehensive Information**: Form score, phase detection, issues, and recommendations
- **Color-coded Feedback**: Green (good), Orange (needs improvement), Red (issues)
- **Real-time Recommendations**: Actionable tips for form improvement

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

### Step-by-Step Guide

1. **Select Analysis Mode**:

   ```
   Available Options:
   1. Squat Form Analysis (Webcam)
   2. Squat Form Analysis (Video File)
   3. Basic Pose Detection (Webcam)
   4. Basic Pose Detection (Video File)
   ```

2. **For Squat Form Analysis**:

   - Position yourself for squats (side view recommended)
   - Ensure good lighting and full body visibility
   - Hold still for 2-3 seconds to get initial feedback
   - Perform squats while the system analyzes your form
   - Watch for real-time feedback and recommendations

3. **Understanding the Feedback**:

   - **Form Score**: 0-100 rating of your overall form
   - **Phase Detection**: Standing, Descending, Bottom, Ascending
   - **Issues**: Specific form problems detected
   - **Recommendations**: Actionable tips for improvement

## 🎮 Controls

### Webcam Mode

- **'q'**: Quit the application
- **Real-time feedback**: Continuous form analysis and recommendations

### Video Processing Mode

- **'q'**: Stop processing early
- **Progress updates**: Displayed every 30 frames

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

- **Enhanced Skeleton**: Thicker, more visible connections
- **Confidence Indicators**: Green rings around high-confidence keypoints
- **Color-coded Feedback**: Bottom overlay with comprehensive information
- **Real-time Scoring**: Live form score updates

## 📁 Project Structure

```
movenet-lightning/
├── main.py                 # Main application script
├── test_squat_analysis.py  # Test script for system verification
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── helpers/
│   ├── __init__.py
│   ├── model_utils.py     # Model loading utilities
│   ├── squat_analyzer.py  # Squat form analysis engine
│   ├── feedback_utils.py  # Comprehensive feedback system
│   ├── pose_processor.py  # Pose detection processing
│   └── visualization_utils.py  # Enhanced visualization functions
├── data/                  # Sample data directory
└── output/               # Output directory for processed videos
```

## 🔧 Configuration

### Adjustable Parameters

In the squat analyzer (`helpers/squat_analyzer.py`), you can modify:

- **Angle Thresholds**: Back rounding detection sensitivity
- **Depth Thresholds**: Squat depth requirements
- **Alignment Thresholds**: Knee-to-toe alignment tolerance
- **Confidence Thresholds**: Keypoint detection sensitivity

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

Run the test script to verify all components work:

```bash
python3 test_squat_analysis.py
```

This will test:

- Squat form analyzer
- Feedback system
- Visualization components

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

- Support for other exercises (deadlift, bench press)
- Rep counting and set tracking
- Form history and progress tracking
- Mobile app version
- Integration with fitness apps
