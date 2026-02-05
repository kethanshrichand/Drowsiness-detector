# Drowsy Detector

A real-time drowsiness detection system that monitors driver alertness using computer vision and facial landmark analysis.

## Features

- **Eye Closure Detection**: Uses Eye Aspect Ratio (EAR) to detect drowsiness
- **Yawn Detection**: Monitors mouth opening distance
- **Head Nodding Detection**: Tracks vertical head movement
- **Gaze Detection**: Detects when user looks away from screen
- **Engagement Tasks**: Random letter prompts to verify user attention
- **Real-time Alerting**: Text-to-speech warnings and visual alerts
- **Image Capture**: Automatically saves images when drowsiness is detected
- **Logging**: Comprehensive activity logging

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pillow
- Pyttsx3
- Tkinter (built-in)

## Installation

```bash
pip install opencv-python mediapipe numpy pillow pyttsx3
```

## Usage

Run the main application:

```bash
python drowsiness/try2.py
```

## Project Structure

```
├── drowsiness/
│   └── try2.py          # Main application code
├── drowsy_images/       # Captured drowsy images
├── yawn_images/         # Captured yawn images
├── drowsiness_log.txt   # Activity log
├── class_labels.pkl     # Classification labels (if used)
└── drowsiness_model.pkl # Trained model (if used)
```

## License

MIT License