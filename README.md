# Python_FullBodyController-emphasizes-full-body-tracking.

**Full-body hand and gesture tracking system with Tron-style UI**

Control your computer using natural gestures: move the mouse, click, adjust volume, and scroll. Built with Python, MediaPipe, OpenCV, and PyAutoGUI for real-time interaction with a futuristic Tron-style interface.

---

## Features

- Full-body tracking using **MediaPipe Pose** (head, shoulders, elbows, hips, knees, ankles).  
- Hand gesture control using **MediaPipe Hands**:
  - Move mouse with index finger.  
  - Click with index + middle finger.  
  - Adjust volume with thumb + index pinch.  
  - Optional scroll gestures.  
- Smooth cursor control with adjustable sensitivity.  
- Tron-style UI:
  - Glowing skeleton overlay for hands and body.  
  - Animated cursor trail.  
  - On-screen FPS and status display.

---

## Requirements

- Python 3.10 (recommended)  

### Install required libraries

You can install all required libraries at once using `pip`:

```bash
pip install opencv-python mediapipe pyautogui pycaw comtypes numpy
```

Alternatively, install them one by one:

```bash
pip install opencv-python
pip install mediapipe
pip install pyautogui
pip install pycaw
pip install comtypes
pip install numpy
```

> **Tip:** If installation of `mediapipe` fails, make sure you are using Python 3.10 or 3.11.

---

## Usage

Run the application:

```bash
python app.py
```

### Controls

- **Move mouse:** Index finger up, middle finger down  
- **Click:** Index + middle finger together  
- **Volume control:** Pinch thumb + index finger  
- **Exit:** Press `ESC`  

### Tips

- Adjust `smoothFactor` in the code to control cursor responsiveness.  
- Use a well-lit environment and plain background for best tracking performance.  
- Optional: tweak `detectionCon` and `trackCon` in `HandDetector` for detection stability.

---

## Contributing

Contributions are welcome:

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/your-feature`  
3. Commit changes: `git commit -m "Add feature"`  
4. Push branch: `git push origin feature/your-feature`  
5. Open a Pull Request  

---

## Acknowledgements

- **MediaPipe** — Real-time hand and body tracking  
- **OpenCV** — Image processing and visualization  
- **PyAutoGUI** — Mouse and keyboard automation  
- **Pycaw** — Windows audio control  



