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
- `opencv-python`  
- `mediapipe`  
- `pyautogui`  
- `pycaw`  
- `comtypes`  
- `numpy`  

Install dependencies using:

```bash
pip install -r requirements.txt

