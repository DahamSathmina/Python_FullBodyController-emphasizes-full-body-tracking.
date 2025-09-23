import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyautogui
import random
from collections import deque

# ----------------------- Hand Detector -----------------------
class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=maxHands,
                                        min_detection_confidence=detectionCon,
                                        min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                lmList.append((id, int(lm.x * w), int(lm.y * h)))
        return lmList

    def fingersUp(self, lmList):
        fingers = []
        if lmList:
            fingers.append(1 if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0]-1][1] else 0)
            for id in range(1,5):
                fingers.append(1 if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id]-2][2] else 0)
        return fingers

# ----------------------- Body Detector -----------------------
class BodyDetector:
    def __init__(self, detectionCon=0.7, trackCon=0.7):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=detectionCon,
                                     min_tracking_confidence=trackCon)
        self.results = None

    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        return img

    def drawSkeleton(self, img):
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for conn in mp.solutions.pose.POSE_CONNECTIONS:
                start, end = conn
                lm_start = self.results.pose_landmarks.landmark[start]
                lm_end = self.results.pose_landmarks.landmark[end]
                cv2.line(img, (int(lm_start.x*w), int(lm_start.y*h)),
                         (int(lm_end.x*w), int(lm_end.y*h)), (0,255,0), 2)

# ----------------------- Gesture Controller -----------------------
class GestureController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prevMouseX, self.prevMouseY = 0,0
        self.smoothFactor = 4
        self.cursorTrail = deque(maxlen=20)
        self.clickCooldown = 0.3
        self.lastClickTime = 0

    def moveMouse(self, x, y, frame_w, frame_h):
        screenX = int((x/frame_w)*self.screen_w)
        screenY = int((y/frame_h)*self.screen_h)
        smoothX = self.prevMouseX + (screenX - self.prevMouseX)/self.smoothFactor
        smoothY = self.prevMouseY + (screenY - self.prevMouseY)/self.smoothFactor
        try:
            pyautogui.moveTo(smoothX, smoothY)
        except Exception:
            pass
        self.prevMouseX, self.prevMouseY = smoothX, smoothY
        self.cursorTrail.append((int(smoothX), int(smoothY)))

    def click(self):
        if time.time() - self.lastClickTime > self.clickCooldown:
            try:
                pyautogui.click()
            except Exception:
                pass
            self.lastClickTime = time.time()

# ----------------------- Camera Thread -----------------------
class CameraThread(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(np.ndarray)
    statsReady = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.handDetector = HandDetector()
        self.bodyDetector = BodyDetector()
        self.controller = GestureController()
        # Face mesh setup
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1,
                                                 min_detection_confidence=0.7,
                                                 min_tracking_confidence=0.7)
        self.drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
        self.frame_w, self.frame_h = 640, 480

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.running = True
        pTime = time.time()

        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Body + hand
            frame = self.bodyDetector.findPose(frame)
            self.bodyDetector.drawSkeleton(frame)
            frame = self.handDetector.findHands(frame)
            lmList = self.handDetector.findPosition(frame)

            # Face mesh overlay
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=faceLms,
                        connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.drawSpec
                    )

            # Hand gestures
            if lmList:
                fingers = self.handDetector.fingersUp(lmList)
                if len(fingers) >= 2:
                    if fingers[1]==1 and fingers[2]==0:
                        x,y = lmList[8][1], lmList[8][2]
                        self.controller.moveMouse(x,y,w,h)
                    if fingers[1]==1 and fingers[2]==1:
                        self.controller.click()

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime

            self.frameReady.emit(frame)
            self.statsReady.emit({"fps": int(fps)})
            QtCore.QThread.msleep(8)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(2000)

# ----------------------- Matrix Background -----------------------
class MatrixEffect:
    def __init__(self, width, height, font_size=12, layers=3):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.layers = []
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*()"
        for i in range(layers):
            columns = self.width // font_size
            drops = [random.randint(0, self.height) for _ in range(columns)]
            speed = random.randint(2,6)
            intensity = 50 + i*50
            self.layers.append({"drops": drops, "speed": speed, "intensity": intensity})

    def draw(self, frame):
        overlay = np.zeros_like(frame)
        for layer in self.layers:
            for i, drop in enumerate(layer["drops"]):
                char = random.choice(self.chars)
                x = i*self.font_size
                y = drop
                color = (0, layer["intensity"], 0)
                cv2.putText(overlay, char, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,1,cv2.LINE_AA)
                layer["drops"][i] += layer["speed"]
                if layer["drops"][i] > self.height:
                    layer["drops"][i] = 0
        return cv2.addWeighted(frame, 0.6, overlay, 0.4,0)

# ----------------------- Floating Hacker Texts -----------------------
class FloatingTexts:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.texts = ["Access Granted", "Decrypting...", "Firewall Bypassed", "LOADING...", "Root Shell Active"]
        self.active = [{"text": random.choice(self.texts),
                        "x": random.randint(0,width),
                        "y": random.randint(0,height),
                        "timer": time.time()} for _ in range(5)]

    def draw(self, frame):
        for t in self.active:
            if time.time() - t["timer"] > 1.5:
                t["text"] = random.choice(self.texts)
                t["x"] = random.randint(0,self.width)
                t["y"] = random.randint(0,self.height)
                t["timer"] = time.time()
            cv2.putText(frame, t["text"], (t["x"], t["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1,cv2.LINE_AA)
        return frame

# ----------------------- Glitch Effect -----------------------
def glitch_effect(frame):
    h,w,_ = frame.shape
    for _ in range(5):
        x = random.randint(0,w-20)
        y = random.randint(0,h-20)
        w_g = random.randint(5,20)
        h_g = random.randint(5,20)
        dx = random.randint(-10,10)
        frame[y:y+h_g, x:x+w_g] = np.roll(frame[y:y+h_g, x:x+w_g], shift=dx, axis=1)
    return frame

# ----------------------- Hacker Window -----------------------
class HackerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hacker UI Body Controller")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: black; color: #00ff00; font-family: monospace;")
        # Set app icon
        self.setWindowIcon(QtGui.QIcon("Assets/icon/icon.png"))  # <-- add your icon path here

        # Main video label
        self.videoLabel = QtWidgets.QLabel()
        self.videoLabel.setStyleSheet("background-color: black; border-radius:8px;")
        self.setCentralWidget(self.videoLabel)

        # HUD Panel
        self.hudLabel = QtWidgets.QLabel(self.videoLabel)
        self.hudLabel.setStyleSheet("background-color: rgba(0,0,0,120); border:2px solid #00ff00; padding:5px;")
        self.hudLabel.move(10,10)
        self.hudLabel.resize(220,80)

        self.cameraThread = CameraThread()
        self.cameraThread.frameReady.connect(self.updateFrame)
        self.cameraThread.statsReady.connect(self.updateStats)
        self.cameraThread.start()

        # Matrix & floating texts
        self.matrix = MatrixEffect(self.videoLabel.width(), self.videoLabel.height())
        self.floatingTexts = FloatingTexts(self.videoLabel.width(), self.videoLabel.height())

    @QtCore.pyqtSlot(np.ndarray)
    def updateFrame(self, frame):
        frame = self.matrix.draw(frame)
        frame = self.floatingTexts.draw(frame)
        frame = glitch_effect(frame)

        # Neon cursor trail
        for i, (x,y) in enumerate(list(self.cameraThread.controller.cursorTrail)[-20:]):
            alpha = (i+1)/20
            overlay = frame.copy()
            cv2.circle(overlay, (x,y), 6, (0,255,0), -1)
            frame = cv2.addWeighted(frame,1,overlay,alpha,0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch*w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.videoLabel.size(), QtCore.Qt.KeepAspectRatio)
        self.videoLabel.setPixmap(pix)

    @QtCore.pyqtSlot(dict)
    def updateStats(self, stats):
        self.hudLabel.setText(f"FPS: {stats.get('fps',0)}\nCursor Trail: {len(self.cameraThread.controller.cursorTrail)}")

    def closeEvent(self, event):
        try:
            self.cameraThread.stop()
        except:
            pass
        event.accept()

# ----------------------- Main -----------------------
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = HackerWindow()
    win.show()
    sys.exit(app.exec_())
