import cv2
import mediapipe as mp
import pyautogui
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# ---------------- Hand Detector ----------------
class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=maxHands,
                                        min_detection_confidence=detectionCon,
                                        min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            h,w,_=img.shape
            for id,lm in enumerate(myHand.landmark):
                cx,cy=int(lm.x*w), int(lm.y*h)
                lmList.append((id,cx,cy))
        return lmList

    def fingersUp(self,lmList):
        fingers=[]
        if lmList:
            fingers.append(1 if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0]-1][1] else 0)
            for id in range(1,5):
                fingers.append(1 if lmList[self.tipIds[id]][2]< lmList[self.tipIds[id]-2][2] else 0)
        return fingers
# ---------------- Body Detector ----------------
class BodyDetector:
    def __init__(self,detectionCon=0.7,trackCon=0.7):
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(min_detection_confidence=detectionCon,
                                   min_tracking_confidence=trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findPose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        return img

    def drawSkeleton(self,img,draw=True):
        if self.results.pose_landmarks and draw:
            for connection in mp.solutions.pose.POSE_CONNECTIONS:
                start_idx,end_idx=connection
                h,w,_=img.shape
                lm_start=self.results.pose_landmarks.landmark[start_idx]
                lm_end=self.results.pose_landmarks.landmark[end_idx]
                x1,y1=int(lm_start.x*w),int(lm_start.y*h)
                x2,y2=int(lm_end.x*w),int(lm_end.y*h)
                cv2.line(img,(x1,y1),(x2,y2),(0,255,255),2)
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                x,y=int(lm.x*w),int(lm.y*h)
                color=(0,0,255) if id<11 else (0,255,0)
                cv2.circle(img,(x,y),6,color,-1)

    def getLandmarks(self,img):
        lmList=[]
        if self.results.pose_landmarks:
            h,w,_=img.shape
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append((id,cx,cy))
        return lmList

# ---------------- Gesture Controller ----------------
class GestureController:
    def __init__(self):
        self.screen_w,self.screen_h=pyautogui.size()
        devices=AudioUtilities.GetSpeakers()
        interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
        self.volume=cast(interface,POINTER(IAudioEndpointVolume))
        self.clickCooldown=0.3
        self.lastClickTime=0
        self.prevMouseX,self.prevMouseY=0,0
        self.smoothFactor=3
        self.prevVol=-1
        self.cursorTrail=[]

    def moveMouse(self,x,y,frame_w,frame_h):
        screenX=int((x/frame_w)*self.screen_w)
        screenY=int((y/frame_h)*self.screen_h)
        smoothX=self.prevMouseX + (screenX-self.prevMouseX)/self.smoothFactor
        smoothY=self.prevMouseY + (screenY-self.prevMouseY)/self.smoothFactor
        pyautogui.moveTo(smoothX,smoothY)
        self.prevMouseX,self.prevMouseY=smoothX,smoothY
        self.cursorTrail.append((int(smoothX),int(smoothY)))
        if len(self.cursorTrail)>20:
            self.cursorTrail.pop(0)

    def click(self):
        if time.time()-self.lastClickTime>self.clickCooldown:
            pyautogui.click()
            self.lastClickTime=time.time()

    def controlVolume(self,x1,y1,x2,y2):
        length=math.hypot(x2-x1,y2-y1)
        vol=(length-20)/200
        vol=max(0,min(1,vol))
        if abs(vol-self.prevVol)>0.02:
            self.volume.SetMasterVolumeLevelScalar(vol,None)
            self.prevVol=vol
        return int(vol*100)

    def drawCursorTrail(self,img):
        for i,(x,y) in enumerate(self.cursorTrail):
            alpha=(i+1)/len(self.cursorTrail)
            cv2.circle(img,(x,y),int(20*alpha),(0,0,255),-1)

# ---------------- Main Loop ----------------
def main():
    cap=cv2.VideoCapture(0)
    handDetector=HandDetector(maxHands=1,detectionCon=0.8,trackCon=0.8)
    bodyDetector=BodyDetector(detectionCon=0.7,trackCon=0.7)
    controller=GestureController()
    pTime=0

    # Make window resizable and support maximize/minimize
    cv2.namedWindow("Tron-Style Full-Body Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tron-Style Full-Body Controller", 1280, 720)

    while True:
        success,img=cap.read()
        if not success: continue
        img=cv2.flip(img,1)
        h,w,_=img.shape

        # Detect body
        img=bodyDetector.findPose(img)
        bodyDetector.drawSkeleton(img)
        bodyLms=bodyDetector.getLandmarks(img)

        # Detect hand
        img=handDetector.findHands(img)
        lmList=handDetector.findPosition(img)

        # Dashboard overlay
        overlay=np.zeros_like(img, dtype=np.uint8)
        cv2.rectangle(overlay,(0,0),(400,120),(0,0,0),-1)
        img=cv2.addWeighted(overlay,0.5,img,0.5,0)

        # Hand gestures
        if lmList:
            fingers=handDetector.fingersUp(lmList)
            totalFingers=fingers.count(1)
            cv2.putText(img,f'Fingers: {totalFingers}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # Mouse move
            if fingers[1]==1 and fingers[2]==0:
                x,y=lmList[8][1],lmList[8][2]
                controller.moveMouse(x,y,w,h)
                controller.drawCursorTrail(img)
                cv2.circle(img,(x,y),20,(0,0,255),cv2.FILLED)
                cv2.putText(img,'Move',(x+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

            # Click
            if fingers[1]==1 and fingers[2]==1:
                controller.click()
                cv2.putText(img,'CLICK!',(lmList[8][1]+20,lmList[8][2]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # Volume
            if fingers[0]==1 and fingers[1]==1:
                x1,y1=lmList[4][1],lmList[4][2]
                x2,y2=lmList[8][1],lmList[8][2]
                volPercent=controller.controlVolume(x1,y1,x2,y2)
                cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
                cv2.circle(img,(x2,y2),10,(255,0,0),cv2.FILLED)
                # Circular volume bar
                center=(70,170)
                radius=45
                cv2.ellipse(img,center,(radius,radius),-90,0,int(360*volPercent/100),(0,255,255),10)
                cv2.putText(img,f'Volume: {volPercent}%',(10,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
                cv2.putText(img,'Volume Control',(x1-40,y1-40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        # FPS
        cTime=time.time()
        fps=1/(cTime-pTime) if cTime!=pTime else 0
        pTime=cTime
        cv2.putText(img,f'FPS: {int(fps)}',(w-120,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.putText(img,'Next-Level Tron UI',(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # Show image
        cv2.imshow("Tron-Style Full-Body Controller",img)

        # Handle ESC key or window close
        key=cv2.waitKey(1) & 0xFF
        if key==27 or cv2.getWindowProperty("Tron-Style Full-Body Controller", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()



