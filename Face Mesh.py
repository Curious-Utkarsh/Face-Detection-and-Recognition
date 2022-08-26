import cv2
print(cv2.__version__)
import numpy as np

width=1280
height=720
bright = 180

cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_BRIGHTNESS,bright)
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

class mpFaceMesh:
    import mediapipe as mp
    def __init__(self,maxFaces=2,tol1=.5,tol2=.5,drawMesh=True):
        self.faceMesh = self.mp.solutions.face_mesh.FaceMesh(False,maxFaces,tol1,tol2)
        self.mpDraw = self.mp.solutions.drawing_utils
        self.draw = drawMesh
    def parseLandMarks(self,frame):
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.faceMesh.process(frameRGB)
        faceMeshLandmarks = []
        if results.multi_face_landmarks != None:
            for faceLandmarks in results.multi_face_landmarks:
                faceMeshLandmark = []
                if self.draw == True:
                    self.mpDraw.draw_landmarks(frame,faceLandmarks,self.mp.solutions.face_mesh.FACE_CONNECTIONS)
                for LandMark in faceLandmarks.landmark:
                    X=int(LandMark.x*width)
                    Y=int(LandMark.y*height)
                    faceMeshLandmark.append((X,Y))
                faceMeshLandmarks.append(faceMeshLandmark)
        return faceMeshLandmarks             

findFaceMesh=mpFaceMesh(maxFaces=2,drawMesh=True)

while True:
    ignore,frame=cam.read()
    x = np.zeros([height,width,3], dtype=np.uint8)
    x[:,:]=(0,0,0)
    faceMeshLandmarks=findFaceMesh.parseLandMarks(frame)
    if(len(faceMeshLandmarks) != 0):
        for oneFaceLandmarks in faceMeshLandmarks:
            for faceMark in oneFaceLandmarks:
                cv2.circle(x,faceMark,1,(255,255,255),-1)
    cv2.imshow('my WEBcam',x)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()    

