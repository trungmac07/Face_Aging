import cv2
import mediapipe as mp
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image



class FaceMeshDetector():

    def __init__(self, staticMode=True, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y,z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                     #           0.7, (0, 255, 0), 1)

                    #print(id,x,y)
                    face.append([x,y,z])
                faces.append(face)
        return img, faces


def main():

    detector = FaceMeshDetector(maxFaces=2)
    faces = None
    img = Image.open('FGNET/images/073A16.jpg').convert("RGB")
    img=np.array(img)
    print(img)
    
    img, faces = detector.findFaceMesh(img)
    
    pts = np.array(faces[0])
    pts = pts[:,0:2]
    print("PTS",pts)
    
    plt.imshow(img)
    plt.plot(markersize=1)  # og:shorthand for green circle
    plt.axis('equal')
    plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="red", s=1)
    plt.show()



if __name__ == "__main__":
    main()