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
                    x,y,z = lm.x, lm.y, lm.z
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                     #           0.7, (0, 255, 0), 1)
                    
                    #print(id,x,y)
                    face.append([x,y,z])
                faces.append(face)
        return img, faces

def rotation_matrix_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rotation_matrix_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_matrix_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])



def main():

    detector = FaceMeshDetector(maxFaces=2)
    faces = None
    #img = Image.open('img/Haimap.jpg').convert("RGB")

    img = Image.open('FGNET/images/049A05.jpg').convert("RGB")
    img= np.array(img)
    print(img)
    
    img, faces = detector.findFaceMesh(img)
    
    pts = np.array(np.array(faces[0].copy()).astype(np.float64))
    #pts = pts[:,0:2]
    print("PTS",pts)
    
    fig = plt.figure()


    plt.imshow(img)


    cameraMatrix = camera_matrix = np.array([
                                            [409, 0.0, 504 / 2],  # Principal point at the image center
                                            [0.0, 409, 409 / 2],
                                            [0.0, 0.0, 1.0]
                                        ], dtype=np.float64)

    dist_matrix = np.zeros((4,1),dtype=np.float64)

    
            


    # print("RPTS: ",r_pts)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(0, 0.7)
    # ax.set_ylim(0, 0.7)
    # ax.set_zlim(-0.35, 0.35)
    # #ax.axis('equal')
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:,2] )
    # ax.scatter(r_pts[:, 0], r_pts[:, 1], r_pts[:,2] )
    # plt.show()


    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(0, 0.7)
    # ax.set_ylim(0, 0.7)
    # ax.set_zlim(-0.35, 0.35)
    # ax.axis('equal')

    
    center_axis = (0,1,4,5,6,8,9,10)
    ear = (234,454)
    face_landmarks = tuple(set([i for i in range(0,468)]) - set(center_axis) - set(ear))

    # ax.scatter(pts[face_landmarks, 0], pts[face_landmarks, 1], pts[face_landmarks,2])
    # ax.scatter(pts[center_axis, 0], pts[center_axis, 1], pts[center_axis,2])
    # ax.scatter(pts[ear, 0], pts[ear, 1], pts[ear,2])

    # ax.scatter(pts[19:30, 1], pts[19:30, 0], pts[19:30,2])


    plt.show()

    print()
    print(pts.shape)
    print(np.min(pts[:,0]),"--->",np.max(pts[:,0]))
    print(np.min(pts[:,1]),"--->",np.max(pts[:,1]))
    print(np.min(pts[:,2]),"--->",np.max(pts[:,2]))
    

    

    # theta = np.pi/4
    # Rx = rotation_matrix_x(theta)

    # # Example: Rotation around the y-axis by 30 degrees (pi/6 radians)
    # theta = np.pi/6
    # Ry = rotation_matrix_y(theta)

    # # Example: Rotation around the z-axis by 60 degrees (pi/3 radians)
    # theta = np.pi/3
    # Rz = rotation_matrix_z(theta)

    # r_pts = pts@Rx@Ry@Rz
    # print(r_pts)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(0, 0.7)
    # ax.set_ylim(0, 0.7)
    # ax.set_zlim(-0.35, 0.35)
    # #ax.axis('equal')
    # ax.scatter(r_pts[:, 1], r_pts[:, 0], r_pts[:,2] )

    # print()
    # print(pts.shape)
    # print(np.min(r_pts[:,0]),"--->",np.max(r_pts[:,0]))
    # print(np.min(r_pts[:,1]),"--->",np.max(r_pts[:,1]))
    # print(np.min(r_pts[:,2]),"--->",np.max(r_pts[:,2]))
    
    #plt.show()


   
    # ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(pts, pts[:,0:2], (321,386), None, None)
    # print(rvecs)


    #(success, rotation_vector, translation_vector) = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)


if __name__ == "__main__":
    main()


# 0.48479944467544556 ---> 0.6728165149688721
# 0.20297002792358398 ---> 0.33148372173309326
# -0.0396420881152153 ---> 0.1087481901049614

# 0.23894456028938293 ---> 0.7651523351669312
# 0.31959354877471924 ---> 0.775615394115448
# -0.1243928000330925 ---> 0.26527050137519836