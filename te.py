import cv2
import mediapipe as mp

# Initialize MediaPipe Face and Pose modules
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

# Initialize the MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)  # Use the desired camera index (e.g., 0 for the default camera)

# Initialize Face Detection and Pose Estimation models
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
pose_estimation = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


while True:
    success, frame = cap.read()
    if not success:
        continue

    # Convert the frame to RGB format for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    face_results = face_detection.process(frame_rgb)

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Perform pose estimation using the face region
            face_center = (x + w // 2, y + h // 2)
            frame_pose = frame[y:y + h, x:x + w]
            frame_pose_rgb = cv2.cvtColor(frame_pose, cv2.COLOR_BGR2RGB)

            # Perform pose estimation on the face region
            pose_results = pose_estimation.process(frame_pose_rgb)

            if pose_results.pose_landmarks:
                # You can access pose landmarks and orientation information here
                landmarks = pose_results.pose_landmarks
                # Extract relevant pose landmarks and calculate face orientation

                # Draw landmarks and orientation lines
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Face Orientation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
