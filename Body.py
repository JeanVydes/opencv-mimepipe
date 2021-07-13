import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

videoWidth = 1440
videoHeight = 640

videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, videoWidth)
videoCapture.set(4, videoHeight)

while True:
    _, frame = videoCapture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = frame.shape

    with mp_pose.Pose() as pose_tracker:
        result = pose_tracker.process(frame)
        if result.pose_landmarks is not None:
            for i in result.pose_landmarks.landmark:
                landmarkX = int(i.x * width)
                landmarkY = int(i.y * height)

                cv2.circle(frame, (landmarkX, landmarkY), 8, (0, 255, 0), -1)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(frame)
        if not result.detections:
            continue

        for detection in result.detections:
            mp_drawing.draw_detection(frame, detection)



    cv2.imshow("Hello!", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.DestroyAllWindows()
