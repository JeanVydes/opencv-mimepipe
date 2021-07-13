import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

widthCap = 1440
heightCap = 650

vCap = cv2.VideoCapture(0)
vCap.set(3, widthCap)
vCap.set(4, heightCap)

while True:
    _, frame = vCap.read()

    height, width, _ = frame.shape

    result = face_mesh.process(frame)
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                point = facial_landmarks.landmark[i]
                pointX = int(point.x * width)
                pointY = int(point.y * height)

                cv2.circle(frame, (pointX, pointY), 2, (176,224,230), -1) 

    cv2.imshow("Face Mesk", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.DestroyAllWindows()