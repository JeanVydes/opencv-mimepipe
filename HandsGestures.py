import cv2
import time
import numpy as np
import math
import mediapipe as mp
import screen_brightness_control as sbc
from subprocess import call

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

widthCam = 1200
heightCam = 520

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam) 

pTime = 0


def volume(lmList):
    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]

    cx, cy = (x1+x2) // 2, (y1 + y2) // 2

    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)
    if length < 25:
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        _ = call(["amixer", "-D", "pulse", "sset", "Master", "0%"])
    else:
        _ = call(["amixer", "-D", "pulse", "sset", "Master", str(length/2.5)+"%"])

    return


while(True):
    _, frame = cap.read()

    hands = mp_hands.Hands()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    xList = []
    yList = []
    bbox = []
    lmList = []

    if results.multi_hand_landmarks is None:
        continue

    for handLms in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            lmList.append([id, cx, cy])

        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax

        cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20),
        (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

    img = frame

    lmList = lmList[0]
    if len(lmList) != 0:
        pulgar1, pulgar2 = lmList[4][1], lmList[4][2]
        indice1, indice2 = lmList[8][1], lmList[8][2]
        medio1, medio2 = lmList[12][1], lmList[12][2]
        anular1, anular2 = lmList[16][1], lmList[16][2]
        menique1, menique2 = lmList[20][1], lmList[20][2]

        cv2.circle(img, (pulgar1, pulgar2), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (indice1, indice2), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (medio1, medio2), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (anular1, anular2), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (menique1, menique2), 12, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (pulgar1, pulgar2), (indice1, indice2), (255, 0, 255), 2)
        cv2.line(img, (indice1, indice2), (medio1, medio2), (255, 0, 255), 2)
        cv2.line(img, (medio1, medio2), (anular1, anular2), (255, 0, 255), 2)
        cv2.line(img, (anular1, anular2), (menique1, menique2), (255, 0, 255), 2)


        PulgarIndice1, PulgarIndice2 = (pulgar1 + pulgar2) // 2, (indice1 + indice2) // 2
        IndiceMedio1, IndiceMedio2 = (indice1 + indice2) // 2, (medio1 + medio2) // 2
        MedioAnular1, MedioAnular2 = (medio1 + medio2) // 2, (anular1 + anular2) // 2
        AnularMenique1, AnularMenique2 = (anular1 + anular2) // 2, (menique1 + menique2) // 2

        lengthBetweenPulgarIndice = math.hypot(indice1 - pulgar1, indice2 - pulgar2)
        lengthBetweenIndiceMedio = math.hypot(medio1 - indice1, medio2 - indice2)

        if lengthBetweenPulgarIndice < 25 and lengthBetweenIndiceMedio > 48:
            _ = call(["amixer", "-D", "pulse", "sset", "Master", "0%"])

            cv2.putText(frame, 'Action: volume', (240, 70), cv2.FONT_HERSHEY_COMPLEX, 
                1, (220, 197, 237), 2)
        elif lengthBetweenPulgarIndice > 25 and lengthBetweenIndiceMedio > 65 and lengthBetweenIndiceMedio > 48:
            _ = call(["amixer", "-D", "pulse", "sset", "Master", str(lengthBetweenPulgarIndice/2.5)+"%"])

            cv2.putText(frame, 'Action: volume', (240, 70), cv2.FONT_HERSHEY_COMPLEX, 
                1, (220, 197, 237), 2)
        elif lengthBetweenIndiceMedio < 50:
            percentage = math.hypot(indice1 - pulgar1, indice2 - pulgar2)
            sbc.set_brightness(percentage/2)

            cv2.putText(frame, 'Action: brightness', (240, 70), cv2.FONT_HERSHEY_COMPLEX, 
                1, (220, 197, 237), 2)
        else:
            cv2.putText(frame, 'Action: none', (240, 70), cv2.FONT_HERSHEY_COMPLEX, 
                1, (220, 197, 237), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 
                1, (220, 197, 237), 2)

    cv2.imshow('XXX',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
