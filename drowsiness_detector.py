import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20

# Load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark positions
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        coords = np.zeros((68, 2), dtype="int")

        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            flag += 1
            if flag >= FRAME_CHECK:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            flag = 0

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
