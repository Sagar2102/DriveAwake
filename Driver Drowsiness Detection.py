from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import time
import dlib
import cv2
import numpy as np
import sys

from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# Platform-specific beep
if sys.platform.startswith('win'):
    import winsound
    def beep_alert(freq=1000, dur=200):
        winsound.Beep(freq, dur)
else:
    def beep_alert(freq=1000, dur=200):
        print('\a', end='', flush=True)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat'
)

print("[INFO] searching for camera...")
valid_idx = None
for idx in range(5):
    cap_t = cv2.VideoCapture(idx)
    if cap_t.isOpened():
        valid_idx = idx
        cap_t.release()
        print(f"[INFO] camera found at index {idx}")
        break
    cap_t.release()

if valid_idx is None:
    raise RuntimeError("No webcam detected. Connect a camera and restart.")

print("[INFO] initializing video stream...")
vs = VideoStream(src=valid_idx).start()
time.sleep(2.0)

frame_w, frame_h = 1024, 576
image_points = np.array([
    (359, 391), (399, 561), (337, 297),
    (513, 301), (345, 465), (453, 469)
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
EYE_AR_THRESH = 0.15
MOUTH_AR_THRESH = 0.79

# Blink detection timing variables
BLINK_TIME_THRESH = 1.5  # seconds
eyes_closed_start_time = None

(mStart, mEnd) = (49, 68)

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)
    cv2.putText(frame, f"{len(rects)} face(s) found", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eyes
        leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        cv2.drawContours(frame, [cv2.convexHull(leftEye), cv2.convexHull(rightEye)],
                         -1, (0, 255, 0), 1)

        # Eye closure timing and continuous beep/message
        if ear < EYE_AR_THRESH:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elapsed = time.time() - eyes_closed_start_time
            if elapsed >= BLINK_TIME_THRESH:
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                beep_alert()
        else:
            eyes_closed_start_time = None

        # Mouth
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (650, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            beep_alert(freq=1500, dur=300)

        for i, (x, y) in enumerate(shape):
            color = (0, 255, 0) if i in (33, 8, 36, 45, 48, 54) else (0, 0, 255)
            cv2.circle(frame, (x, y), 1, color, -1)
            if i in (33, 8, 36, 45, 48, 54):
                image_points[{33:0, 8:1, 36:2, 45:3, 48:4, 54:5}[i]] = np.array([x, y], dtype='double')

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

        head_tilt, sp, ep, ep_alt = getHeadTiltAndCoords(size, image_points, frame_h)
        cv2.line(frame, sp, ep, (255, 0, 0), 2)
        cv2.line(frame, sp, ep_alt, (0, 0, 255), 2)
        if head_tilt:
            cv2.putText(frame, f'Head Tilt: {head_tilt[0]}°', (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
