import cv2
import time
import numpy as np

camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

no_face_start = None
no_face_alerted = False
multiple_face_alerted = False
absence_count = 0
multiple_face_count = 0
events = []

def add_event(text):
    timestamp = time.strftime("%H:%M:%S")
    events.append(f"{text} {timestamp}")
    if len(events) > 3:
        events.pop(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) == 0:
        if no_face_start is None:
            no_face_start = time.time()
        else:
            elapsed = time.time() - no_face_start
            if elapsed > 3 and not no_face_alerted:
                no_face_alerted = True
                absence_count += 1
                add_event("⚠ No Face Detected")
    else:
        no_face_start = None
        no_face_alerted = False

    if len(faces) > 1:
        if not multiple_face_alerted:
            multiple_face_count += 1
            add_event("⚠ Multiple Faces Detected")
        multiple_face_alerted = True
    else:
        multiple_face_alerted = False

    attention_score = max(0, 100 - (absence_count*10 + multiple_face_count*5))

    dashboard = np.zeros((480, 300, 3), dtype=np.uint8)
    dashboard[:] = (50, 50, 50)

    cv2.putText(dashboard, f"Absence Count: {absence_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(dashboard, f"Multiple Faces: {multiple_face_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(dashboard, "Attention", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.rectangle(dashboard, (10, 170), (210, 200), (255,255,255), 2)
    cv2.rectangle(dashboard, (10, 170), (10 + int(attention_score*2), 200), (0, 255, 0), cv2.FILLED)

    y_pos = 230
    if no_face_alerted:
        cv2.rectangle(dashboard, (10, y_pos), (290, y_pos+40), (0,0,255), cv2.FILLED)
        cv2.putText(dashboard, "⚠ No Face Detected!", (15, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_pos += 60
    if multiple_face_alerted:
        cv2.rectangle(dashboard, (10, y_pos), (290, y_pos+40), (0,0,255), cv2.FILLED)
        cv2.putText(dashboard, "⚠ Multiple Faces!", (15, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_pos += 60

    cv2.putText(dashboard, "Recent Events:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    y_pos += 30
    for event in events:
        cv2.putText(dashboard, event, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y_pos += 25

    combined_frame = np.hstack((frame, dashboard))
    cv2.imshow("Smart Proctoring System", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
