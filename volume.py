import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # Mirror the image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Thumb tip (4) and index tip (8)
            x1, y1 = lm_list[4]
            x2, y2 = lm_list[8]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw landmarks
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

            # Distance between thumb and index finger
            length = np.hypot(x2 - x1, y2 - y1)

            # Convert distance to volume
            vol = np.interp(length, [20, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Display volume bar
            vol_bar = np.interp(length, [20, 200], [400, 150])
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
            cv2.putText(img, f'{int(np.interp(length, [20, 200], [0, 100]))} %', 
                        (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()