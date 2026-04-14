import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# ---------------- Initialization ----------------
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Dataset directory (update if needed)
dataset_path = "D:\\sign2text_dataset_3.0\\AtoZ_3.1\\"

# Start with letter A
c_dir = 'A'
os.makedirs(os.path.join(dataset_path, c_dir), exist_ok=True)
count = len(os.listdir(os.path.join(dataset_path, c_dir)))

offset = 15
step = 1
flag = False   # True → collecting images
suv = 0        # how many collected for current letter

# White background for skeleton
white_base = np.ones((400, 400, 3), np.uint8) * 255

# ---------------- Main Loop ----------------
while True:
    try:
        ret, frame = capture.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)

        # Reset fresh white canvas each loop
        white = white_base.copy()
        skeleton1 = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Safe crop (avoid negative slicing errors)
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(frame.shape[1], x + w + offset)
            y2 = min(frame.shape[0], y + h + offset)
            image = frame[y1:y2, x1:x2]

            # Detect again inside cropped region
            handz, imz = hd2.findHands(image, draw=False, flipType=True)
            if handz:
                pts = handz[0]['lmList']
                osx = ((400 - w) // 2) - 15
                osy = ((400 - h) // 2) - 15

                # Draw skeleton lines
                for t in range(0, 4):
                    cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy),
                             (pts[t+1][0] + osx, pts[t+1][1] + osy), (0, 255, 0), 3)
                for t in range(5, 8):
                    cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy),
                             (pts[t+1][0] + osx, pts[t+1][1] + osy), (0, 255, 0), 3)
                for t in range(9, 12):
                    cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy),
                             (pts[t+1][0] + osx, pts[t+1][1] + osy), (0, 255, 0), 3)
                for t in range(13, 16):
                    cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy),
                             (pts[t+1][0] + osx, pts[t+1][1] + osy), (0, 255, 0), 3)
                for t in range(17, 20):
                    cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy),
                             (pts[t+1][0] + osx, pts[t+1][1] + osy), (0, 255, 0), 3)

                # Palm lines
                cv2.line(white, (pts[5][0] + osx, pts[5][1] + osy),
                         (pts[9][0] + osx, pts[9][1] + osy), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + osx, pts[9][1] + osy),
                         (pts[13][0] + osx, pts[13][1] + osy), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + osx, pts[13][1] + osy),
                         (pts[17][0] + osx, pts[17][1] + osy), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + osx, pts[0][1] + osy),
                         (pts[5][0] + osx, pts[5][1] + osy), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + osx, pts[0][1] + osy),
                         (pts[17][0] + osx, pts[17][1] + osy), (0, 255, 0), 3)

                # Draw landmark points
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + osx, pts[i][1] + osy), 2, (0, 0, 255), 1)

                skeleton1 = white.copy()
                cv2.imshow("Skeleton", skeleton1)

        # Overlay text
        frame = cv2.putText(frame, f"Letter={c_dir}  Count={count}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        # Key controls
        interrupt = cv2.waitKey(1)

        if interrupt & 0xFF == 27:  # Esc → Exit
            break

        if interrupt & 0xFF == ord('n'):  # Next letter
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):
                c_dir = 'A'
            os.makedirs(os.path.join(dataset_path, c_dir), exist_ok=True)
            count = len(os.listdir(os.path.join(dataset_path, c_dir)))
            flag = False
            suv = 0

        if interrupt & 0xFF == ord('a'):  # Toggle collection
            flag = not flag
            suv = 0

        # Save images if collecting
        if flag and skeleton1 is not None:
            if suv >= 180:   # Stop after 180 samples
                flag = False
            else:
                if step % 3 == 0:  # Save every 3rd frame
                    save_path = os.path.join(dataset_path, c_dir, f"{count}.jpg")
                    cv2.imwrite(save_path, skeleton1)
                    count += 1
                    suv += 1
                step += 1

    except Exception:
        print("==", traceback.format_exc())

# Cleanup
capture.release()
cv2.destroyAllWindows()
