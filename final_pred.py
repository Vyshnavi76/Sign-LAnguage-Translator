import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import pyttsx3
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import traceback

# ------------------- Initialization -------------------
model = load_model("cnn8grps_rad1_model.h5")
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # female voice

# Global variables
current_symbol = ''
sentence = ''
suggestions = ["", "", "", ""]

# Tkinter Window
root = tk.Tk()
root.title("Sign Language to Text Converter")
root.geometry("1000x700")
root.configure(bg="#f4f4f4")

camera_panel = Label(root, bg="black")
camera_panel.place(x=20, y=20, width=400, height=300)

hand_panel = Label(root, bg="black")
hand_panel.place(x=450, y=20, width=400, height=300)

label_char = Label(root, text="Character: ", font=("Arial", 16), bg="#f4f4f4")
label_char.place(x=20, y=350)

label_sentence = Label(root, text="Sentence: ", font=("Arial", 16), bg="#e81313")
label_sentence.place(x=20, y=400)

# Suggestion buttons
btns = []
for i in range(4):
    b = Button(root, text="", font=("Arial", 14), command=lambda i=i: choose_suggestion(i))
    b.place(x=20 + (i * 150), y=450, width=140, height=40)
    btns.append(b)

btn_speak = Button(root, text="Speak", font=("Arial", 14), command=lambda: speak_fun())
btn_speak.place(x=20, y=520, width=140, height=40)

btn_clear = Button(root, text="Clear", font=("Arial", 14), command=lambda: clear_fun())
btn_clear.place(x=180, y=520, width=140, height=40)

# Hand detectors
hd = HandDetector(maxHands=1, detectionCon=0.8)
hd2 = HandDetector(maxHands=1, detectionCon=0.8)

# Camera
cap = cv2.VideoCapture(0)
offset = 15

# ------------------- Functions -------------------

def speak_fun():
    global sentence
    engine.say(sentence)
    engine.runAndWait()

def clear_fun():
    global sentence, suggestions
    sentence = ""
    suggestions = ["", "", "", ""]
    update_labels()

def choose_suggestion(idx):
    global sentence, suggestions
    words = sentence.split()
    if words:
        words[-1] = suggestions[idx]
        sentence = " ".join(words) + " "
    update_labels()

def update_labels():
    label_char.config(text=f"Character: {current_symbol}")
    label_sentence.config(text=f"Sentence: {sentence}")
    for i in range(4):
        btns[i].config(text=suggestions[i])

def predict(white_img):
    """
    Takes the white image with drawn landmarks and returns predicted char.
    """
    img_resized = cv2.resize(white_img, (400, 400))
    img_input = np.expand_dims(img_resized, axis=0)  # shape (1,400,400,3)
    pred = model.predict(img_input)
    class_idx = np.argmax(pred[0])

    # Map indices to your gesture characters (update with your classes)
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # update as per your model
    return classes[class_idx]

def draw_skeleton_on_white(lmList, w, h):
    """
    Draws the skeleton exactly like your data collection script on white canvas.
    """
    white = np.ones((400, 400, 3), dtype=np.uint8) * 255
    os_x = ((400 - w) // 2) - offset
    os_y = ((400 - h) // 2) - offset

    # Draw finger lines
    for t in range(0,4):
        cv2.line(white, (lmList[t][0]+os_x, lmList[t][1]+os_y),
                 (lmList[t+1][0]+os_x, lmList[t+1][1]+os_y), (0,255,0), 3)
    for t in range(5,8):
        cv2.line(white, (lmList[t][0]+os_x, lmList[t][1]+os_y),
                 (lmList[t+1][0]+os_x, lmList[t+1][1]+os_y), (0,255,0), 3)
    for t in range(9,12):
        cv2.line(white, (lmList[t][0]+os_x, lmList[t][1]+os_y),
                 (lmList[t+1][0]+os_x, lmList[t+1][1]+os_y), (0,255,0), 3)
    for t in range(13,16):
        cv2.line(white, (lmList[t][0]+os_x, lmList[t][1]+os_y),
                 (lmList[t+1][0]+os_x, lmList[t+1][1]+os_y), (0,255,0), 3)
    for t in range(17,20):
        cv2.line(white, (lmList[t][0]+os_x, lmList[t][1]+os_y),
                 (lmList[t+1][0]+os_x, lmList[t+1][1]+os_y), (0,255,0), 3)

    # Palm lines
    cv2.line(white, (lmList[5][0]+os_x, lmList[5][1]+os_y), (lmList[9][0]+os_x, lmList[9][1]+os_y), (0,255,0),3)
    cv2.line(white, (lmList[9][0]+os_x, lmList[9][1]+os_y), (lmList[13][0]+os_x, lmList[13][1]+os_y), (0,255,0),3)
    cv2.line(white, (lmList[13][0]+os_x, lmList[13][1]+os_y), (lmList[17][0]+os_x, lmList[17][1]+os_y), (0,255,0),3)
    cv2.line(white, (lmList[0][0]+os_x, lmList[0][1]+os_y), (lmList[5][0]+os_x, lmList[5][1]+os_y), (0,255,0),3)
    cv2.line(white, (lmList[0][0]+os_x, lmList[0][1]+os_y), (lmList[17][0]+os_x, lmList[17][1]+os_y), (0,255,0),3)

    # Draw points
    for i in range(21):
        cv2.circle(white, (lmList[i][0]+os_x, lmList[i][1]+os_y), 2, (0,0,255), 1)

    return white

def video_loop():
    global current_symbol, sentence, suggestions

    try:
        ret, frame = cap.read()
        if not ret:
            root.after(10, video_loop)
            return
        frame = cv2.flip(frame, 1)
        hands, img = hd.findHands(frame, draw=False)
        current_symbol = ""

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            img_crop = frame[y-offset:y+h+offset, x-offset:x+w+offset]

            if img_crop.size != 0:
                hands2, _ = hd2.findHands(img_crop, draw=False)
                if hands2:
                    lmList = hands2[0]['lmList']
                    white_skel = draw_skeleton_on_white(lmList, w, h)

                    # Predict character
                    char = predict(white_skel)
                    current_symbol = char

                    # Update sentence if new char
                    if len(sentence) == 0 or (len(sentence) > 0 and char != sentence[-1]):
                        sentence += char

                    # Update suggestions (dummy example)
                    suggestions = [char*2, char*3, char*4, char*5]

                    # Show skeleton panel
                    white_rgb = cv2.cvtColor(white_skel, cv2.COLOR_BGR2RGB)
                    white_pil = ImageTk.PhotoImage(Image.fromarray(white_rgb))
                    hand_panel.configure(image=white_pil)
                    hand_panel.image = white_pil

        # Show camera panel
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        camera_panel.configure(image=img_pil)
        camera_panel.image = img_pil

        update_labels()
        root.after(10, video_loop)

    except Exception:
        print("==", traceback.format_exc())
        root.after(10, video_loop)

# ------------------- Main Loop -------------------
video_loop()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()
