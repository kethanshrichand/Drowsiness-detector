import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyttsx3
import mediapipe as mp
import threading
import time
import os

engine = pyttsx3.init()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_LANDMARKS = [13, 14]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_CONSEC_FRAMES = 18
YAWN_THRESHOLD = 20
YAWN_CONSEC_FRAMES = 20
NO_FACE_LIMIT = 100

COUNTER = 0
yawn_counter = 0
ALARM_ON = False
no_face_counter = 0

earthreshold = 0.28

prev_nose_y = None
nodding_counter = 0
NOD_THRESHOLD = 2

cap = cv2.VideoCapture(0)

app = tk.Tk()
app.title("Drowsy Detector")
app.configure(bg="#1e1e1e")
app.geometry("800x650")
app.resizable(False, False)

title = tk.Label(app, text="Drowsy Detector", font=("Helvetica", 20), bg="#1e1e1e", fg="white")
title.pack(pady=5)

vidFrame = tk.Frame(app, bg="#1e1e1e")
vidFrame.pack(pady=5)

vid = tk.Label(vidFrame, bg="#1e1e1e")
vid.pack()

status_label = tk.Label(app, text="", font=("Helvetica", 14), bg="#1e1e1e", fg="white")
status_label.pack(pady=5)

warning_label = tk.Label(app, text="", font=("Helvetica", 18, "bold"), bg="#1e1e1e", fg="red")
warning_label.pack()

ear_label = tk.Label(app, text="", font=("Helvetica", 12), bg="#1e1e1e", fg="cyan")
ear_label.pack()

ear_thresh_slider = tk.Scale(app, from_=0.15, to=0.30, resolution=0.01, orient="horizontal",
                             label="EAR Threshold", bg="#1e1e1e", fg="white", troughcolor="gray")
ear_thresh_slider.set(0.25)
ear_thresh_slider.pack(pady=5)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

def detect():
    global COUNTER, ALARM_ON, no_face_counter, yawn_counter
    global prev_nose_y, nodding_counter

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    ih, iw, _ = frame.shape

    if results.multi_face_landmarks:
        status_label.config(text="Face(s) detected", fg="lightgreen")
        no_face_counter = 0
        min_x = iw + 1
        leftmost_face = None

        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            x = int(nose.x * iw)
            if x < min_x:
                min_x = x
                leftmost_face = face_landmarks

        if leftmost_face:
            left_eye = []
            right_eye = []

            for idx in LEFT_EYE:
                lm = leftmost_face.landmark[idx]
                left_eye.append(np.array([lm.x * iw, lm.y * ih]))

            for idx in RIGHT_EYE:
                lm = leftmost_face.landmark[idx]
                right_eye.append(np.array([lm.x * iw, lm.y * ih]))

            left_ear = eye_aspect_ratio(np.array(left_eye))
            right_ear = eye_aspect_ratio(np.array(right_eye))
            ear = (left_ear + right_ear) / 2.0

            upper_lip = leftmost_face.landmark[MOUTH_LANDMARKS[0]]
            lower_lip = leftmost_face.landmark[MOUTH_LANDMARKS[1]]
            lip_distance = np.linalg.norm(
                np.array([upper_lip.x * iw, upper_lip.y * ih]) -
                np.array([lower_lip.x * iw, lower_lip.y * ih])
            )

            ear_color = "lightgreen" if ear >= earthreshold else "orange"
            ear_label.config(text=f"EAR: {ear:.2f} | Mouth: {lip_distance:.2f}", fg=ear_color)

            if ear < earthreshold:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES and not ALARM_ON:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    os.makedirs("drowsy_images", exist_ok=True)
                    cv2.imwrite(f"drowsy_images/drowsy_{timestamp}.jpg", frame)
                    with open("drowsiness_log.txt", "a") as f:
                        f.write(f"[{timestamp}] Drowsiness detected\n")
                    speak("Warning! You are drowsy. Please wake up!")
                    warning_label.config(text="DROWSINESS DETECTED!")
                    ALARM_ON = True
                    COUNTER = 0
            else:
                COUNTER = 0
                ALARM_ON = False
                warning_label.config(text="")

            if lip_distance > YAWN_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    os.makedirs("yawn_images", exist_ok=True)
                    cv2.imwrite(f"yawn_images/yawn_{timestamp}.jpg", frame)
                    with open("drowsiness_log.txt", "a") as f:
                        f.write(f"[{timestamp}] Yawning detected\n")
                    speak("You seem to be yawning. Please stay alert!")
                    warning_label.config(text="YAWNING DETECTED!")
                    yawn_counter = 0
            else:
                yawn_counter = 0

            nose_y = int(leftmost_face.landmark[1].y * ih)
            if prev_nose_y is not None:
                delta_y = nose_y - prev_nose_y
                if delta_y > NOD_THRESHOLD:
                    nodding_counter += 1
                    if nodding_counter >= 3:
                        speak("Head nodding detected. Stay alert!")
                        warning_label.config(text="HEAD NODDING DETECTED!")
                        nodding_counter = 0
                else:
                    nodding_counter = 0
            prev_nose_y = nose_y

            eye_x_positions = [leftmost_face.landmark[idx].x for idx in LEFT_EYE + RIGHT_EYE]
            avg_eye_x = np.mean(eye_x_positions)
            nose_x = leftmost_face.landmark[1].x
            gaze_diff = avg_eye_x - nose_x
            if abs(gaze_diff) > 0.05:
                speak("Please look at the screen.")
                warning_label.config(text="GAZE AWAY DETECTED!")

            mp_drawing.draw_landmarks(frame, leftmost_face, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
    else:
        no_face_counter += 1
        status_label.config(text="No face detected", fg="red")
        if no_face_counter >= NO_FACE_LIMIT:
            speak("No face detected. Are you okay?")
            no_face_counter = 0


    imgarr = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vid.after(30, detect)

def on_close():
    cap.release()
    try:
        face_mesh.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
detect()
app.mainloop()
