import cv2
import face_recognition
import pickle
import os
import csv
import time
from datetime import datetime
from gtts import gTTS
from playsound import playsound
import dlib
from numpy.linalg import norm
import numpy as np

# Fungsi text-to-speech
def speak(text):
    audio = gTTS(text=text, lang="id")
    audio.save("audio.mp3")
    playsound("audio.mp3")
    os.remove("audio.mp3")

# Kamera & background
video = cv2.VideoCapture(0)
imgBackground = cv2.imread("background.png")
COL_NAMES = ['NAMA', 'WAKTU']

# Detektor wajah & ekspresi
facedetector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
smile_detector = cv2.CascadeClassifier('data/haarcascade_smile.xml')

# Load known face embeddings
with open("data/known_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# Threshold untuk mata & smile
threshold_eye = 0.2
eye_closed = False
kedip = 'X'
color_kedip = (0,0,255)
senyum = 'X'
color_senyum = (0,0,255)

# Fungsi EAR
def mid_line_distance(p1, p2, p3, p4):
    p5 = np.array([(p1[0]+p2[0])//2, (p1[1]+p2[1])//2])
    p6 = np.array([(p3[0]+p4[0])//2, (p3[1]+p4[1])//2])
    return norm(p5 - p6)

def aspect_ratio(landmarks, eye_range):
    eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in eye_range])
    B = norm(eye[0] - eye[3])
    A = mid_line_distance(eye[1], eye[2], eye[5], eye[4])
    return A / B

# Loop utama
while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetector(gray)

    # Deteksi mata (kedip)
    for rect in faces:
        landmarks = predictor(gray, rect)
        left_ear = aspect_ratio(landmarks, range(42,48))
        right_ear = aspect_ratio(landmarks, range(36,42))
        ear = (left_ear + right_ear)/2.0
        if ear < threshold_eye:
            eye_closed = True
        elif ear >= threshold_eye and eye_closed:
            kedip = 'Berhasil'
            color_kedip = (0,255,0)
            eye_closed = False

        cv2.putText(frame, f"Pejamkan Mata: {kedip}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_kedip, 2)

    # Deteksi senyum
    smiles = smile_detector.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=50, minSize=(25,25))
    if len(smiles) > 0:
        senyum = 'Berhasil'
        color_senyum = (0,255,0)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(255,0,0),1)
    else:
        senyum = 'X'
        color_senyum = (0,0,255)

    cv2.putText(frame, f"Perlihatkan Gigi: {senyum}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_senyum, 2)

    # Deteksi wajah & klasifikasi dengan face embeddings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    nama_pengabsen = ''

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) > 0:
            min_distance = min(distances)
            idx = np.argmin(distances)
            if min_distance < 0.6:  # threshold
                nama_pengabsen = known_names[idx]
            else:
                nama_pengabsen = 'UNKNOWN'
                speak("Wajah tidak dikenal!")
        else:
            nama_pengabsen = 'UNKNOWN'
            speak("Wajah tidak dikenal!")

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.putText(frame, nama_pengabsen, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2)

    # Tampilkan frame di background
    imgBackground[140:140+480, 40:40+640] = frame
    cv2.imshow("ABSEN ANTI NITIP v2.0", imgBackground)
    k = cv2.waitKey(1)

    # Proses absen
    if kedip == 'Berhasil' and senyum == 'Berhasil' and nama_pengabsen != '' and nama_pengabsen != 'UNKNOWN':
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = date + datetime.fromtimestamp(ts).strftime(" %H:%M")
        absen_file = f"Absen/Absen_{date}.csv"
        if not os.path.exists("Absen"):
            os.makedirs("Absen")

        already_absent = False
        if os.path.isfile(absen_file):
            with open(absen_file,'r',newline='') as f:
                reader = csv.reader(f,delimiter=';')
                next(reader,None)
                for row in reader:
                    if len(row)>0 and row[0] == nama_pengabsen:
                        already_absent = True
                        break

        if already_absent:
            speak(f"Halo {nama_pengabsen}, kamu sudah absen hari ini")
        else:
            with open(absen_file,'a',newline='') as f:
                writer = csv.writer(f,delimiter=';')
                if os.path.getsize(absen_file)==0:
                    writer.writerow(COL_NAMES)
                writer.writerow([nama_pengabsen,timestamp])
            speak(f"Halo {nama_pengabsen}, Terimakasih Sudah Absen")

        kedip = 'X'
        color_kedip = (0,0,255)
        senyum = 'X'
        color_senyum = (0,0,255)

    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
