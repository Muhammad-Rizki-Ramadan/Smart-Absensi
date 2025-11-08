import cv2
import face_recognition
import pickle
import os
import numpy as np

# Inisialisasi kamera
video = cv2.VideoCapture(0)

# Input nama pengguna
name = input("Masukkan Nama Anda: ").strip()
if name == "":
    print("Nama tidak boleh kosong!")
    exit()

faces_data = []
total_capture = 75

print("[INFO] Tekan 'q' untuk berhenti kapan saja...")

while True:
    ret, frame = video.read()
    if not ret:
        print("[ERROR] Gagal membaca frame dari kamera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi lokasi wajah di frame
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        # Ambil embedding langsung dari frame penuh menggunakan koordinat wajah
        try:
            face_enc = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
        except IndexError:
            continue  # skip jika gagal mendapatkan encoding

        if len(faces_data) < total_capture:
            faces_data.append(face_enc)

        # Gambar rectangle di layar
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{len(faces_data)}/{total_capture}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Dataset Capture", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= total_capture:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.array(faces_data)

# Simpan embeddings dan nama
if not os.path.exists("data"):
    os.makedirs("data")

embeddings_file = "data/known_embeddings.pkl"

if os.path.isfile(embeddings_file):
    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)
    data["encodings"].extend(faces_data.tolist())
    data["names"].extend([name]*len(faces_data))
else:
    data = {
        "encodings": faces_data.tolist(),
        "names": [name]*len(faces_data)
    }

with open(embeddings_file, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Dataset untuk {name} berhasil disimpan dengan {len(faces_data)} wajah!")
