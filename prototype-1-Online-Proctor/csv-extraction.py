import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import speech_recognition as sr
import threading
import time
import psutil
import csv

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Blink Detection Parameters
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0
blink_total = 0
cheat_intensity = 0
voice_detected = False

# Iris and Eye Landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Lip Landmarks
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14
lip_open_count = 0
lip_open = False

# Lists to collect performance data
cpu_usage_data = []
memory_usage_data = []
fps_data = []

# Lists to collect event data
blink_data = []
voice_data = []
lip_open_data = []
gaze_data = []

# EAR Calculation Function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Extract Eye Landmarks
def extract_eye_landmarks(landmarks, indices):
    return np.array([(landmarks[i].x, landmarks[i].y) for i in indices])

# Voice Detection Thread
def detect_voice():
    global voice_detected
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=2)
                recognizer.recognize_google(audio)
                voice_detected = True
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                voice_detected = False

# Gaze Direction Functions
def get_landmark_coordinates(landmarks, index, image_shape):
    x = int(landmarks[index].x * image_shape[1])
    y = int(landmarks[index].y * image_shape[0])
    return (x, y)

def get_gaze_direction(landmarks, image_shape):
    left_eye = get_landmark_coordinates(landmarks, 33, image_shape)
    right_eye = get_landmark_coordinates(landmarks, 263, image_shape)
    nose_tip = get_landmark_coordinates(landmarks, 1, image_shape)
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    nose_to_eye_vector = np.array([eye_center[0] - nose_tip[0], eye_center[1] - nose_tip[1]])
    if nose_to_eye_vector[0] > 20:
        return "head turned Left"
    elif nose_to_eye_vector[0] < -20:
        return "head turned Right"
    else:
        return "Looking Forward"

# Start voice detection thread
voice_thread = threading.Thread(target=detect_voice, daemon=True)
voice_thread.start()

cap = cv2.VideoCapture(0)
start_time = time.time()
frame_data = []
frame_index = 0  # Frame index for resets

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as face_mesh:

    while cap.isOpened() and (time.time() - start_time) < 100:
        frame_start_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w = image.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                left_eye = extract_eye_landmarks(face_landmarks.landmark, LEFT_EYE)
                right_eye = extract_eye_landmarks(face_landmarks.landmark, RIGHT_EYE)
                left_eye_ear = eye_aspect_ratio(left_eye)
                right_eye_ear = eye_aspect_ratio(right_eye)
                ear = (left_eye_ear + right_eye_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_total += 1
                        cheat_intensity += 1
                    blink_counter = 0

                gaze_direction = get_gaze_direction(face_landmarks.landmark, image.shape)

                upper_lip = face_landmarks.landmark[UPPER_LIP_INDEX]
                lower_lip = face_landmarks.landmark[LOWER_LIP_INDEX]
                lip_distance = abs(upper_lip.y - lower_lip.y)
                if lip_distance > 0.05:
                    if not lip_open:
                        lip_open_count += 1
                        lip_open = True
                else:
                    lip_open = False

                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in face_landmarks.landmark])
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
                right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
                left_gaze_ratio = (center_left[0] - left_eye[0][0]) / left_eye_width
                right_gaze_ratio = (center_right[0] - right_eye[0][0]) / right_eye_width
                iris_gaze_direction = "Center"
                if left_gaze_ratio < 0.5 and right_gaze_ratio < 0.5:
                    iris_gaze_direction = "Looking Left"
                elif left_gaze_ratio > 0.5 and right_gaze_ratio > 0.5:
                    iris_gaze_direction = "Looking Right"

                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                fps = 1 / (time.time() - frame_start_time)

                gaze_direction_map = {"head turned Left": -1, "Looking Forward": 0, "head turned Right": 1}
                suspicious = int(
                    (lip_open_count >= 3) or
                    (voice_detected) or
                    (gaze_direction in ["head turned Left", "head turned Right"]) or
                    (blink_total > 5)
                )

                frame_info = {
                    "frame": frame_index,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "fps": fps,
                    "blink_total": blink_total,
                    "voice_detected": int(voice_detected),
                    "lip_open_count": lip_open_count,
                    "gaze_direction": gaze_direction_map.get(gaze_direction, 0),
                    "iris_gaze_direction": 1 if iris_gaze_direction == "Looking Right" else -1 if iris_gaze_direction == "Looking Left" else 0,
                    "suspicious": suspicious
                }

                frame_data.append(frame_info)

                # Draw Info on Screen
                cv2.putText(image, f"Blinks: {blink_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Voice: {'Detected' if voice_detected else 'None'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Gaze: {gaze_direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f'Iris Gaze: {iris_gaze_direction}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Lip Opens: {lip_open_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, f"CPU: {cpu_usage:.2f}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(image, f"Memory: {memory_usage:.2f}%", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(image, f"FPS: {fps:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                if suspicious:
                    cv2.putText(image, "Suspicious Activity Detected!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow('Gaze, Blink, Voice, Lip, and Iris Detector', image)

                # 🔁 Reset every 15 seconds (assuming 30 FPS ≈ 450 frames)
                reset_interval = 450
                if frame_index % reset_interval == 0:
                    blink_total = 0
                    lip_open_count = 0
                    voice_detected = False


                frame_index += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Save to CSV
csv_file = "final_suspicious_activity_data.csv"
csv_columns = ["frame", "cpu_usage", "memory_usage", "fps", "blink_total", "voice_detected", "lip_open_count", "gaze_direction", "iris_gaze_direction", "suspicious"]
try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in frame_data:
            writer.writerow(data)
    print(f"Data saved to {csv_file}")
except IOError:
    print("I/O error occurred while saving CSV")

cap.release()
cv2.destroyAllWindows()
