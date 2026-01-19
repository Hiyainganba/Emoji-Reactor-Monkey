import cv2
import mediapipe as mp
import numpy as np
import math

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

MOUTH_OPEN_THRESHOLD = 0.05
EYE_LOOK_UP_THRESHOLD = 0.45
BITE_THRESHOLD = 0.08

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def load_and_resize(filename):
    try:
        img = cv2.imread(filename)
        if img is None:
            return None
        return cv2.resize(img, EMOJI_WINDOW_SIZE)
    except Exception:
        return None

noreaction_img = load_and_resize("noreaction.jpg")
surprise_img = load_and_resize("surprise.jpg")
think_img = load_and_resize("think.jpg")
gotit_img = load_and_resize("gotit.jpg")

images_list = [noreaction_img, surprise_img, think_img, gotit_img]
if any(img is None for img in images_list):
    print("Error: Missing image files.")
    exit()

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_mouth_open(face_landmarks):
    upper = face_landmarks.landmark[13]
    lower = face_landmarks.landmark[14]
    face_top = face_landmarks.landmark[10]
    chin = face_landmarks.landmark[152]
    
    mouth_h = get_distance(upper, lower)
    face_h = get_distance(face_top, chin)
    
    if face_h == 0:
        return False
    return (mouth_h / face_h) > MOUTH_OPEN_THRESHOLD

def is_looking_up(face_landmarks):
    eye_top = face_landmarks.landmark[386]
    eye_bottom = face_landmarks.landmark[374]
    iris_center = face_landmarks.landmark[468]
    
    eye_height = eye_bottom.y - eye_top.y
    if eye_height == 0:
        return False
    
    iris_relative_y = (iris_center.y - eye_top.y) / eye_height
    return iris_relative_y < EYE_LOOK_UP_THRESHOLD

def is_index_pointing_up(hand_landmarks):
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    return index_up and middle_down and ring_down and pinky_down

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        
        looking_up = False
        mouth_open = False
        mouth_center = None
        
        if face_results.multi_face_landmarks:
            fl = face_results.multi_face_landmarks[0]
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=fl,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            h, w, c = frame.shape
            x_min, x_max = w, 0
            y_min, y_max = h, 0
            
            for lm in fl.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            looking_up = is_looking_up(fl)
            mouth_open = is_mouth_open(fl)
            mx = (fl.landmark[13].x + fl.landmark[14].x) / 2
            my = (fl.landmark[13].y + fl.landmark[14].y) / 2
            mouth_center = type('obj', (object,), {'x': mx, 'y': my})

        hand_count = 0
        pointing_up = False
        touching_mouth = False
        
        if hand_results.multi_hand_landmarks:
            hand_count = len(hand_results.multi_hand_landmarks)
            for hl in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                
                if is_index_pointing_up(hl):
                    pointing_up = True
                
                if mouth_center:
                    index_tip = hl.landmark[8]
                    dist = get_distance(index_tip, mouth_center)
                    if dist < BITE_THRESHOLD:
                        touching_mouth = True

        state = "No Reaction"
        final_img = noreaction_img
        
        if touching_mouth and looking_up:
            state = "Thinking"
            final_img = think_img
        elif pointing_up and looking_up and mouth_open:
            state = "Got It!"
            final_img = gotit_img
        elif hand_count == 2 and mouth_open:
            state = "Surprised"
            final_img = surprise_img
            
        disp_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        cv2.putText(disp_frame, f"State: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Camera", disp_frame)
        cv2.imshow("Reaction", final_img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
