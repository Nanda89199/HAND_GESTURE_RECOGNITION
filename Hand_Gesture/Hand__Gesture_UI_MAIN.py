import cv2
import mediapipe as mp
from collections import deque, Counter
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ------------------- Global variables ------------------- #
gesture_text = "No Hand"  # Stores the current gesture name shown in the UI
stop_camera = False  # Controls when to stop the camera
current_frame = None  # Holds the current video frame for the UI display

# ------------------- MediaPipe setup ------------------- #
mp_hands = mp.solutions.hands  # Loads MediaPipe's hand detection tool
mp_drawing = mp.solutions.drawing_utils  # Helps draw hand landmarks on the video
HISTORY_LEN = 10  # Number of past gestures to track for smoother results

# Converts MediaPipe's hand landmark positions to pixel coordinates on the video frame
def landmarks_to_pixels(hand_landmarks, img_w, img_h):
    return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in hand_landmarks.landmark]

# Calculates the angle between three points (like finger joints) to check if a finger is bent or straight
def calculate_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = max(min(dot_product / (mag1 * mag2), 1), -1)
    return math.degrees(math.acos(cos_angle))

# Checks which fingers are up or down based on their positions and angles
def finger_statuses(landmarks, handedness_label):
    finger_indices = {
        'thumb': (4, 3, 2, 1),  # thumb joints: tip, ip, mcp, cmc
        'index': (8, 7, 6, 5),  # index finger joints
        'middle': (12, 11, 10, 9),  # middle finger joints
        'ring': (16, 15, 14, 13),  # ring finger joints
        'pinky': (20, 19, 18, 17)  # pinky finger joints
    }
    
    finger_states = {}  # Stores whether each finger is up (True) or down (False)
    finger_angles = {}  # Stores the angle of each finger for debugging
    wrist = landmarks[0]  # Wrist position for reference
    
    for finger, indices in finger_indices.items():
        tip = landmarks[indices[0]]  # Tip of the finger
        dip = landmarks[indices[1]] if len(indices) > 3 else None  # Distal interphalangeal joint (not for thumb)
        pip = landmarks[indices[2]] if len(indices) > 2 else landmarks[indices[1]]  # Proximal interphalangeal joint
        mcp = landmarks[indices[3]] if len(indices) > 3 else landmarks[indices[2]]  # Metacarpophalangeal joint
        
        angle_pip = calculate_angle(tip, pip, mcp)  # Angle at PIP joint
        angle_dip = calculate_angle(dip, pip, mcp) if dip else angle_pip  # Angle at DIP joint (if exists)
        finger_angles[finger] = angle_pip  # Save PIP angle for display
        
        if finger != 'thumb':
            # For non-thumb fingers, check if tip is above PIP and angles are large (finger extended)
            is_up = (tip[1] < pip[1] - 10) and (angle_pip > 140) and (angle_dip > 140)
            finger_states[finger] = is_up
        else:
            cmc = landmarks[indices[3]]  # Carpometacarpal joint for thumb
            thumb_up = False
            # Check if thumb is extended (above wrist or far sideways)
            if tip[1] < wrist[1] - 50 or abs(tip[0] - wrist[0]) > 100:
                thumb_up = True
            # Check thumb angles to confirm extension
            angle_ip = calculate_angle(tip, landmarks[indices[1]], landmarks[indices[2]])
            if angle_ip > 150 and angle_pip > 150:
                thumb_up = True
            # Adjust for left or right hand
            if handedness_label == 'Right':
                if tip[0] > mcp[0] + 30:
                    thumb_up = True
            else:
                if tip[0] < mcp[0] - 30:
                    thumb_up = True
            # If thumb is curled (small angle), it's not up
            if angle_pip < 120:
                thumb_up = False
            finger_states['thumb'] = thumb_up
    
    return finger_states, finger_angles

# Decides the gesture (like Thumbs Up or Peace) based on which fingers are up
def classify_gesture(fingers):
    t, i, m, r, p = fingers['thumb'], fingers['index'], fingers['middle'], fingers['ring'], fingers['pinky']
    up_count = sum([t, i, m, r, p])  # Count how many fingers are up
    
    if t and not any([i, m, r, p]) and up_count == 1:
        return 'Thumbs Up'  # Only thumb up
    if i and m and not t and not r and not p:
        return 'Peace'  # Index and middle up, others down
    if up_count == 5:
        return 'Open Palm'  # All fingers up
    if up_count == 0:
        return 'Fist'  # No fingers up
    if up_count >= 4:
        return 'Open Palm'  # Almost all fingers up
    return 'Unknown'  # No clear gesture

# Picks the most common gesture from recent frames to avoid flickering
def majority_label(history):
    if not history:
        return 'No Hand'  # No gestures detected
    counter = Counter(history)
    most_common = counter.most_common(1)[0]
    label, count = most_common
    # Require higher agreement (70%) for Thumbs Up and Peace to be sure
    threshold = 0.7 if label in ['Thumbs Up', 'Peace'] else 0.6
    return label if count >= len(history) * threshold else 'Unknown'

# Runs the camera in a separate thread to capture video and detect gestures
def run_camera():
    global gesture_text, stop_camera, current_frame
    cap = cv2.VideoCapture(0)  # Start the webcam
    if not cap.isOpened():
        gesture_text = "ERROR: Cannot access camera"
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )  # Set up MediaPipe to track one hand with high confidence

    history = deque(maxlen=HISTORY_LEN)  # Store recent gestures
    p_time = 0  # For calculating FPS

    while not stop_camera:
        ret, frame = cap.read()  # Grab a frame from the webcam
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        result = hands.process(rgb)  # Process frame to detect hands

        display_label = 'No Hand'
        if result.multi_hand_landmarks and result.multi_handedness:
            hand_landmarks = result.multi_hand_landmarks[0]
            handedness_label = result.multi_handedness[0].classification[0].label
            confidence = result.multi_handedness[0].classification[0].score
            
            if confidence > 0.75:
                pts = landmarks_to_pixels(hand_landmarks, w, h)  # Convert landmarks to pixels
                fingers, angles = finger_statuses(pts, handedness_label)  # Check finger states
                pred = classify_gesture(fingers)  # Identify gesture
                history.append(pred)  # Add to history
                display_label = majority_label(history)  # Get most common gesture
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw hand landmarks
                fs = f"T:{int(fingers['thumb'])} I:{int(fingers['index'])} M:{int(fingers['middle'])} R:{int(fingers['ring'])} P:{int(fingers['pinky'])}"
                cv2.putText(frame, fs, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)  # Show finger states
                cv2.putText(frame, f"Angles T:{int(angles['thumb'])} I:{int(angles['index'])} M:{int(angles['middle'])} R:{int(angles['ring'])} P:{int(angles['pinky'])}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)  # Show angles
            else:
                history.append('No Hand')
                display_label = majority_label(history)

        gesture_text = display_label  # Update UI gesture text

        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time else 0.0
        p_time = c_time
        cv2.putText(frame, f'Gesture: {display_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # Show gesture
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Show FPS

        # Resize frame for UI
        frame = cv2.resize(frame, (600, 450))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        current_frame = ImageTk.PhotoImage(image=img_pil)  # Convert for Tkinter

    cap.release()  # Free the webcam
    hands.close()  # Close MediaPipe
    gesture_text = "Camera Closed"

# Starts the camera in a new thread when the "Open Camera" button is clicked
def start_camera():
    global stop_camera
    stop_camera = False
    threading.Thread(target=run_camera, daemon=True).start()

# Stops the camera when the "Quit Camera" button is clicked
def stop_camera_action():
    global stop_camera
    stop_camera = True

# Updates the UI with the latest gesture and video frame
def update_ui():
    gesture_label.config(text=f"Gesture: {gesture_text}")  # Update gesture text
    if current_frame is not None:
        video_label.config(image=current_frame)  # Update video display
        video_label.image = current_frame
    root.after(100, update_ui)  # Keep updating every 100ms

# ------------------- UI setup ------------------- #
# Set up the main window with a dark theme
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("700x650")
root.configure(bg="#2E2E2E")

# Apply a modern look to buttons and labels
style = ttk.Style()
style.theme_use('clam')

# Style buttons to look green and bold with a hover effect
style.configure('TButton', 
                font=('Helvetica', 14, 'bold'),
                padding=10,
                background='#4CAF50',
                foreground='white')
style.map('TButton', 
          background=[('active', '#45A049')],
          foreground=[('active', 'white')])

# Style the gesture label to be bold with a dark background
style.configure('TLabel', 
                font=('Helvetica', 16, 'bold'),
                background='#3C3F41',
                foreground='white',
                padding=10)

# Create a frame to hold all UI elements
main_frame = ttk.Frame(root)
main_frame.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
main_frame.configure(style='Main.TFrame')
style.configure('Main.TFrame', background='#2E2E2E')

# Add the "Open Camera" button
btn_start = ttk.Button(main_frame, text="Open Camera", command=start_camera)
btn_start.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

# Add the "Quit Camera" button
btn_stop = ttk.Button(main_frame, text="Quit Camera", command=stop_camera_action)
btn_stop.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

# Add the label that shows the current gesture
gesture_label = ttk.Label(main_frame, text="Gesture: No Hand", style='TLabel')
gesture_label.grid(row=1, column=0, columnspan=2, pady=10, sticky='ew')

# Add a bordered area for the video feed
video_label = tk.Label(main_frame, bg='#3C3F41', bd=2, relief='solid')
video_label.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')

# Make the layout stretch to fit the window
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

update_ui()  # Start updating the UI
root.mainloop()  # Run the application