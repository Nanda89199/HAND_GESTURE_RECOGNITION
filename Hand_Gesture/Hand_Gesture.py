import cv2
import mediapipe as mp
from collections import deque, Counter
import time
import math

print("=== Starting Enhanced Gesture Script (Peace Gesture Fix) ===")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HISTORY_LEN = 10  # Consistent history length for smoothing

def landmarks_to_pixels(hand_landmarks, img_w, img_h):
    return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in hand_landmarks.landmark]

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 as vertex) in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(min(cos_angle, 1), -1)  # Clamp to avoid math errors
    return math.degrees(math.acos(cos_angle))

def finger_statuses(landmarks, handedness_label):
    """Determine if each finger is up or down based on relative positions and angles."""
    finger_indices = {
        'thumb': (4, 3, 2),  # tip, ip, mcp
        'index': (8, 6, 5),
        'middle': (12, 10, 9),
        'ring': (16, 14, 13),
        'pinky': (20, 18, 17)
    }
    
    finger_states = {}
    finger_angles = {}  # For debug display
    
    for finger, (tip_idx, pip_idx, mcp_idx) in finger_indices.items():
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        
        # Calculate angle at pip joint
        angle = calculate_angle(tip, pip, mcp)
        finger_angles[finger] = angle
        
        # For fingers (except thumb), up if angle is large and tip is above mcp
        if finger != 'thumb':
            finger_states[finger] = tip[1] < mcp[1] and angle > 150  # Increased angle threshold for clarity
        else:
            # Thumb-specific logic
            thumb_up = False
            if handedness_label == 'Right':
                if tip[0] > pip[0] + 20 and tip[1] < mcp[1] - 20:  # Stricter thresholds
                    thumb_up = True
            else:
                if tip[0] < pip[0] - 20 and tip[1] < mcp[1] - 20:
                    thumb_up = True
            # Thumb must be folded for Peace gesture
            if angle < 100:  # Thumb folded if angle is small
                thumb_up = False
            finger_states['thumb'] = thumb_up
    
    return finger_states, finger_angles

def classify_gesture(fingers):
    """Classify gesture with strict conditions for Peace gesture."""
    t, i, m, r, p = fingers['thumb'], fingers['index'], fingers['middle'], fingers['ring'], fingers['pinky']
    
    up_count = sum([t, i, m, r, p])
    
    # Peace: Strictly index and middle up, others down
    if i and m and not t and not r and not p:
        return 'Peace'
    
    # Open Palm: All fingers up
    if up_count == 5:
        return 'Open Palm'
    
    # Fist: No fingers up
    if up_count == 0:
        return 'Fist'
    
    # Thumbs Up: Only thumb up
    if t and not any([i, m, r, p]):
        return 'Thumbs Up'
    
    # Partial open palm: At least 4 fingers up
    if up_count >= 4:
        return 'Open Palm'
    
    # Default to Unknown for ambiguous cases
    return 'Unknown'

def majority_label(history):
    """Return the most common label in history, with stricter threshold for Peace."""
    if not history:
        return 'No Hand'
    counter = Counter(history)
    most_common = counter.most_common(1)[0]
    label, count = most_common
    # Require 70% agreement for Peace, 60% for others
    threshold = 0.7 if label == 'Peace' else 0.6
    if count >= len(history) * threshold:
        return label
    return 'Unknown'

def main():
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    print("Webcam initialized successfully!")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    history = deque(maxlen=HISTORY_LEN)
    p_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            display_label = 'No Hand'

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_landmarks = result.multi_hand_landmarks[0]
                handedness_label = result.multi_handedness[0].classification[0].label
                confidence = result.multi_handedness[0].classification[0].score
                
                if confidence > 0.7:
                    pts = landmarks_to_pixels(hand_landmarks, w, h)
                    fingers, angles = finger_statuses(pts, handedness_label)
                    pred = classify_gesture(fingers)
                    history.append(pred)
                    display_label = majority_label(history)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fs = f"T:{int(fingers['thumb'])} I:{int(fingers['index'])} M:{int(fingers['middle'])} R:{int(fingers['ring'])} P:{int(fingers['pinky'])}"
                    angles_text = f"Angles T:{int(angles['thumb'])} I:{int(angles['index'])} M:{int(angles['middle'])} R:{int(angles['ring'])} P:{int(angles['pinky'])}"
                    cv2.putText(frame, fs, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                    cv2.putText(frame, angles_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                else:
                    history.append('No Hand')
                    display_label = majority_label(history)
            else:
                history.append('No Hand')
                display_label = majority_label(history)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if p_time else 0.0
            p_time = c_time
            cv2.putText(frame, f'Gesture: {display_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Enhanced Hand Gesture Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting on user request...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Released webcam and closed windows.")

if __name__ == '__main__':
    main()