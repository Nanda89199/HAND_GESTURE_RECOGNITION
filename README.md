# Hand Gesture Recognition System
# Author: Nanda Vardhan Reddy
This project is a real-time hand gesture recognition application built with Python. It uses computer vision to detect and classify four hand gestures—Thumbs Up, Peace, Open Palm, and Fist—from a webcam feed. The application features a modern Tkinter GUI that displays the live video feed with gesture annotations and debug information.

# Technology Justification

The following libraries and frameworks were chosen for their suitability in addressing the hand gesture recognition problem:
# OpenCV (opencv-python): 
Used for capturing video from the webcam, processing frames, and overlaying text and landmarks. OpenCV is ideal due to its robust, efficient image processing capabilities and wide adoption in computer vision tasks, ensuring reliable real-time performance.

# MediaPipe: 
Selected for hand landmark detection because it provides a pre-trained, high-accuracy model for tracking 21 hand landmarks per hand. Its lightweight design and CPU-based processing make it suitable for real-time applications without requiring specialized hardware.

# Pillow (PIL): 
Chosen for converting OpenCV frames to Tkinter-compatible images. Pillow is efficient for image format conversions and integrates seamlessly with Tkinter for GUI rendering.

# Tkinter: 
Used for the GUI due to its simplicity and native integration with Python, allowing quick development of a responsive interface with buttons and video display. The ttk module enhances Tkinter with modern, themed widgets for a polished look.

# NumPy: 
Essential for numerical computations, such as angle calculations and array operations, required by OpenCV and MediaPipe. Its efficiency in handling arrays ensures
fast processing of landmark coordinates.

# Gesture Logic Explanation:

# Thumbs Up:
1.Logic: The thumb is extended (up or sideways), while all other fingers (index, middle, ring, pinky) are folded down.
2.Implementation: Check if the thumb tip is above the wrist or significantly offset horizontally. Verify thumb joint angles (IP and PIP) are large (>150°) to ensure extension. Confirm other fingers have small angles (<140°) and tips below their PIP joints to ensure they’re folded.
3.Key Check: thumb_up=True, index/middle/ring/pinky=False, and total up_count=1.

# Peace:
1.Logic: Index and middle fingers are extended, while thumb, ring, and pinky are folded down.
2.Implementation: Verify index and middle finger tips are above their PIP joints with large angles (>140°) at PIP and DIP joints. Ensure thumb, ring, and pinky have small angles (<120° for thumb, <140° for others) and tips below PIP joints.
3.Key Check: index=True, middle=True, thumb/ring/pinky=False.

# Open Palm:
1.Logic: All five fingers (thumb, index, middle, ring, pinky) are extended.
2.Implementation: Confirm all finger tips are above their PIP joints and have large angles (>140° for fingers, >150° for thumb). A partial match (4+ fingers up) also classifies as Open Palm to account for slight variations.
3.Key Check: up_count=5 or up_count>=4.

# Fist:
1.Logic: All fingers are folded down, forming a closed hand.
2.Implementation: Check that all finger tips are below their PIP joints and joint angles are small (<140° for fingers, <120° for thumb).
3.Key Check: up_count=0.


# Setup and Execution Instructions

# Prerequisites:
1.Python 3.12 or later installed.
2.A webcam connected to your computer.
3.Git installed (optional, for cloning).

# Using the Application:
1.Click Open Camera to start the webcam and gesture detection.
2.Position your hand clearly in front of the webcam to perform gestures (Thumbs Up, Peace, Open Palm, Fist).
3.The GUI displays the live video feed with hand landmarks, finger states, joint angles, and the detected gesture.
4.Click Quit Camera to stop the webcam and close the application.























These technologies were chosen for their synergy, open-source availability, and proven reliability in real-time computer vision tasks. MediaPipe’s hand tracking model, combined with OpenCV’s video processing and Tkinter’s GUI, provides a complete solution that balances accuracy, performance, and ease of use.
