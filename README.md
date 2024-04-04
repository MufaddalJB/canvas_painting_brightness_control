# canvas_painting_brightness_control
Hand Gesture Controlled Drawing with OpenCV and MediaPipe
This project enables hand gesture-controlled drawing using a webcam and the OpenCV library in Python. It detects hand gestures such as open palm and closed palm to control the screen brightness, while finger gestures are used for painting on the canvas. The drawing interface is implemented using the MediaPipe library for hand landmark detection and OpenCV for image processing.

Features:
Real-time Hand Gesture Detection: Utilizes the MediaPipe library to detect hand landmarks in real-time from the webcam feed.
Open Palm Detection for Brightness Control: Recognizes when the user's hand is in an open palm position to increase screen brightness.
Closed Palm Detection for Brightness Control: Identifies when the user's hand is in a closed palm position to decrease screen brightness.
Finger Gestures for Painting: Utilizes finger positions to draw lines on the canvas, with different colors corresponding to fingers.
Color Selection: Allows the user to select different colors (blue, green, red, yellow) by positioning the fingers in designated areas of the screen.
Clear Screen Functionality: Provides an option to clear the drawing canvas by touching a predefined clear button area on the screen.
Requirements:
Python 3.x
OpenCV
NumPy
Mediapipe
Screen Brightness Control (sbc)
Usage:
Clone the repository to your local machine.
Install the required dependencies using pip install -r requirements.txt.
Run the main() function in the hand_gesture_drawing.py file.
Point your webcam towards your hand and start drawing by making gestures as described in the application.
Acknowledgments:
This project utilizes the OpenCV and MediaPipe libraries, along with other open-source contributions, to create an interactive hand gesture-controlled drawing experience.
