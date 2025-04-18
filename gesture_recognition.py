import cv2
import mediapipe as mp
import numpy as np
import time

class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Gesture tracking variables
        self.prev_landmarks = None
        self.gesture_cooldown = 0
        self.last_gesture_time = time.time()
        self.cooldown_period = 1.0  # Cooldown in seconds

        # Store hand movement history (for swipe detection)
        self.hand_position_history = []
        self.history_length = 10

    def detect_gestures(self, frame):
        """Detect hand gestures in the frame"""
        # Flip the image horizontally for a more natural feel
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)

        # Initialize detected gesture to None
        detected_gesture = None

        # Draw hand landmarks on the frame and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Detect gestures only if cooldown period has passed
                current_time = time.time()
                if current_time - self.last_gesture_time > self.cooldown_period:
                    # Get landmarks as a numpy array for easier processing
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                    ])

                    # Track hand position for swipe detection
                    palm_position = landmarks[0]  # Wrist position
                    self.hand_position_history.append(palm_position)
                    if len(self.hand_position_history) > self.history_length:
                        self.hand_position_history.pop(0)

                    # Check for thumbs up gesture
                    if self._is_thumbs_up(landmarks):
                        detected_gesture = "thumbs_up"
                        self.last_gesture_time = current_time

                    # Check for thumbs down gesture
                    elif self._is_thumbs_down(landmarks):
                        detected_gesture = "thumbs_down"
                        self.last_gesture_time = current_time

                    # Check for swipe gestures if we have enough history
                    elif len(self.hand_position_history) >= self.history_length:
                        # Check for swipe left
                        if self._is_swipe_left():
                            detected_gesture = "swipe_left"
                            self.last_gesture_time = current_time

                        # Check for swipe right
                        elif self._is_swipe_right():
                            detected_gesture = "swipe_right"
                            self.last_gesture_time = current_time

                        # Check for swipe up
                        elif self._is_swipe_up():
                            detected_gesture = "swipe_up"
                            self.last_gesture_time = current_time

                        # Check for swipe down
                        elif self._is_swipe_down():
                            detected_gesture = "swipe_down"
                            self.last_gesture_time = current_time

                # Display the detected gesture on the frame
                if detected_gesture:
                    cv2.putText(
                        frame,
                        f"Gesture: {detected_gesture}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        return frame, detected_gesture

    def _is_thumbs_up(self, landmarks):
        """Check if the gesture is thumbs up"""
        # Thumb tip is pointing up and other fingers are folded
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]

        # Check if thumb is pointing up (y-coordinate decreases)
        thumb_up = thumb_tip[1] < thumb_base[1]

        # Check if other fingers are folded
        index_folded = landmarks[8][1] > landmarks[5][1]
        middle_folded = landmarks[12][1] > landmarks[9][1]
        ring_folded = landmarks[16][1] > landmarks[13][1]
        pinky_folded = landmarks[20][1] > landmarks[17][1]

        return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

    def _is_thumbs_down(self, landmarks):
        """Check if the gesture is thumbs down"""
        # Thumb tip is pointing down and other fingers are folded
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]

        # Check if thumb is pointing down (y-coordinate increases)
        thumb_down = thumb_tip[1] > thumb_base[1]

        # Check if other fingers are folded
        index_folded = landmarks[8][1] > landmarks[5][1]
        middle_folded = landmarks[12][1] > landmarks[9][1]
        ring_folded = landmarks[16][1] > landmarks[13][1]
        pinky_folded = landmarks[20][1] > landmarks[17][1]

        return thumb_down and index_folded and middle_folded and ring_folded and pinky_folded

    def _is_swipe_left(self):
        """Check if the gesture is a swipe left"""
        start_x = self.hand_position_history[0][0]
        end_x = self.hand_position_history[-1][0]

        # Check if hand moved significantly to the left
        # (x-coordinate decreases significantly)
        return (start_x - end_x) > 0.2

    def _is_swipe_right(self):
        """Check if the gesture is a swipe right"""
        start_x = self.hand_position_history[0][0]
        end_x = self.hand_position_history[-1][0]

        # Check if hand moved significantly to the right
        # (x-coordinate increases significantly)
        return (end_x - start_x) > 0.2

    def _is_swipe_up(self):
        """Check if the gesture is a swipe up"""
        start_y = self.hand_position_history[0][1]
        end_y = self.hand_position_history[-1][1]

        # Check if hand moved significantly upward
        # (y-coordinate decreases significantly)
        return (start_y - end_y) > 0.2

    def _is_swipe_down(self):
        """Check if the gesture is a swipe down"""
        start_y = self.hand_position_history[0][1]
        end_y = self.hand_position_history[-1][1]

        # Check if hand moved significantly downward
        # (y-coordinate increases significantly)
        return (end_y - start_y) > 0.2

    def release(self):
        """Release resources"""
        self.hands.close()

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize gesture recognizer
    gesture_recognizer = GestureRecognizer()

    print("Gesture recognition started. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Detect gestures
        frame, detected_gesture = gesture_recognizer.detect_gestures(frame)

        # If a gesture is detected, print it
        if detected_gesture:
            print(f"Detected gesture: {detected_gesture}")

        # Display the frame
        cv2.imshow('Gesture Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    gesture_recognizer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

if __name__ == "__main__":
    main()
