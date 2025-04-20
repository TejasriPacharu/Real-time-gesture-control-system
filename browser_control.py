import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from enum import IntEnum


# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings
class Gest(IntEnum):
    """
    Enum for mapping hand gestures to integer values
    """
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16
    PALM = 31

    # Browser control specific gestures
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

    # Swipe gestures
    SWIPE_LEFT = 50
    SWIPE_RIGHT = 51
    SWIPE_UP = 52
    SWIPE_DOWN = 53

    # Thumbs gestures
    THUMBS_UP = 60
    THUMBS_DOWN = 61

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

class HandRecognizer:
    """
    Convert MediaPipe landmarks to recognizable gestures
    """

    def __init__(self, hand_label):
        """
        Initialize HandRecognizer object

        Parameters:
        hand_label (HLabel): Indicates if this is the major or minor hand
        """
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label

        # For swipe detection
        self.prev_hand_center = None
        self.hand_history = []
        self.history_length = 5
        self.swipe_threshold = 0.1

    def update_hand_result(self, hand_result):
        """Update hand landmarks result"""
        self.hand_result = hand_result

        # Update hand history for swipe detection
        if hand_result:
            # Calculate hand center (using index finger base as reference)
            current_pos = np.array([
                hand_result.landmark[5].x,
                hand_result.landmark[5].y
            ])

            # Add to history
            self.hand_history.append(current_pos)
            if len(self.hand_history) > self.history_length:
                self.hand_history.pop(0)

            # Update previous hand center
            self.prev_hand_center = current_pos

    def get_signed_dist(self, point):
        """
        Calculate signed Euclidean distance between landmarks

        Parameters:
        point (list): Two landmark indices

        Returns:
        float: Signed distance between landmarks
        """
        if self.hand_result is None:
            return 0

        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = np.sqrt(dist)
        return dist * sign

    def get_gesture(self):
        """
        Determine the current hand gesture with confidence threshold

        Returns:
        Gest: Detected gesture
        """
        if self.hand_result is None:
            return Gest.PALM

        # First check for predefined gestures

        # Check for thumbs up/down
        if self.detect_thumbs_up():
            current_gesture = Gest.THUMBS_UP
        elif self.detect_thumbs_down():
            current_gesture = Gest.THUMBS_DOWN
        else:
            # Check for swipe gestures
            swipe_gesture = self.detect_swipe()
            if swipe_gesture:
                current_gesture = swipe_gesture
            # Default to finger state based gesture
            else:
                # Check for pinch gesture
                if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
                    if self.hand_label == HLabel.MINOR:
                        current_gesture = Gest.PINCH_MINOR
                    else:
                        current_gesture = Gest.PINCH_MAJOR
                # Check for V gesture
                elif Gest.FIRST2 == self.finger:
                    point = [[8, 12], [5, 9]]
                    dist1 = self.get_dist(point[0])
                    dist2 = self.get_dist(point[1])
                    ratio = dist1/dist2
                    if ratio > 1.7:
                        current_gesture = Gest.V_GEST
                    else:
                        if self.get_dz([8, 12]) < 0.1:
                            current_gesture = Gest.TWO_FINGER_CLOSED
                        else:
                            current_gesture = Gest.MID
                else:
                    current_gesture = self.finger

        # Handle gesture stabilization to reduce flickering
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        # Only return gesture if it's been stable for several frames
        # Increased the frame threshold for better confidence
        if self.frame_count > 6:  # Increased from 4 to 6
            self.ori_gesture = current_gesture
            return self.ori_gesture
        elif self.frame_count > 2:  # Return the original gesture if somewhat stable
            return self.ori_gesture
        else:
            return None  # Return None if not confident enough

    def get_dist(self, point):
        """
        Calculate Euclidean distance between landmarks

        Parameters:
        point (list): Two landmark indices

        Returns:
        float: Distance between landmarks
        """
        if self.hand_result is None:
            return 0

        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = np.sqrt(dist)
        return dist

    def get_dz(self, point):
        """
        Calculate absolute difference on z-axis between landmarks

        Parameters:
        point (list): Two landmark indices

        Returns:
        float: Z-axis difference between landmarks
        """
        if self.hand_result is None:
            return 0

        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self):
        """
        Set finger state based on hand landmark positions
        """
        if self.hand_result is None:
            return

        # Points represent fingertips and knuckles
        # Format: [fingertip idx, base knuckle idx, palm idx]
        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0

        # Initialize thumb state (will be set properly in specific gesture detection)
        self.finger = self.finger | 0

        # Check each finger
        for idx, point in enumerate(points):
            # Calculate distance ratios for finger extension
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])

            try:
                ratio = round(dist/dist2, 1)
            except:
                ratio = 0.1  # Default if division by zero

            # Shift finger state and set bit if finger is extended
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1


    def detect_thumbs_up(self):
        """
        Detect thumbs up gesture

        Returns:
        bool: True if thumbs up detected
        """
        if self.hand_result is None:
            return False

        # Thumb points
        thumb_tip = self.hand_result.landmark[4]
        thumb_mcp = self.hand_result.landmark[2]

        # Check thumb direction (pointing up)
        thumb_up = thumb_tip.y < thumb_mcp.y

        # Check if other fingers are folded
        index_folded = self.hand_result.landmark[8].y > self.hand_result.landmark[5].y
        middle_folded = self.hand_result.landmark[12].y > self.hand_result.landmark[9].y
        ring_folded = self.hand_result.landmark[16].y > self.hand_result.landmark[13].y
        pinky_folded = self.hand_result.landmark[20].y > self.hand_result.landmark[17].y

        return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded


    def detect_thumbs_down(self):
        """

        """

        if self.hand_result is None:
            return False

        # Thumb points
        thumb_tip = self.hand_result.landmark[4]
        thumb_mcp = self.hand_result.landmark[2]

        # Check thumb direction (pointing down)
        thumb_down = thumb_tip.y > thumb_mcp.y

        # Check if other fingers are folded
        index_folded = self.hand_result.landmark[8].y > self.hand_result.landmark[5].y
        middle_folded = self.hand_result.landmark[12].y > self.hand_result.landmark[9].y
        ring_folded = self.hand_result.landmark[16].y > self.hand_result.landmark[13].y
        pinky_folded = self.hand_result.landmark[20].y > self.hand_result.landmark[17].y

        return thumb_down and index_folded and middle_folded and ring_folded and pinky_folded


    def detect_swipe(self):
        """
        Detect swipe gestures

        Returns:
        Gest: Swipe gesture or None
        """
        if len(self.hand_history) < self.history_length:
            return None

        # Calculate movement vector between first and last position
        start_pos = self.hand_history[0]
        end_pos = self.hand_history[-1]
        movement = end_pos - start_pos

        # Calculate magnitude of movement
        magnitude = np.sqrt(np.sum(movement**2))

        # Only detect swipes with significant movement
        if magnitude < self.swipe_threshold:
            return None

        # Determine direction of swipe
        dx, dy = movement

        # Check if movement is primarily horizontal or vertical
        if abs(dx) > abs(dy):
            # Horizontal swipe
            if dx > 0:
                return Gest.SWIPE_RIGHT
            else:
                return Gest.SWIPE_LEFT
        else:
            # Vertical swipe
            if dy > 0:
                return Gest.SWIPE_DOWN
            else:
                return Gest.SWIPE_UP

class BrowserController:
    """
    Controller for browser actions based on hand gestures
    """

    def __init__(self):
        """Initialize Browser Controller"""
        # Set a flag to indicate if the system is running
        self.running = True

        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True

        # For tracking hands
        self.hand_detector = None
        self.hand_major = None
        self.hand_minor = None

        # Define gesture-to-action mapping
        self.gesture_actions = {
            Gest.SWIPE_LEFT: self.browser_back,
            Gest.SWIPE_RIGHT: self.browser_forward,
            Gest.SWIPE_UP: self.scroll_up,
            Gest.SWIPE_DOWN: self.scroll_down,
            Gest.THUMBS_UP: self.new_tab,
            Gest.THUMBS_DOWN: self.close_tab,
            Gest.V_GEST: self.refresh_page
        }

        # Cooldown for actions to prevent multiple triggers
        self.last_action_time = 0
        self.action_cooldown = 1.0  # seconds

    def detect_browser(self):
        """
        Detect which browser is currently active to use appropriate shortcuts

        try:
            # This uses pyautogui to get the active window title
            # Note: requires pygetwindow package
            active_window = gw.getActiveWindow()
            if active_window:
                title = active_window.title.lower()

                # Check for common browser names in the window title
                if 'chrome' in title:
                    return 'chrome'
                elif 'firefox' in title:
                    return 'firefox'
                elif 'edge' in title:
                    return 'edge'
                elif 'safari' in title:
                    return 'safari'
                else:
                    return 'unknown'
        except:
            # If we can't determine the browser, default to common shortcuts

        """
        return 'unknown'

    def test_browser_actions(self):
        """Test all browser control actions"""
        print("Testing browser controls...")
        time.sleep(1)
        print("Opening new tab...")
        self.new_tab()
        time.sleep(2)
        print("Refreshing page...")
        self.refresh_page()
        time.sleep(2)
        print("Going back...")
        self.browser_back()
        time.sleep(2)
        print("Going forward...")
        self.browser_forward()
        time.sleep(2)
        print("Closing tab...")
        self.close_tab()
        print("Test complete.")

    def browser_back(self):
        """Navigate back in browser history"""
        print("Action: Going back - sending Alt+Left")
        try:
                # Alternative methods
            pyautogui.hotkey('alt', 'left')
                # If that doesn't work, try browser-specific shortcuts
                # pyautogui.hotkey('backspace')  # Alternative for some browsers
        except Exception as e:
            print(f"Error executing back command: {e}")

    def browser_forward(self):
        """Navigate forward in browser history"""
        print("Action: Going forward - sending Alt+Right")
        try:
            pyautogui.hotkey('alt', 'right')
                # Alternative: pyautogui.hotkey('shift', 'backspace')
        except Exception as e:
            print(f"Error executing forward command: {e}")

    def scroll_up(self):
        """Scroll page up"""
        print("Action: Scrolling up")
        pyautogui.scroll(300)  # Positive value scrolls up

    def scroll_down(self):
        """Scroll page down"""
        print("Action: Scrolling down")
        pyautogui.scroll(-300)  # Negative value scrolls down

    def new_tab(self):
        """Open a new tab"""
        print("Action: Opening new tab - sending Ctrl+T")
        try:
                # Try pressing keys with slight delay between them
            pyautogui.keyDown('ctrl')
            pyautogui.press('t')
            pyautogui.keyUp('ctrl')
        except Exception as e:
            print(f"Error opening new tab: {e}")

    def close_tab(self):
        """Close the current tab"""
        print("Action: Closing tab - sending Ctrl+W")
        try:
            pyautogui.keyDown('ctrl')
            pyautogui.press('w')
            pyautogui.keyUp('ctrl')
        except Exception as e:
            print(f"Error closing tab: {e}")

    def refresh_page(self):
        """Refresh the current page"""
        print("Action: Refreshing page - sending F5")
        try:
                # Alternative:
            pyautogui.keyDown('ctrl')
            pyautogui.press('r')
            pyautogui.keyUp('ctrl')
        except Exception as e:
            print(f"Error refreshing page: {e}")

    def zoom_in(self):
        """
           Zoom in on the page
        """
        print("Action: Zooming In")
        pyautogui.hotkey('ctrl', '+')

    def zoom_out(self):
        """Zoom out on the page"""
        print("Action: Zooming out")
        pyautogui.hotkey('ctrl', '-')

    def switch_tab(self):
        """Switch to the next tab"""
        print("Action: Switching to next tab")
        pyautogui.hotkey('ctrl', 'tab')

    def execute_gesture_action(self, gesture):
        """
        Execute the corresponding action for a detected gesture

        Parameters:
        gesture (Gest): Detected gesture

        Returns:
        bool: True if action was executed
        """
        current_time = time.time()

        # Check if we're still in cooldown period
        if current_time - self.last_action_time < self.action_cooldown:
            return False

        if gesture in self.gesture_actions:
            try:
                if gesture in [Gest.SWIPE_LEFT, Gest.SWIPE_RIGHT, Gest.THUMBS_UP,
                                          Gest.THUMBS_DOWN, Gest.V_GEST]:

                    browser = self.detect_browser()
                    if browser == 'unknown':
                        print("Warning: No browser detected for browser-specific action")
                                    # Could display a notification to the user here

            # Execute the action associated with the gesture
                self.gesture_actions[gesture]()
                self.last_action_time = current_time
                return True
            except Exception as e:
                print(f"Error executing action: {e}")

                if hasattr(self, f"{gesture.name.lower()}_fallback"):
                    try:
                        getattr(self, f"{gesture.name.lower()}_fallback")()
                        return True
                    except:
                        return False

                return False

        return False

    def process_hands(self, results):
        """
        Process hand landmarks and execute corresponding actions

        Parameters:
        results: MediaPipe hand detection results

        Returns:
        tuple: (frame with visualization, detected gesture)
        """
        detected_gesture = None

        # Classify hands (left/right and major/minor)
        if results.multi_handedness and results.multi_hand_landmarks:
            # Initialize hands if not done already
            if self.hand_major is None:
                self.hand_major = HandRecognizer(HLabel.MAJOR)

            if self.hand_minor is None:
                self.hand_minor = HandRecognizer(HLabel.MINOR)

            # Process each detected hand
            for idx, handedness in enumerate(results.multi_handedness):
                # Check if we have landmarks for this hand
                if idx < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[idx]

                    # Determine if this is right or left hand
                    label = handedness.classification[0].label

                    # Update appropriate hand recognizer
                    # Assuming right hand is major (dominant) hand
                    if label == "Right":
                        self.hand_major.update_hand_result(hand_landmarks)
                        self.hand_major.set_finger_state()
                        detected_gesture = self.hand_major.get_gesture()
                    else:
                        self.hand_minor.update_hand_result(hand_landmarks)
                        self.hand_minor.set_finger_state()
                        minor_gesture = self.hand_minor.get_gesture()

                        # Only use minor hand gesture if major hand didn't detect anything important
                        if detected_gesture is None or detected_gesture == Gest.PALM:
                            detected_gesture = minor_gesture

        return detected_gesture

    def run(self):
        """Run the browser controller with webcam input"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Browser control started. Press 'q' to quit.")
        print("Available gestures:")
        print("- Swipe left: Go back")
        print("- Swipe right: Go forward")
        print("- Swipe up: Scroll up")
        print("- Swipe down: Scroll down")
        print("- Thumbs up: New tab")
        print("- Thumbs down: Close tab")
        print("- V gesture: Refresh page")

        # Initialize MediaPipe Hands
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            while self.running:
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture image")
                    break

                # Flip the image horizontally for a more natural feel
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe Hands
                results = hands.process(rgb_frame)

                # Process hand landmarks and get detected gestures
                detected_gesture = self.process_hands(results)

                # Draw hand landmarks on the frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                # Execute action if a gesture is detected
                if detected_gesture:
                    success = self.execute_gesture_action(detected_gesture)
                    if success:
                        # Add a visual feedback (green rectangle around the frame)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)

                        # Display the detected gesture
                        gesture_name = str(detected_gesture).split('.')[1]
                        cv2.putText(
                            frame,
                            f"Gesture: {gesture_name}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                # Display the system status
                cv2.putText(
                    frame,
                    "Browser Control Active",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Display the frame
                cv2.imshow('Browser Gesture Control', frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Browser control stopped.")


def main():
    """Main function to run the browser controller"""
    controller = BrowserController()
    controller.run()


if __name__ == "__main__":
    main()
