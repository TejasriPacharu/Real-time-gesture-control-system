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
        Detect thumbs down gesture

        Returns:
        bool: True if thumbs down detected
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

        # For tracking last detected gesture
        self.last_detected_gesture = None

        # Cooldown for actions to prevent multiple triggers
        self.last_action_time = 0
        self.action_cooldown = 1.0  # seconds

    def detect_browser(self):
        """
        Detect which browser is currently active to use appropriate shortcuts
        """
        # This is a simplified version as the original depends on OS-specific libraries
        # In a web-based version, we can't reliably detect the active browser
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
            pyautogui.hotkey('alt', 'left')
        except Exception as e:
            print(f"Error executing back command: {e}")

    def browser_forward(self):
        """Navigate forward in browser history"""
        print("Action: Going forward - sending Alt+Right")
        try:
            pyautogui.hotkey('alt', 'right')
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
        print("Action: Refreshing page - sending ctrl + R")
        try:
            pyautogui.keyDown('ctrl')
            pyautogui.press('r')
            pyautogui.keyUp('ctrl')
        except Exception as e:
            print(f"Error refreshing page: {e}")

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
                # Execute the action associated with the gesture
                self.gesture_actions[gesture]()
                self.last_action_time = current_time
                return True
            except Exception as e:
                print(f"Error executing action: {e}")
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

        # Update last detected gesture
        if detected_gesture:
            self.last_detected_gesture = detected_gesture

        return detected_gesture
