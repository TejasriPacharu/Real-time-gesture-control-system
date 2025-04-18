import cv2
import mediapipe as mp
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the updated BrowserController
from browser_control import BrowserController, Gest

class GestureControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Browser Control")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize the browser controller
        self.browser_controller = BrowserController()

        # Set up the UI
        self.setup_ui()

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_message("Error: Could not open webcam.")
            return

        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Variables
        self.is_running = False
        self.thread = None

        # Set up a flag to indicate if the app is closing
        self.is_closing = False

        # Start with system disabled
        self.toggle_system()

    def setup_ui(self):
        """Set up the application UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top control frame
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X)

        # Toggle button
        self.toggle_btn = ttk.Button(
            control_frame,
            text="Enable Gesture Control",
            command=self.toggle_system
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(
            control_frame,
            text="System: Disabled",
            foreground="red"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Middle split frame for video and log
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Video frame (left side)
        video_frame = ttk.LabelFrame(middle_frame, text="Camera Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log frame (right side)
        log_frame = ttk.LabelFrame(middle_frame, text="Activity Log")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.log_text = tk.Text(log_frame, height=10, width=40, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom frame for gesture reference
        bottom_frame = ttk.LabelFrame(main_frame, text="Gesture Reference")
        bottom_frame.pack(fill=tk.X, pady=(5, 0))

        # Create a table-like display for gestures and actions
        gesture_frame = ttk.Frame(bottom_frame)
        gesture_frame.pack(fill=tk.X, padx=5, pady=5)

        # Define gestures and their actions
        gestures = [
            ("Swipe Left", "Go Back"),
            ("Swipe Right", "Go Forward"),
            ("Swipe Up", "Scroll Up"),
            ("Swipe Down", "Scroll Down"),
            ("Thumbs Up", "New Tab"),
            ("Thumbs Down", "Close Tab"),
            ("V Gesture", "Refresh Page")
        ]

        # Create headers
        ttk.Label(gesture_frame, text="Gesture", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(gesture_frame, text="Action", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        # Add separator
        ttk.Separator(gesture_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

        # Fill in gesture-action pairs
        for i, (gesture, action) in enumerate(gestures):
            ttk.Label(gesture_frame, text=gesture).grid(row=i+2, column=0, padx=5, pady=2, sticky="w")
            ttk.Label(gesture_frame, text=action).grid(row=i+2, column=1, padx=5, pady=2, sticky="w")

    def toggle_system(self):
        """Toggle the gesture control system on/off"""
        if self.is_running:
            # Stop the system
            self.is_running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            self.toggle_btn.config(text="Enable Gesture Control")
            self.status_label.config(text="System: Disabled", foreground="red")
            self.log_message("Gesture control disabled.")
        else:
            # Start the system
            self.is_running = True
            self.thread = threading.Thread(target=self.run_gesture_control)
            self.thread.daemon = True  # Thread will close when main program exits
            self.thread.start()
            self.toggle_btn.config(text="Disable Gesture Control")
            self.status_label.config(text="System: Enabled", foreground="green")
            self.log_message("Gesture control enabled.")

    def run_gesture_control(self):
        """Run the gesture control system in a separate thread"""
        self.browser_controller.running = True

        while self.is_running and not self.is_closing:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            if not ret:
                self.log_message("Error: Failed to capture image")
                break

            # Flip the image horizontally for a more natural feel
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            results = self.hands.process(rgb_frame)

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

            # Process hand landmarks and detect gestures
            detected_gesture = self.browser_controller.process_hands(results)

            # Show gesture name on frame if detected
            if detected_gesture:
                try:
                    # Convert enum value to name
                    gesture_name = str(detected_gesture).split('.')[1]
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture_name}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # Log the detected gesture
                    self.log_message(f"Detected: {gesture_name}")

                    # Execute the action
                    success = self.browser_controller.execute_gesture_action(detected_gesture)
                    if success:
                        # Add visual feedback
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
                        self.log_message(f"Executed action for: {gesture_name}")
                except Exception as e:
                    self.log_message(f"Error processing gesture: {e}")

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

            # Update the video display
            self.update_video(frame)

            # Sleep briefly to reduce CPU usage
            time.sleep(0.01)

        self.browser_controller.running = False

    def update_video(self, frame):
        """Update the video display with the current frame"""
        # Convert frame to RGB format (for PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(frame_rgb)

        # Resize to fit the display
        width, height = 400, 300
        pil_img = pil_img.resize((width, height), Image.LANCZOS)

        # Convert to Tkinter PhotoImage
        tk_img = ImageTk.PhotoImage(image=pil_img)

        # Update the label with the new image
        self.video_label.configure(image=tk_img)
        self.video_label.image = tk_img  # Keep a reference to prevent garbage collection

    def log_message(self, message):
        """Add a message to the log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # Enable text widget, insert message, then disable again
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)  # Scroll to bottom
        self.log_text.configure(state=tk.DISABLED)

    def on_closing(self):
        """Handle window closing"""
        self.is_closing = True
        self.is_running = False

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Release the webcam and other resources
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.hands.close()
        self.root.destroy()

def main():
    # Create the main application window
    root = tk.Tk()
    app = GestureControlApp(root)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
