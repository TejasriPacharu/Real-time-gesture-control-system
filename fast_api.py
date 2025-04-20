from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from contextlib import asynccontextmanager
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Import our browser control module
from browser_control_fast import BrowserController, Gest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gesture_control")

# Create a directory for static files if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up")
    yield
    # Shutdown logic
    gesture_processor.stop_camera()
    logger.info("Application shutting down")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Gesture Browser Control",
    lifespan=lifespan
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# Initialize connection manager
manager = ConnectionManager()

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Video processor class
class GestureProcessor:
    def __init__(self):
        self.browser_controller = BrowserController()
        self.cap = None
        self.is_running = False
        self.last_frame_time = 0
        self.fps = 15  # Target FPS for processing
        self.frame_interval = 1.0 / self.fps

        # Initialize MediaPipe with the correct approach
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            model_complexity=1
        )

    def initialize_camera(self, camera_id=0):
        """Initialize the camera with the given ID"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            logger.error(f"Could not open camera with ID {camera_id}")
            return False

        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return True

    def stop_camera(self):
        """Stop and release the camera"""
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def process_frame(self, frame):
        """Process a single frame with MediaPipe and detect gestures"""
        if frame is None:
            return None, None

        # Flip the image horizontally for a more natural feel (mirror effect)
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Process hand landmarks and detect gestures
        detected_gesture = self.browser_controller.process_hands(results)

        # Display the detected gesture on the frame
        if detected_gesture:
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

            # Execute the action if a gesture is detected
            success = self.browser_controller.execute_gesture_action(detected_gesture)
            if success:
                # Add a visual feedback (green rectangle)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)

                # Log the executed gesture
                logger.info(f"Executed action for gesture: {gesture_name}")

                return frame, {"gesture": gesture_name, "action_executed": True}
            else:
                return frame, {"gesture": gesture_name, "action_executed": False}

        return frame, None

    def encode_frame(self, frame):
        """Encode a frame to JPEG format and then to base64 for sending over WebSocket"""
        if frame is None:
            return None

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            return None

        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

# Initialize gesture processor
gesture_processor = GestureProcessor()

# Function to handle frame processing in an async loop
async def process_frames(websocket: WebSocket):
    gesture_processor.is_running = True

    try:
        if gesture_processor.cap is None or not gesture_processor.cap.isOpened():
            if not gesture_processor.initialize_camera():
                await websocket.send_json({
                    "error": "Could not initialize camera"
                })
                return

        while gesture_processor.is_running:
            # Limit processing rate to avoid overwhelming the system
            current_time = time.time()
            if current_time - gesture_processor.last_frame_time < gesture_processor.frame_interval:
                await asyncio.sleep(0.001)  # Small sleep to not block the event loop
                continue

            gesture_processor.last_frame_time = current_time

            # Read frame from the camera
            ret, frame = gesture_processor.cap.read()
            if not ret:
                logger.error("Failed to get frame from camera")
                await asyncio.sleep(0.1)
                continue

            # Process the frame
            processed_frame, gesture_info = gesture_processor.process_frame(frame)

            # Encode the processed frame
            encoded_frame = gesture_processor.encode_frame(processed_frame)
            if encoded_frame is None:
                continue

            # Prepare the message to send
            message = {
                "frame": encoded_frame,
                "timestamp": time.time()
            }

            # Add gesture information if available
            if gesture_info:
                message["gesture"] = gesture_info

            # Send the frame to the client
            await websocket.send_json(message)

            # Small sleep to let other tasks run
            await asyncio.sleep(0.001)

    except Exception as e:
        logger.error(f"Error in frame processing: {str(e)}")

    finally:
        gesture_processor.stop_camera()

# Define a route for the HTML index page
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

# WebSocket endpoint for real-time video streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        # First, send a welcome message
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to Gesture Control Server"
        })

        # Start processing frames from the camera
        processing_task = asyncio.create_task(process_frames(websocket))

        # Handle incoming messages from the client
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "start":
                logger.info("Client requested to start gesture recognition")
                if not gesture_processor.is_running:
                    gesture_processor.is_running = True
                    processing_task = asyncio.create_task(process_frames(websocket))

            elif message.get("type") == "stop":
                logger.info("Client requested to stop gesture recognition")
                gesture_processor.is_running = False
                if not processing_task.done():
                    processing_task.cancel()

            elif message.get("type") == "test_actions":
                logger.info("Client requested to test browser actions")
                gesture_processor.browser_controller.test_browser_actions()
                await websocket.send_json({
                    "type": "test_complete",
                    "message": "Browser actions test completed"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
        gesture_processor.is_running = False
        gesture_processor.stop_camera()

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        gesture_processor.is_running = False
        gesture_processor.stop_camera()

if __name__ == "__main__":
    # Ensure the static directory exists
    if not static_dir.exists():
        static_dir.mkdir(parents=True)

    # Copy the HTML file to the static directory
    html_path = static_dir / "index.html"
