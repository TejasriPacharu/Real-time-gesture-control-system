# Real-time-gesture-control-system


## Problem Statement:

"Design a lightweight real-time gesture-based control system for browser navigation aimed at accessibility enhancement for individuals with motor disabilities."

##Project Overview:
Develop a webcam-based gesture recognition system that maps simple hand gestures (like swipe left/right, thumbs up/down, etc.) to common browser actions such as:

Going back or forward in history

Scrolling up/down

Opening/closing tabs

Refreshing page

This system would provide an alternative interface for users with limited mobility, helping them interact with the web without using a keyboard or mouse.

##Unique Ideology:

Unlike typical gesture recognition projects, this one focuses on low-resource environments, using lightweight models or OpenCV + MediaPipe, and emphasizes practical daily utility in web navigation — something rarely targeted in CV projects.

##Core Features:

Real-time webcam capture

Hand gesture detection using MediaPipe or YOLO-Nano/Tiny

Gesture-to-command mapping (e.g., swipe right → “next tab”)

System that hooks into browser events (with a simple browser extension or automation)

Minimal UI to display detected gesture + action

##Tools You Can Use:

Python + OpenCV + MediaPipe (for gesture detection)

Flask/FastAPI (for bridging gesture recognition to browser)

pyautogui / Selenium / Puppeteer (for browser control simulation)

##Optional: Create a tiny browser extension to send commands

Stretch Goals (if time permits):

Customizable gesture-action mapping via UI

Voice feedback for actions taken

Support for more gestures (pinch, circle, etc.)

Cross-platform support
