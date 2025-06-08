# Imports
import cv2
import numpy as np
import time
from color_mask import create_mask

# Outside directory imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import color_to_detect, fx, fy, cx, cy

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def get_contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

# Added function to convert to world frame
def pixel_to_camera_coords(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

# ----------- CAMERA SETUP ------------

print("Checking for camera...")

cap = None
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
    print(f"Trying backend {backend} for camera index 0")
    temp_cap = cv2.VideoCapture(0, backend)
    time.sleep(1)
    if temp_cap.isOpened():
        ret, frame = temp_cap.read()
        if ret and frame is not None and frame.size > 0:
            cap = temp_cap
            print(f"Success with backend {backend}")
            break
        temp_cap.release()
        print(f"Backend {backend} opened but failed to read frame.")
    else:
        print(f"Failed to open camera with backend {backend}")

if cap is None or not cap.isOpened():
    print("Could not open webcam. Exiting.")
    exit()

print("Camera successfully opened, starting video stream...")
time.sleep(1)

last_feedback_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("Frame grab failed, exiting...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = create_mask(hsv, color_to_detect)

    largest_contour = get_largest_contour(color_mask)

    if largest_contour is not None:
        centroid = get_contour_centroid(largest_contour)
        if centroid:
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            now = time.time()
            if now - last_feedback_time > 2:
                print(f"Found {color_to_detect} centroid:")
                print(f"[{time.strftime('%H:%M:%S')}] Blob centroid: x={centroid[0]}, y={centroid[1]}")
                last_feedback_time = now

                ### ADDING CONVERSION
                Z = 21 # inches
                world_coords = pixel_to_camera_coords(centroid[0], centroid[1], Z, fx, fy, cx, cy)
                print(f"[{time.strftime('%H:%M:%S')}] World Coordinates: X={world_coords[0]} in, Y={world_coords[1]} in, Z={world_coords[2]} in")
        else:
            print("Could not compute centroid.")
    else:
        now = time.time()
        if now - last_feedback_time > 2:
            print(f"[{time.strftime('%H:%M:%S')}] No blob found.")
            last_feedback_time = now

    cv2.imshow('Webcam Feed', frame)
    cv2.imshow('Color Mask', color_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting webcam stream.")
        break

cap.release()
cv2.destroyAllWindows()
