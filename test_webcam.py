#test_webcam.py

import cv2
import numpy as np
import time

# ----------- USER SETTINGS ------------

TARGET_COLOR = 'blue'       # 'blue' or 'red'
VERTICAL_OFFSET_PX = 40     # Move target this many pixels ABOVE the topmost one
MIN_BLOB_SIZE = 10          # Minimum number of connected pixels for valid detection


# ----------- COLOR RANGES IN HSV ------------
color_ranges = {
    'blue': {
        'lower': np.array([90, 50, 70]),
        'upper': np.array([140, 255, 255])
    },
    'red': {
        'lower1': np.array([0, 120, 70]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([170, 120, 70]),
        'upper2': np.array([180, 255, 255])
    }
}

# ----------- CAMERA SETUP ------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()


print(f"🔍 Tracking topmost {TARGET_COLOR} pixel with vertical offset of {VERTICAL_OFFSET_PX}px...")
last_feedback_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask
    if TARGET_COLOR == 'red':
        mask1 = cv2.inRange(hsv, color_ranges['red']['lower1'], color_ranges['red']['upper1'])
        mask2 = cv2.inRange(hsv, color_ranges['red']['lower2'], color_ranges['red']['upper2'])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, color_ranges[TARGET_COLOR]['lower'], color_ranges[TARGET_COLOR]['upper'])

    # --- Filter out small blobs ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_SIZE:
            filtered_mask[labels == i] = 255

    # Find topmost pixel in filtered mask
    coords = cv2.findNonZero(filtered_mask)

    if coords is not None:
        topmost = min(coords, key=lambda pt: pt[0][1])
        x_top, y_top = topmost[0]

        y_offset = max(0, y_top - VERTICAL_OFFSET_PX)

        # Draw markers
        cv2.circle(frame, (x_top, y_top), 5, (0, 255, 255), -1)  # original
        cv2.circle(frame, (x_top, y_offset), 5, (255, 0, 0), -1)  # offset target

        now = time.time()
        if now - last_feedback_time > 2:
            print(f"[{time.strftime('%H:%M:%S')}] Topmost {TARGET_COLOR} pixel: x={x_top}, y={y_top} → Target w/ offset: x={x_top}, y={y_offset}")
            last_feedback_time = now
    else:
        now = time.time()
        if now - last_feedback_time > 2:
            print(f"[{time.strftime('%H:%M:%S')}] No {TARGET_COLOR} pixel found.")
            last_feedback_time = now

    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Mask', filtered_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting webcam stream.")
        break

cap.release()
cv2.destroyAllWindows()
