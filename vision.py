"""
vision.py - Computer vision module for tracking the topmost color pixel.
Modular version of the test_webcam script for use in control loops.
"""

import cv2
import numpy as np
from config import TARGET_COLOR, VERTICAL_OFFSET_PX, MIN_BLOB_SIZE

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

def get_top_pixel_from_frame(frame, target_color=TARGET_COLOR):
    """
    Returns the topmost pixel (with vertical offset) of a given color from a frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if target_color == 'red':
        mask1 = cv2.inRange(hsv, color_ranges['red']['lower1'], color_ranges['red']['upper1'])
        mask2 = cv2.inRange(hsv, color_ranges['red']['lower2'], color_ranges['red']['upper2'])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, color_ranges[target_color]['lower'], color_ranges[target_color]['upper'])

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_SIZE:
            filtered_mask[labels == i] = 255

    coords = cv2.findNonZero(filtered_mask)
    if coords is not None:
        topmost = min(coords, key=lambda pt: pt[0][1])
        x_top, y_top = topmost[0]
        y_offset = max(0, y_top - VERTICAL_OFFSET_PX)
        return (x_top, y_top, x_top, y_offset), filtered_mask
    else:
        return None, filtered_mask

def show_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print(f"🔍 Tracking topmost {TARGET_COLOR} pixel with vertical offset of {VERTICAL_OFFSET_PX}px...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, mask = get_top_pixel_from_frame(frame)
        if result:
            x_top, y_top, x_off, y_off = result
            cv2.circle(frame, (x_top, y_top), 5, (0, 255, 255), -1)
            cv2.circle(frame, (x_off, y_off), 5, (255, 0, 0), -1)
            print(f"Detected: x={x_top}, y={y_top} | Offset target: x={x_off}, y={y_off}")
        else:
            print("No valid pixel found.")

        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Filtered Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera_feed()
