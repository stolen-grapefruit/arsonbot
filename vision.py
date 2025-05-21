"""
vision.py - Computer vision module for tracking colored pixels from two cameras.
Camera 1: side view (x-z plane) → use x and z info.
Camera 2: front view (y-z plane) → use y info only.
"""

import cv2
import numpy as np
from config import TARGET_COLOR, VERTICAL_OFFSET_PX, MIN_BLOB_SIZE

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


def get_visual_info(mode="side", debug=True):
    """
    Returns image-space error vector for one or two camera views.
    mode: "side" = camera 1, "front" = camera 2, "dual" = both
    """
    cap1 = cv2.VideoCapture(0)  # Camera 1 (side view)
    cap2 = cv2.VideoCapture(1) if mode == "dual" else None

    ret1, frame1 = cap1.read()
    cap1.release()

    ret2, frame2 = (cap2.read() if cap2 else (None, None))
    if cap2: cap2.release()

    error_xz = None
    error_y = None

    if ret1:
        pixel_info_1, _ = get_top_pixel_from_frame(frame1)
        if pixel_info_1:
            x1, y1, x1_off, y1_off = pixel_info_1
            error_xz = np.array([x1 - x1_off, y1 - y1_off])  # [x_error, z_error]
            if debug:
                print(f"[Camera 1] x error: {x1 - x1_off}, z error: {y1 - y1_off}")

    if ret2 and mode == "dual":
        pixel_info_2, _ = get_top_pixel_from_frame(frame2)
        if pixel_info_2:
            x2, y2, _, _ = pixel_info_2
            error_y = np.array([y2])  # vertical in image = y-axis world
            if debug:
                print(f"[Camera 2] y error (approx): {error_y[0]}")

    if mode == "side":
        return {"error": error_xz}
    elif mode == "front":
        return {"error": error_y}
    elif mode == "dual":
        if error_xz is not None and error_y is not None:
            return {"error": np.array([error_xz[0], error_y[0], error_xz[1]])}  # [x, y, z]
        else:
            return {"error": None}


if __name__ == "__main__":
    # DEBUG camera 1 only
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pixel_info, mask = get_top_pixel_from_frame(frame)
        if pixel_info:
            x, y, x_offset, y_offset = pixel_info
            print(f"[Test] Camera 1 - Top: ({x},{y}), Offset: ({x_offset},{y_offset})")
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.circle(frame, (x_offset, y_offset), 5, (255, 0, 0), -1)

        cv2.imshow("Camera 1", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
