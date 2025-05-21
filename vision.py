"""
vision.py - Optimized computer vision for ARSONBOT.
- Tracks target and end-effector markers.
- Supports camera 1 (side xz) and optional camera 2 (yz).
- Includes frame rate limiting, downsampling, and live debug visualization.
"""

import cv2
import numpy as np
import time
from config import TARGET_COLOR, ENDEFFECTOR_COLOR, TARGET_PIXEL_OFFSET, ENDEFFECTOR_PIXEL_OFFSET, MIN_BLOB_SIZE

# ---------- USER TUNABLE PARAMETERS ----------
DEBUG_VISUAL = True
VISION_FPS = 10
FRAME_SCALE = 0.5
CAMERA_MODE = "side"  # options: "side", "dual"

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

def get_top_pixel_from_frame(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == 'red':
        mask1 = cv2.inRange(hsv, color_ranges['red']['lower1'], color_ranges['red']['upper1'])
        mask2 = cv2.inRange(hsv, color_ranges['red']['lower2'], color_ranges['red']['upper2'])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, color_ranges[color]['lower'], color_ranges[color]['upper'])

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_SIZE:
            filtered_mask[labels == i] = 255

    coords = cv2.findNonZero(filtered_mask)
    if coords is not None:
        topmost = min(coords, key=lambda pt: pt[0][1])
        x_top, y_top = topmost[0]
        return (x_top, y_top), filtered_mask
    else:
        return None, filtered_mask


def get_visual_info(mode="side", debug=True):
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1) if mode == "dual" else None

    ret1, frame1 = cap1.read()
    cap1.release()

    ret2, frame2 = (cap2.read() if cap2 else (None, None))
    if cap2: cap2.release()

    error_xz = None
    error_y = None

    if ret1:
        frame1 = cv2.resize(frame1, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        pixel_target, _ = get_top_pixel_from_frame(frame1, TARGET_COLOR)
        pixel_ee, _ = get_top_pixel_from_frame(frame1, ENDEFFECTOR_COLOR)
        if pixel_target and pixel_ee:
            xt, zt = pixel_target
            xe, ze = pixel_ee

            xt += TARGET_PIXEL_OFFSET[0]
            zt += TARGET_PIXEL_OFFSET[1]
            xe += ENDEFFECTOR_PIXEL_OFFSET[0]
            ze += ENDEFFECTOR_PIXEL_OFFSET[1]

            error_xz = np.array([xe - xt, ze - zt])

            if debug:
                print(f"[Camera 1] Target: ({xt}, {zt}), EE: ({xe}, {ze})")
                print(f"[Camera 1] x error: {error_xz[0]}, z error: {error_xz[1]}")

    if ret2 and mode == "dual":
        frame2 = cv2.resize(frame2, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        pixel_target2, _ = get_top_pixel_from_frame(frame2, TARGET_COLOR)
        pixel_ee2, _ = get_top_pixel_from_frame(frame2, ENDEFFECTOR_COLOR)
        if pixel_target2 and pixel_ee2:
            yt = pixel_target2[1] + TARGET_PIXEL_OFFSET[1]
            ye = pixel_ee2[1] + ENDEFFECTOR_PIXEL_OFFSET[1]
            error_y = np.array([ye - yt])
            if debug:
                print(f"[Camera 2] y error: {error_y[0]}")

    if mode == "side":
        return {"error": error_xz}
    elif mode == "front":
        return {"error": error_y}
    elif mode == "dual":
        if error_xz is not None and error_y is not None:
            return {"error": np.array([error_xz[0], error_y[0], error_xz[1]])}
        else:
            return {"error": None}


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        pixel_target, mask = get_top_pixel_from_frame(frame, TARGET_COLOR)
        pixel_ee, _ = get_top_pixel_from_frame(frame, ENDEFFECTOR_COLOR)

        if DEBUG_VISUAL:
            if pixel_target:
                xt, zt = pixel_target
                xt += TARGET_PIXEL_OFFSET[0]
                zt += TARGET_PIXEL_OFFSET[1]
                cv2.circle(frame, (xt, zt), 5, (0, 0, 255), -1)
            if pixel_ee:
                xe, ze = pixel_ee
                xe += ENDEFFECTOR_PIXEL_OFFSET[0]
                ze += ENDEFFECTOR_PIXEL_OFFSET[1]
                cv2.circle(frame, (xe, ze), 5, (255, 0, 0), -1)

            cv2.imshow("Camera 1", frame)
            cv2.imshow("Mask", mask)

        elapsed = time.time() - loop_start
        time.sleep(max(0, (1 / VISION_FPS) - elapsed))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
