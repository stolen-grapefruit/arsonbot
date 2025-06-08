import cv2
import numpy as np

# Define HSV ranges for each color (using double masks, even if duplicated)
COLOR_HSV_RANGES = {
    "red": [  # Hue wraps around at 0, so red has two distinct ranges
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([170, 100, 50]), np.array([180, 255, 255])),
    ],
    "orange": [  # Typically doesn't need two masks
        (np.array([11, 100, 50]), np.array([20, 255, 255])),
        (np.array([11, 100, 50]), np.array([20, 255, 255])),
    ],
    "yellow": [
        (np.array([21, 100, 50]), np.array([30, 255, 255])),
        (np.array([21, 100, 50]), np.array([30, 255, 255])),
    ],
    "green": [  # Split to cover light to dark green if needed
        (np.array([35, 100, 50]), np.array([60, 255, 255])),
        (np.array([61, 100, 50]), np.array([85, 255, 255])),
    ],
    "blue": [  # Split for sky blue and deep blue
        (np.array([90, 100, 50]), np.array([110, 255, 255])),
        (np.array([111, 100, 50]), np.array([130, 255, 255])),
    ],
    "purple": [  # Covers violet to magenta
        (np.array([131, 100, 50]), np.array([145, 255, 255])),
        (np.array([146, 100, 50]), np.array([160, 255, 255])),
    ],
}


def create_mask(hsv_image, color_name):
    """Creates a binary mask for a specified color from an HSV image."""
    if color_name not in COLOR_HSV_RANGES:
        raise ValueError(f"Unsupported color: {color_name}")

    (lower1, upper1), (lower2, upper2) = COLOR_HSV_RANGES[color_name]
    mask1 = cv2.inRange(hsv_image, lower1, upper1)
    mask2 = cv2.inRange(hsv_image, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)
