from config import L1, L2, L3, L4
import numpy as np

def compute_IK(x, y, z, ee_angle_deg):
    """
    Compute inverse kinematics for the 4R arm.
    Args:
        x, y, z: desired end-effector position in workspace
        ee_angle_deg: planar angle of end-effector from the horizontal

    Returns:
        q_up: np.array of joint angles [q1, q2, q3, q4] in degrees (elbow-up)
        q_down: np.array of joint angles [q1, q2, q3, q4] in degrees (elbow-down)
    """
    # Convert EE angle to radians
    ee_angle = np.radians(ee_angle_deg)

    # Theta 1 from base plane
    theta1 = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)

    # Wrist position (relative to Joint 2 origin)
    wx = rho - L4 * np.cos(ee_angle)
    wz = z - L4 * np.sin(ee_angle) - L1

    # Solve planar 2R IK for L2 and L3
    D = (wx**2 + wz**2 - L2**2 - L3**2) / (2 * L2 * L3)
    if abs(D) > 1.0:
        raise ValueError("Target out of reach")

    # Elbow-up solution
    theta3_up = np.arctan2(np.sqrt(1 - D**2), D)
    k1_up = L2 + L3 * np.cos(theta3_up)
    k2_up = L3 * np.sin(theta3_up)
    theta2_up = np.arctan2(wz, wx) - np.arctan2(k2_up, k1_up)
    theta4_up = ee_angle - (theta2_up + theta3_up)

    # Elbow-down solution
    theta3_down = np.arctan2(-np.sqrt(1 - D**2), D)
    k1_down = L2 + L3 * np.cos(theta3_down)
    k2_down = L3 * np.sin(theta3_down)
    theta2_down = np.arctan2(wz, wx) - np.arctan2(k2_down, k1_down)
    theta4_down = ee_angle - (theta2_down + theta3_down)

    # Combine results and convert to degrees
    q_up = np.degrees([theta1, theta2_up, theta3_up, theta4_up])
    q_down = np.degrees([theta1, theta2_down, theta3_down, theta4_down])

    return q_up, q_down