

import numpy as np
from config import (
    JOINT_LENGTHS,
    JOINT_OFFSET,
    JOINT_TWIST,
    COM_FRACTIONS
)

def dh_transform(a, alpha, d, theta):
    """
    Compute the homogeneous transformation matrix using DH parameters.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                 d],
        [0,              0,                             0,                             1]
    ])

def compute_forward_kinematics(q, return_all_links=False, com_only=False):
    """
    Compute the end-effector position or all link transforms/COMs.

    Parameters:
        q (np.ndarray): Joint angles [q1, q2, q3, q4] in radians
        return_all_links (bool): If True, return list of link transforms or COMs
        com_only (bool): If True, return COM positions instead of full transforms

    Returns:
        np.ndarray or list: EE position (3,), or list of COM positions (4,3), or transforms (4,4,4)
    """
    assert len(q) == 4
    T = np.eye(4)
    results = []

    for i in range(4):
        a = JOINT_LENGTHS[i]
        d = JOINT_OFFSET[i]
        alpha = JOINT_TWIST[i]
        theta = q[i]

        T_link = dh_transform(a, alpha, d, theta)
        T = T @ T_link

        if return_all_links:
            if com_only:
                # Apply COM offset along local x-axis
                com_offset = a * COM_FRACTIONS[i]
                T_com = T @ np.array([
                    [1, 0, 0, com_offset],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                results.append(T_com[:3, 3])  # COM position
            else:
                results.append(T.copy())  # Full 4x4 transform

    if return_all_links:
        return results
    else:
        return T[:3, 3]  # EE position only

# === Test Harness ===
if __name__ == "__main__":
    q_test = np.radians([45, 30, -15, 10])
    print("End-effector position:", compute_forward_kinematics(q_test))

    print("Link COM positions:")
    coms = compute_forward_kinematics(q_test, return_all_links=True, com_only=True)
    for i, c in enumerate(coms):
        print(f"Link {i+1} COM:", c)
