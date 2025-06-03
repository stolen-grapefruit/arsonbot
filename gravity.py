"""
gravity.py - Gravity compensation torques for 4R manipulator

Computes torque needed at each joint to balance gravity, using link lengths and masses from config.py.
Assumes links are in a vertical plane and the arm is affected by standard gravity.
"""

import numpy as np
from config import m1, m2, m3, m4, g
from FK import compute_forward_kinematics


def compute_gravity_torque(q):
    """
    Calculate gravity compensation torques for a 4R manipulator.

    Parameters:
        q (np.ndarray): Joint angles [q1, q2, q3, q4] in radians

    Returns:
        np.ndarray: Torque vector [tau1, tau2, tau3, tau4]
    """
    
    # Get link transforms from FK
    transforms = compute_forward_kinematics(q, return_all_links=True)

    # Assume each transform includes position of center of mass of that link
    # transforms[i][:3] gives x, y, z position of COM of link i+1
    torques = np.zeros(4)

    for i in range(4):
        # Position vector from base to center of mass of link i
        r = transforms[i][:3]  # assuming FK returns list of 4 np.array([x,y,z])

        # Only consider torque from gravity force in z direction
        F = np.array([0, 0, -g * [m1, m2, m3, m4][i]])

        # Approximate torque as z-axis component of r x F (scalar projection)
        tau = np.cross(r, F)

        # Project onto z-axis torque (assumes all joints rotate about z)
        torques[i] = tau[2]  # simplification, adjust if joints are not all about z

    return torques


if __name__ == "__main__":
    q_example = np.radians([30, 20, -15, 10])
    tau = compute_gravity_torque(q_example)
    print("Gravity torques [Nm]:", tau)


