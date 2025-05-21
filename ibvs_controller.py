import numpy as np
from gravity import compute_gravity_torque
from image_jacobian import compute_image_jacobian


def compute_ibvs_command(image_error, q, qdot, gain_scale=1.0):
    """
    Compute joint torques based on image error and gravity compensation (camera 1 only).
    Assumes camera 1 provides error in the x-z plane.

    Parameters:
        image_error (np.ndarray): 2x1 array, pixel or normalized error in [x_img, z_img]
        q (np.ndarray): Joint angles [q1, q2, q3, q4] in radians
        qdot (np.ndarray): Joint velocities [q1_dot, ..., q4_dot]
        gain_scale (float): Optional gain multiplier

    Returns:
        np.ndarray: Joint torque command [tau1, tau2, tau3, tau4]

    """

    Kp = 0.4 * gain_scale

    # Get full 3x4 Jacobian from image_jacobian
    J = compute_image_jacobian(q)

    # Use rows 0 and 2 (x and z) from camera 1 side view
    Jxz = J[[0, 2], :]  # Shape (2, 4)

    # Compute desired joint velocity direction
    v_img = -Kp * image_error  # 2D desired motion in image plane

    # Pseudo-inverse to compute joint space command
    J_pinv = np.linalg.pinv(Jxz)  # Shape (4, 2)
    q_cmd = J_pinv @ v_img        # Shape (4,)

    # Gravity compensation
    tau = q_cmd + compute_gravity_torque(q)

    return tau


def compute_joint_command(image_error, q, qdot, gain_scale=1.0):
    """
    Returns joint velocity or torque vector from image-space error.
    For use in velocity-based control without gravity terms.
    """

    Kp = 0.4 * gain_scale
    J = compute_image_jacobian(q)
    Jxz = J[[0, 2], :]
    v_img = -Kp * image_error
    J_pinv = np.linalg.pinv(Jxz)
    q_cmd = J_pinv @ v_img
    return q_cmd
