# Test script to check IK.py function, change values in test_ik_plot_3d()
# and run code to visualize robot position

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IK import compute_IK
from config import L1, L2, L3, L4

def forward_kinematics_3d(q):
    """
    Compute 3D positions of the joints for the full 4R arm using DH-style chaining.
    Args:
        q: list of 4 joint angles [theta1, theta2, theta3, theta4] in degrees

    Returns:
        points: list of 3D joint positions (numpy arrays)
    """
    theta1, theta2, theta3, theta4 = np.radians(q)

    # Base origin
    p0 = np.array([0, 0, 0])

    # Joint 1 (rotation around z0, offset along z by L1)
    p1 = p0 + np.array([0, 0, L1])

    # Joint 2 (rotates θ2 in x-z plane)
    x2 = L2 * np.cos(theta1) * np.cos(theta2)
    y2 = L2 * np.sin(theta1) * np.cos(theta2)
    z2 = L1 + L2 * np.sin(theta2)
    p2 = np.array([x2, y2, z2])

    # Joint 3
    theta23 = theta2 + theta3
    x3 = x2 + L3 * np.cos(theta1) * np.cos(theta23)
    y3 = y2 + L3 * np.sin(theta1) * np.cos(theta23)
    z3 = z2 + L3 * np.sin(theta23)
    p3 = np.array([x3, y3, z3])

    # End effector
    theta234 = theta23 + theta4
    x4 = x3 + L4 * np.cos(theta1) * np.cos(theta234)
    y4 = y3 + L4 * np.sin(theta1) * np.cos(theta234)
    z4 = z3 + L4 * np.sin(theta234)
    p4 = np.array([x4, y4, z4])

    return [p0, p1, p2, p3, p4]

def plot_robot_arm_3d(joint_positions, label, color, ax):
    xs, ys, zs = zip(*joint_positions)
    ax.plot(xs, ys, zs, 'o-', color=color, label=label)

def test_ik_plot_3d():
    # Desired end-effector position and orientation
    x, y, z = 0.1, 0.1, 0.05
    ee_angle_deg = -90

    # Compute IK solutions
    q_up, q_down = compute_IK(x, y, z, ee_angle_deg)

    # Compute FK to get joint positions
    joints_up = forward_kinematics_3d(q_up)
    joints_down = forward_kinematics_3d(q_down)

    # Setup 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D IK Visualization of 4R Robot Arm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot both configurations
    plot_robot_arm_3d(joints_up, 'Elbow-Up', 'blue', ax)
    plot_robot_arm_3d(joints_down, 'Elbow-Down', 'green', ax)

    # Plot target position
    ax.scatter([x], [y], [z], c='r', marker='x', s=100, label='Target EE Position')

    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    test_ik_plot_3d()
