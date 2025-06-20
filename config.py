import numpy as np
# === Control & Execution Flags ===
USE_GRAVITY_COMP = False  # Toggle PD vs PD+GC
CONTROL_MODE = "pbvs"    # For future vision mode expansion
VISION_MODE = "image"    # "image" or "pose"

# === Target color settings (used for vision only) ===
TARGET_COLOR = 'red'
ENDEFFECTOR_COLOR = 'blue'

TARGET_PIXEL_OFFSET = (0, 40)
ENDEFFECTOR_PIXEL_OFFSET = (0, 0)
MIN_BLOB_SIZE = 10

COM_PORT = 'COM4'

# === Joint Angle Limits (rad) ===
JOINT_LIMITS = [
    (-np.pi, np.pi),        # Joint 1 (base)
    (-np.pi/2, np.pi/2),    # Joint 2
    (-np.pi/2, np.pi/2),    # Joint 3
    (-np.pi/2, np.pi/2),    # Joint 4
]
# JOINT_LIMITS = [ (0, 360),     # Joint 1 (base) (80, 290),    # Joint 2 (40, 290),    # Joint 3 (40, 290),    # Joint 4 ]

# === Initial Safe Joint Angles ===
INITIAL_POSITION_DEG = [180, 180, 180, 180]  # Safe upright home position
FINAL_POSITION_DEG = [160, 160, 160, 160]

# === DH Parameters (meters) ===
L1 = 0.05    # base height
L2 = 0.0     # no horizontal offset
L3 = 0.1635  # link 3 length
L4 = 0.145   # link 4 length
JOINT_LENGTHS = [L1, L2, L3, L4]

d1 = 0.00
d2 = 0.0935
d3 = 0.00
d4 = 0.00
JOINT_OFFSET = [d1, d2, d3, d4]

alpha1 = 0.0
alpha2 = np.deg2rad(90)
alpha3 = 0.0
alpha4 = 0.0
JOINT_TWIST = [alpha1, alpha2, alpha3, alpha4]


# === Mass and COM Info ===
# === Link and Motor Masses ===
link_masses = [0.29, 0.029, 0.029, 0.01]  # kg
motor_mass = 0.077  # kg

# Assume each link has a motor mounted at its proximal end
m1 = link_masses[0] + motor_mass  # base motor + link1
m2 = link_masses[1] + motor_mass  # motor at joint2 + link2
m3 = link_masses[2] + motor_mass  # motor at joint3 + link3
m4 = link_masses[3] + motor_mass  # motor at joint4 + link4




link_COM_fractions = [
    (link_masses[0] * 0.5 + motor_mass * 1.0) / (link_masses[0] + motor_mass),
    (link_masses[1] * 0.5 + motor_mass * 1.0) / (link_masses[1] + motor_mass),
    (link_masses[2] * 0.5 + motor_mass * 1.0) / (link_masses[2] + motor_mass),
    (link_masses[3] * 0.5 + motor_mass * 1.0) / (link_masses[3] + motor_mass),
]

COM_FRACTIONS = link_COM_fractions  # Export for FK use

# Center of mass distance from joint origin (meters)
lc1 = link_COM_fractions[0] * L1
lc2 = link_COM_fractions[1] * L2
lc3 = link_COM_fractions[2] * L3
lc4 = link_COM_fractions[3] * L4

# === Gravity Constant ===
g = 9.81