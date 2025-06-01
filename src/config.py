# Config parameters for candle robot

# ----------- CONTROL SETTINGS ------------
CONTROL_MODE = "ibvs"      # Options: "ibvs", "pbvs", "hybrid"
VISION_MODE = "image"      # Options: "image", "pose"

# ----------- CAMERA SETTINGS ------------
color_to_detect = "orange"

# ----------- ROBOT PHYSICAL PARAMETERS ------------
L1 = 0.15 # link lengths (m)
L2 = 0.12
L3 = 0.10
L4 = 0.08
link_masses = [0.4, 0.3, 0.2, 0.1] # Get real values (kg)

# ----------- MOTOR PARAMETERS ------------
motor_1 = 1
motor_2 = 2
motor_3 = 3
motor_4 = 4

