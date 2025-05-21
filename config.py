import numpy as np
CONTROL_MODE = "ibvs"      # Options: "ibvs", "pbvs", "hybrid"
VISION_MODE = "image"      # Options: "image", "pose"

# Color tracking parameters
TARGET_COLOR = 'red'         # or 'red'
ENDEFFECTOR_COLOR = 'blue' 

TARGET_PIXEL_OFFSET = (0, 40)       # (horizontal, vertical) offset of target in pixels
ENDEFFECTOR_PIXEL_OFFSET = (0, 0)   # Optional if needed for adjusting EE marker location

MIN_BLOB_SIZE = 10            # minimum size of color blob to count as valid

# Joint angle limits in radians (min, max) for each joint
JOINT_LIMITS = [
    (-np.pi, np.pi),        # Joint 1 INSERT REAL VALUES
    (-np.pi/2, np.pi/2),    # Joint 2
    (-np.pi/2, np.pi/2),    # Joint 3
    (-np.pi/2, np.pi/2),    # Joint 4
]


#DH parameters
L1 = 0.15 #get real values
L2 = 0.12
L3 = 0.10
L4 = 0.08

d1 = 0.07 # get real values m
d2 = 0
d3 = 0
d4 = 0.03

alpha1 = 90
alpha2 = 0
alpha3 = 0
alpha4 = 0


link_masses = [0.4, 0.3, 0.2, 0.1] # Get real values kg
m1, m2, m3, m4 = link_masses
g = 9.81



