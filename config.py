CONTROL_MODE = "ibvs"      # Options: "ibvs", "pbvs", "hybrid"
VISION_MODE = "image"      # Options: "image", "pose"

# Color tracking parameters
TARGET_COLOR = 'blue'         # or 'red'
VERTICAL_OFFSET_PX = 40       # pixels above topmost pixel to aim for
MIN_BLOB_SIZE = 10            # minimum size of color blob to count as valid


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



