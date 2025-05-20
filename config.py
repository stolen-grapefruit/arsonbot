CONTROL_MODE = "ibvs"      # Options: "ibvs", "pbvs", "hybrid"
VISION_MODE = "image"      # Options: "image", "pose"

# Color tracking parameters
TARGET_COLOR = 'blue'         # or 'red'
VERTICAL_OFFSET_PX = 40       # pixels above topmost pixel to aim for
MIN_BLOB_SIZE = 10            # minimum size of color blob to count as valid


L1 = 0.15
L2 = 0.12
L3 = 0.10
L4 = 0.08

link_masses = [0.4, 0.3, 0.2, 0.1] # Get real values



