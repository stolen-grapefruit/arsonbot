from gravity import compute_gravity_torque

def compute_ibvs_command(image_error, q, qdot):
    # 1. Compute image-space control law → desired task-space velocity
    # 2. Map that to joint torques or joint velocities
    # 3. Add gravity compensation
    tau = ...  # control torques from IBVS logic
    tau += compute_gravity_torque(q)
    return tau



def compute_joint_command(image_error, q, qdot, gain_scale=1.0):
    """
    Returns joint torque or velocity vector from image-space error.
    """
    
    
    Kp = 0.4 * gain_scale
    # TODO: Use image Jacobian (or axis-decoupled proportional control)

    # TODO: Optionally include gravity compensation

    return -Kp * image_error  # Placeholder
