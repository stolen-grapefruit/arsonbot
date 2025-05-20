def compute_joint_command(image_error, q, qdot, gain_scale=1.0):
    """
    Returns joint torque or velocity vector from image-space error.
    """
    
    
    Kp = 0.4 * gain_scale
    # TODO: Use image Jacobian (or axis-decoupled proportional control)
    # TODO: Optionally include gravity compensation
    
    return -Kp * image_error  # Placeholder
