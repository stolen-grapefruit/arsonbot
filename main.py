from motor_interface import setup_motors, read_joint_states, send_pwm_command
from ibvs_controller import compute_joint_command
from vision import get_image_error
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager

import signal
import time
import numpy as np

def main():
    # Initialize motors
    motor_group = setup_motors()
    control_freq = 30.0
    loop = FixedFrequencyLoopManager(control_freq)

    # Define base control gain
    Kp = 0.4  # Can be scaled based on proximity
    should_continue = True

    # Setup safe exit
    def signal_handler(*_):
        nonlocal should_continue
        should_continue = False
    signal.signal(signal.SIGINT, signal_handler)

    print("🧠 Starting IBVS Control Loop")
    while should_continue:
        # Step 1: Read joint positions & velocities
        q_rad, qdot_rad_per_s = read_joint_states(motor_group)

        # Step 2: Get visual error from both cameras (pixel space)
        image_error = get_image_error()

        # Step 3: Compute control action (e.g., torques or dq)
        u = compute_joint_command(
            image_error=image_error,
            q=q_rad,
            qdot=qdot_rad_per_s,
            gain_scale=1.0  # Placeholder
        )

        # Step 4: Send command to motors (PWM or velocity)
        send_pwm_command(motor_group, u)

        loop.sleep()

    print("🛑 Exiting loop")
    motor_group.disable_torque()

if __name__ == "__main__":
    main()
