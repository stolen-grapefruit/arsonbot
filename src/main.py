from config import CONTROL_MODE, VISION_MODE
from motor_interface import setup_motors, read_joint_states, send_pwm_command
from vision import get_visual_info
from control import compute_control_action
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager

def main():
    motor_group = setup_motors()
    loop = FixedFrequencyLoopManager(30)
    should_continue = True

    while should_continue:
        q, qdot = read_joint_states(motor_group)

        visual_info = get_visual_info(mode=VISION_MODE)
        u = compute_control_action(
            mode=CONTROL_MODE,
            error=visual_info["error"],
            q=q,
            qdot=qdot
        )

        send_pwm_command(motor_group, u)
        loop.sleep()
