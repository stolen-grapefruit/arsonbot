from config import CONTROL_MODE, VISION_MODE
from motor_interface import setup_motors, read_joint_states, send_pwm_command
from vision import get_top_pixel_from_frame
from control import compute_control_action
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager
import cv2


def get_visual_info(mode="side"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": None}

    pixel_info, _ = get_top_pixel_from_frame(frame)
    if pixel_info:
        x_top, y_top, x_off, y_off = pixel_info

        # Define image-space error (e.g., offset from center or goal)
        # For now, assume goal is fixed pixel location (e.g., center x, target y)
        # Adjust as needed
        goal_pixel = [x_top, y_off]  # target location in image
        error = np.array([x_top - goal_pixel[0], y_top - goal_pixel[1]])

        return {"error": error}

    return {"error": None}



def main():
    motor_group = setup_motors()
    loop = FixedFrequencyLoopManager(30)  # 30 Hz
    should_continue = True

    while should_continue:
        q, qdot = read_joint_states(motor_group)

        visual_info = get_visual_info(mode=VISION_MODE)
        if visual_info["error"] is None:
            continue  # Skip this loop if no visual target
        
        if visual_info["error"] is None:
        print("⚠️ No visual error detected — skipping control.")
        continue

        u = compute_control_action(
            mode=CONTROL_MODE,
            error=visual_info["error"],
            q=q,
            qdot=qdot
        )
        print("q:", q)
        print("qdot:", qdot)
        print("u (control signal):", u)

        send_pwm_command(motor_group, u)
        loop.sleep()


        print("Visual error:", visual_info["error"])
        print("Computed command (u):", u)


if __name__ == "__main__":
    main()
