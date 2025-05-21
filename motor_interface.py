from dxl import DynamixelIO, DynamixelModel, DynamixelMotorFactory
import numpy as np

def setup_motors(control_mode="PWM"):
    dxl_io = DynamixelIO(device_name="COM3", baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io, DynamixelModel.MX28)
    motor_group = motor_factory.create(1, 2, 3, 4)
    motor_group.enable_torque()
    motor_group.set_mode(control_mode)
    return motor_group


def read_joint_states(motor_group):
    q = np.array(list(motor_group.angle_rad.values()))
    qdot = np.array(list(motor_group.velocity_rad_per_s.values()))
    return q, qdot

def send_pwm_command(motor_group, u_pwm, limit=885):  # MX28 default
    u_clipped = np.clip(u_pwm, -limit, limit)
    pwm_dict = {
        dxl_id: int(pwm)
        for dxl_id, pwm in zip(motor_group.dynamixel_ids, u_clipped)
    }
    motor_group.pwm = pwm_dict
