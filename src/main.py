import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from dxl import (
    DynamixelMode, 
    DynamixelModel, 
    DynamixelMotorGroup, 
    DynamixelMotorFactory, 
    DynamixelIO
)
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel


class PD_Controller:
    def __init__(
        self,
        motor_group: DynamixelMotorGroup,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        q_desired_deg: Sequence[float],
        max_duration_s: float = 2.0,
    ):
        # Motor Communication
        self.motor_group = motor_group

        # Controller Setup
        self.q_desired_rad = np.deg2rad(q_desired_deg)
        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)

        # Control Loop Setup
        self.control_freq_Hz = 30.0
        self.max_duration_s = float(max_duration_s)
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True

        # Read initial joint positions
        current_rad = np.asarray(list(self.motor_group.angle_rad.values()))
        self.q_initial_rad = current_rad.copy()
        print("Initial joint positions (deg):", np.degrees(self.q_initial_rad))

        # Logging
        self.joint_position_history = deque()
        self.time_stamps = deque()

        # DC Motor Model
        self.pwm_limits = np.asarray(
            [info.pwm_limit for info in self.motor_group.motor_info.values()]
        )
        self.motor_model = DCMotorModel(
            self.control_period_s, pwm_limits=self.pwm_limits
        )

        # Exit Handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Set to PWM mode
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()

    def start_control_loop(self):
        start_time = time.time()

        while self.should_continue:
            # Read joint states
            q_rad = np.asarray(list(self.motor_group.angle_rad.values()))
            qdot_rad_per_s = q_rad.copy()  # Simplified for now (no velocity sensor)

            # Logging
            self.joint_position_history.append(q_rad)
            self.time_stamps.append(time.time() - start_time)

            # Stop after duration
            if self.time_stamps[-1] > self.max_duration_s:
                self.stop()
                return

            # PD Control
            q_error = self.q_desired_rad - q_rad
            gravity_comp_torques = self.calc_gravity_compensation_torque(q_rad)
            u = self.K_P @ q_error - self.K_D @ qdot_rad_per_s + gravity_comp_torques
            pwm_command = self.motor_model.calc_pwm_command(u)

            # Apply PWM
            self.motor_group.pwm = {
                dxl_id: pwm
                for dxl_id, pwm in zip(self.motor_group.dynamixel_ids, pwm_command, strict=True)
            }

            # Debug print
            # print("q [deg]:", np.degrees(q_rad))

            # Maintain control rate
            self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    def calc_gravity_compensation_torque(self, joint_positions_rad: NDArray[np.double]):
        return np.zeros(len(joint_positions_rad))


if __name__ == "__main__":
    # Desired target joint positions
    q_desired = [180, 180, 180, 180]

    # PD Gains
    K_P = np.diag([2.0, 2.0, 1.5, 1.5])
    K_D = np.diag([0.001, 0.001, 0.001, 0.001])

    # Set up Dynamixel interface
    dxl_io = DynamixelIO(device_name="COM4", baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io=dxl_io, dynamixel_model=DynamixelModel.MX28)
    dynamixel_ids = [1, 2, 3, 4]
    motor_group = motor_factory.create(*dynamixel_ids)

    # Create controller
    controller = PD_Controller(
        motor_group=motor_group,
        K_P=K_P,
        K_D=K_D,
        q_desired_deg=q_desired
    )

    # Run PD control loop
    controller.start_control_loop()

    # Plot results
    time_stamps = np.asarray(controller.time_stamps)
    joint_positions = np.rad2deg(controller.joint_position_history).T
    date_str = datetime.now().strftime("%d-%m_%H-%M-%S")
    fig_file_name = f"joint_positions_vs_time_{date_str}.pdf"

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs = axs.flatten()

    for i in range(4):
        axs[i].plot(time_stamps, joint_positions[i], label="Motor Angle Trajectory", color="black")
        axs[i].axhline(controller.q_desired_rad[i] * 180 / math.pi, ls="--", color="red", label="Setpoint")
        axs[i].axhline((controller.q_desired_rad[i] * 180 / math.pi) + 1, ls=":", color="blue", label="Convergence Bound")
        axs[i].axhline((controller.q_desired_rad[i] * 180 / math.pi) - 1, ls=":", color="blue")
        axs[i].set_title(f"Motor Joint {i}")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Motor Angle [deg]")
        axs[i].legend()

    fig.suptitle("Motor Angles vs Time")
    fig.tight_layout()
    fig.savefig(fig_file_name)
    plt.show()
