# PD_GC.py

import numpy as np
from gravity import compute_gravity_torque
from mechae263C_helpers.minilabs import DCMotorModel


class PDGCController:
    def __init__(self, K_P=None, K_D=None, control_freq=30.0, pwm_limits=None):
        self.K_P = np.diag([2.5, 2.0, 2.0, 1.5]) if K_P is None else np.array(K_P)
        self.K_D = np.diag([0.05, 0.04, 0.04, 0.03]) if K_D is None else np.array(K_D)
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # Assume default PWM limits if not provided
        if pwm_limits is None:
            pwm_limits = np.ones(4) * 885  # Dynamixel MX-28 default range
        self.pwm_limits = np.array(pwm_limits)

        # Use the DCMotorModel provided in your framework
        self.motor_model = DCMotorModel(self.dt, pwm_limits=self.pwm_limits)

    def update(self, q, qdot, q_desired, qdot_desired=None):
        if qdot_desired is None:
            qdot_desired = np.zeros_like(q)

        error = q_desired - q
        derror = qdot_desired - qdot

        pd_term = self.K_P @ error + self.K_D @ derror
        gravity_term = compute_gravity_torque(q)

        tau = pd_term + gravity_term
        return tau

    def torque_to_pwm(self, tau):
        return self.motor_model.calc_pwm_command(tau)
