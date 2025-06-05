# PD_GC.py

import numpy as np
from mechae263C_helpers.minilabs import DCMotorModel
from controller_utils.gravity import compute_gravity_torque


class PDGCController:
    def __init__(
        self,
        K_P=None,
        K_D=None,
        control_freq=30.0,
        pwm_limits=None,
        use_gravity=True,
        debug=False
    ):
        # === Gains ===
        self.K_P = np.diag([25, 20, 20, 15]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([0.2, 0.15, 0.15, 0.1]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq

        # === PWM Scaling Limits ===
        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)

        # === Flags ===
        self.use_gravity = use_gravity
        self.debug = debug

        # === Motor Model ===
        self.motor_model = DCMotorModel(self.dt, pwm_limits=self.pwm_limits)

    def update(self, q, qdot, q_desired, qdot_desired=None):
        if qdot_desired is None:
            qdot_desired = np.zeros_like(q)

        error = q_desired - q
        derror = qdot_desired - qdot

        pd_term = self.K_P @ error + self.K_D @ derror
        gravity_term = compute_gravity_torque(q) if self.use_gravity else np.zeros_like(q)

        tau = pd_term + gravity_term

        if self.debug:
            print("---- PDGC DEBUG ----")
            print("q [deg]:", np.round(np.degrees(q), 2))
            print("q_des [deg]:", np.round(np.degrees(q_desired), 2))
            print("tau [Nm]:", np.round(tau, 3))

        return tau

    def torque_to_pwm(self, tau):
        pwm = self.motor_model.calc_pwm_command(tau)

        if self.debug:
            for i, val in enumerate(tau):
                print(f"Joint {i+1}: tau = {val:.3f}, pwm = {pwm[i]:.1f}")
            if np.all(np.abs(pwm) < 5):
                print("⚠️ WARNING: PWM values are very low. Check gain scale or torque model.")

        return pwm


class PDControllerNoGravity:
    def __init__(
        self,
        K_P=None,
        K_D=None,
        control_freq=30.0,
        pwm_limits=None,
        debug=False
    ):
        self.K_P = np.diag([.1, 0.2, 0.2, 0.2]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([0.05, 0.05, 0.05, 0.05]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq

        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)
        self.debug = debug
        self.motor_model = DCMotorModel(self.dt, pwm_limits=self.pwm_limits)

    def update(self, q, qdot, q_desired, qdot_desired=None):
        if qdot_desired is None:
            qdot_desired = np.zeros_like(q)

        error = q_desired - q
        derror = qdot_desired - qdot

        tau = self.K_P @ error + self.K_D @ derror

        if self.debug:
            print("---- PD DEBUG ----")
            print("qdot:", np.round(qdot, 3))
            print("qdot_desired:", np.round(qdot_desired, 3))
            print("tau [Nm]:", np.round(tau, 3))

            pwm = self.motor_model.calc_pwm_command(tau)
            for i, val in enumerate(tau):
                print(f"Joint {i+1}: tau = {val:.3f}, pwm = {pwm[i]:.1f}")
            if np.all(np.abs(pwm) < 5):
                print("⚠️ WARNING: PWM values are very low. Check gain scale or torque model.")

        return tau

    def torque_to_pwm(self, tau):
        pwm = self.motor_model.calc_pwm_command(tau)
        if self.debug:
            print("PWM:", np.round(pwm, 1))
        return pwm
