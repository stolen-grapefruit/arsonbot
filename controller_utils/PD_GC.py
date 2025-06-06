import numpy as np
from controller_utils.gravity import compute_gravity_torque


class PDControllerNoGravity:
    def __init__(
        self,
        K_P=None,
        K_D=None,
        control_freq=30.0,
        pwm_limits=None,
        debug=False
    ):
        self.K_P = np.diag([1000, 400, 300, 300]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([100, 400, 10, 10]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq
        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)
        self.debug = debug

    def update(self, q, qdot, q_desired, qdot_desired=None):
        if qdot_desired is None:
            qdot_desired = np.zeros_like(q)

        error = q_desired - q
        derror = qdot_desired - qdot
        pwm_out = self.K_P @ error + self.K_D @ derror

        if self.debug:
            print("---- PD DEBUG ----")
            print("q:", np.round(np.degrees(q), 1))
            print("q_desired:", np.round(np.degrees(q_desired), 1))
            print("pwm out:", np.round(pwm_out, 1))
            if np.all(np.abs(pwm_out) < 5):
                print("⚠️ PWM too small — won't move motors.")

        return pwm_out

    def torque_to_pwm(self, tau):
        return tau  # identity mapping — not needed if update() returns PWM directly


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
        self.K_P = np.diag([25, 20, 20, 15]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([0.2, 0.15, 0.15, 0.1]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq

        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)

        self.use_gravity = use_gravity
        self.debug = debug

    def update(self, q, qdot, q_desired, qdot_desired=None):
        if qdot_desired is None:
            qdot_desired = np.zeros_like(q)

        error = q_desired - q
        derror = qdot_desired - qdot

        pd_term = self.K_P @ error + self.K_D @ derror
        gravity_term = compute_gravity_torque(q) if self.use_gravity else np.zeros_like(q)

        pwm_out = pd_term + gravity_term  # send this directly as PWM

        if self.debug:
            print("---- PDGC DEBUG ----")
            print("q [deg]:", np.round(np.degrees(q), 2))
            print("q_des [deg]:", np.round(np.degrees(q_desired), 2))
            print("pwm out:", np.round(pwm_out, 1))
            if np.all(np.abs(pwm_out) < 5):
                print("⚠️ WARNING: PWM values are very low. Check gain scale.")

        return pwm_out

    def torque_to_pwm(self, tau):
        # Bypassed entirely: tau is treated as PWM
        return tau

