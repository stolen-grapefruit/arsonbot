import numpy as np
from controller_utils.gravity import compute_gravity_torque
from mechae263C_helpers.minilabs import DCMotorModel


def torque_to_pwm(tau, torque_constant=0.003, pwm_limit=885):
    pwm = tau / torque_constant
    return np.clip(pwm, -pwm_limit, pwm_limit)



class PDControllerNoGravity:
    def __init__(
        self,
        K_P=None,
        K_D=None,
        control_freq=30.0,
        pwm_limits=None,
        debug=False
    ):
        self.K_P = np.diag([2000, 2000, 2000, 2000]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([150, 250, 250, 250]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq
        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)
        self.debug = debug

        # Adding prev info
        self.q_previous = np.zeros(4)
        self.q_desired_previous = np.zeros(4)

    def update(self, q, qdot, q_desired, qdot_desired=None):

        if q is None or len(q) != 4:
            print(f"[Warning] Invalid q received: {q}, Using previous")
            q = self.q_previous

        if q_desired is None or len(q_desired) != 4:
            print(f"[Warning] Invalid q_des received: {q_desired}, Using previous")
            q_desired = self.q_desired_previous

        if qdot_desired is None:
            qdot_desired = np.zeros_like(q)

        if qdot is None or len(qdot) != 4:
            print(f"[Warning] Invalid qdot received: {qdot}, substituting zeros.")
            qdot = np.zeros(4)

        self.q_previous = q
        self.q_desired_previous = q_desired

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

        # PD term in PWM space (if gains tuned accordingly)
        error = q_desired - q
        derror = qdot_desired - qdot
        pd_pwm = self.K_P @ error + self.K_D @ derror

        # Gravity torque → PWM scaling
        if self.use_gravity:
            gravity_torque = compute_gravity_torque(q)
            gravity_pwm = torque_to_pwm(gravity_torque)
        else:
            gravity_pwm = np.zeros_like(q)

        pwm_out = pd_pwm + gravity_pwm

        if self.debug:
            print("---- PDGC DEBUG ----")
            print("q [deg]:", np.round(np.degrees(q), 2))
            print("q_des [deg]:", np.round(np.degrees(q_desired), 2))
            print("PWM PD term:", np.round(pd_pwm, 1))
            print("PWM Gravity comp:", np.round(gravity_pwm, 1))
            print("PWM total:", np.round(pwm_out, 1))
            if np.all(np.abs(pwm_out) < 5):
                print("⚠️ PWM values are very low — may not overcome stiction.")

        return pwm_out

    def torque_to_pwm(self, tau):
        return torque_to_pwm(tau)  # optional access to reuse externally







