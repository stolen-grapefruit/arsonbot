import numpy as np
from controller_utils.gravity import compute_gravity_torque

################################# PD Control #################################

class PDControllerNoGravity:
    def __init__(
        self,
        K_P=None,
        K_D=None,
        control_freq=60.0,
        pwm_limits=None,
        debug=False
    ):
        self.K_P = np.diag([1000, 2400, 2200, 1800]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([150, 150, 100, 30]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq
        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)
        self.debug = debug

        self.q_previous = np.zeros(4)
        self.q_desired_previous = np.zeros(4)


    def update(self, q, qdot, q_desired, qdot_desired=None):

        # If status info not received substitute values
        if q is None or len(q) != 4:
            print(f"[Warning] Invalid q received: {q}, Using previous")
            q = self.q_previous
        if q_desired is None or len(q_desired) != 4:
            print(f"[Warning] Invalid q_desired received: {q_desired}, Using previous")
            q_desired = self.q_desired_previous
        if qdot_desired is None or len(qdot_desired) != 4:
            print(f"[Warning] Invalid qdot_desired received: {qdot_desired}, substituting zeros.")
            qdot_desired = np.zeros_like(q)
        if qdot is None or len(qdot) != 4:
            print(f"[Warning] Invalid qdot received: {qdot}, substituting zeros.")
            qdot = np.zeros(4)

        self.q_previous = q
        self.q_desired_previous = q_desired

        error = q_desired - q
        derror = qdot_desired - qdot
        pwm_out = self.K_P @ error + self.K_D @ derror
        print(f"PWM: ", pwm_out)

        if self.debug:
            print("---- PD DEBUG ----")
            print("q:", np.round(np.degrees(q), 1))
            print("q_desired:", np.round(np.degrees(q_desired), 1))
            print("pwm out:", np.round(pwm_out, 1))
            if np.all(np.abs(pwm_out) < 5):
                print("PWM too small — won't move motors.")

        return pwm_out

    def torque_to_pwm(self, tau):
        # Bypassed entirely: tau is treated as PWM
        return tau



################################# PD with Gravity Comp #################################

class PDGCController:
    def __init__(
        self,
        K_P=None,
        K_D=None,
        control_freq=60.0,
        pwm_limits=None,
        debug=False
    ):
        self.K_P = np.diag([1000, 2400, 2200, 1800]) if K_P is None else np.atleast_2d(K_P)
        self.K_D = np.diag([150, 150, 100, 30]) if K_D is None else np.atleast_2d(K_D)
        self.dt = 1.0 / control_freq
        self.pwm_limits = np.ones(4) * 885 if pwm_limits is None else np.array(pwm_limits)
        self.debug = debug

        # Adding prev info
        self.q_previous = np.zeros(4)
        self.q_desired_previous = np.zeros(4)


    def update(self, q, qdot, q_desired, qdot_desired=None):

        # If status info not received substitute values
        if q is None or len(q) != 4:
            print(f"[Warning] Invalid q received: {q}, Using previous")
            q = self.q_previous
        if q_desired is None or len(q_desired) != 4:
            print(f"[Warning] Invalid q_desired received: {q_desired}, Using previous")
            q_desired = self.q_desired_previous
        if qdot_desired is None or len(qdot_desired) != 4:
            print(f"[Warning] Invalid qdot_desired received: {qdot_desired}, substituting zeros.")
            qdot_desired = np.zeros_like(q)
        if qdot is None or len(qdot) != 4:
            print(f"[Warning] Invalid qdot received: {qdot}, substituting zeros.")
            qdot = np.zeros(4)

        self.q_previous = q
        self.q_desired_previous = q_desired

        error = q_desired - q
        derror = qdot_desired - qdot
        pwm_out = self.K_P @ error + self.K_D @ derror
        print(f"PWM: ", pwm_out)

        # ADDING GRAVITY
        gravity_term = compute_gravity_torque(q)
        gravity_pwm = self.convert_gravity(gravity_term)

        print(f"Gravity Comp: ", gravity_pwm)
        pwm_out = pwm_out + gravity_pwm

        if self.debug:
            print("---- PD DEBUG ----")
            print("q:", np.round(np.degrees(q), 1))
            print("q_desired:", np.round(np.degrees(q_desired), 1))
            print("pwm out:", np.round(pwm_out, 1))
            if np.all(np.abs(pwm_out) < 5):
                print("PWM too small — won't move motors.")

        return pwm_out
    
    def convert_gravity(self, torque):
        # Converting gravity term from torque to pwm!!!
        MAX_TORQUE = 2.5  # Nm for MX-28AR at 12V
        MAX_PWM = 885     # max PWM value

        pwm = (torque / MAX_TORQUE) * MAX_PWM
        pwm = np.clip(pwm, -MAX_PWM, MAX_PWM)  # safety clipping
        return pwm.astype(int)

    def torque_to_pwm(self, tau):
        # Bypassed entirely: tau is treated as PWM
        return tau
