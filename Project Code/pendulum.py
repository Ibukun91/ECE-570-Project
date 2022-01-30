import numpy as np

class Pendulum:
    def __init__(self):
        self.max_speed = 8.
        self.dt = .05

    def theta_dot(self, theta_dot, sin_theta, torque):
        g = 10.
        m = 1.
        l = 1.
        return theta_dot + (3*g / (2*l) * sin_theta + 3/(m * l**2) * torque) * self.dt

    def theta(self, theta, new_theta_dot):
        return theta + new_theta_dot * self.dt

    def clip_theta_dot(self, new_theta_dot):
        return np.clip(new_theta_dot, -self.max_speed, self.max_speed)
