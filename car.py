import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class GroundRobotController:
    def __init__(self, length, gains):
        self.L = length  # Length of the car
        self.k1, self.k2, self.k3 = gains  # Controller gains

    def compute_control_inputs(self, xr, yr, theta, xd, yd, vd, wd):
        # Compute errors
        xe = np.cos(theta) * (xd - xr) + np.sin(theta) * (yd - yr)
        ye = -np.sin(theta) * (xd - xr) + np.cos(theta) * (yd - yr)
        theta_d = np.arctan2(yd - yr, xd - xr)
        theta_e = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))  # Ensures angle is within [-π, π]

        # Control law
        v = vd * np.cos(theta_e) + self.k1 * xe
        omega = wd + vd * (self.k2 * ye + self.k3 * np.sin(theta_e))

        # Compute steering angle (delta) while avoiding division by zero
        v_safe = v if abs(v) > 1e-5 else 1e-5
        delta = np.arctan((self.L * omega) / v_safe)

        # Limit inputs
        return np.clip(delta, -np.pi / 4, np.pi / 4), np.clip(v, -10, 10)

# Desired trajectory function
def desired_path(t):
    return 4 * t, 2 * t

# Car simulation function
def simulate_robot():
    controller = GroundRobotController(length=5, gains=(0.5, 0.5, 0.1))
    dt = 0.02
    t = np.arange(0, 20, dt)
    xd, yd = desired_path(t)

    # Compute desired velocities
    dxdt, dydt = np.gradient(xd, t), np.gradient(yd, t)
    vd = np.sqrt(dxdt**2 + dydt**2)
    wd = np.gradient(np.arctan2(dydt, dxdt), t)

    # Initial state (rear wheel position and heading)
    xr, yr, theta = -1, -2, 0

    # Arrays to store trajectories
    state_log = []

    # Trajectory update using control inputs
    for i in range(1, len(t)):
        delta, v = controller.compute_control_inputs(xr, yr, theta, xd[i], yd[i], vd[i], wd[i])

        # Kinematic update
        xr += v * np.cos(theta) * dt
        yr += v * np.sin(theta) * dt
        theta += (v / controller.L) * np.tan(delta) * dt

        # Front wheel position
        xf = xr + controller.L * np.cos(theta)
        yf = yr + controller.L * np.sin(theta)

        # Store positions
        state_log.append([xf, yf, xr, yr])

    state_log = np.array(state_log)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.axis('equal')

    def animate(i):
        ax.clear()
        ax.plot(xd, yd, 'r--', label='Desired Path')
        ax.plot(state_log[:i, 2], state_log[:i, 3], 'b-', label='Rear Wheel Path')
        ax.plot(state_log[:i, 0], state_log[:i, 1], 'g-', label='Front Wheel Path')

        # Car body
        ax.plot([state_log[i, 2], state_log[i, 0]], [state_log[i, 3], state_log[i, 1]], 'k-', label='Body')

        # Axles
        ax.plot([state_log[i, 2] - 0.5, state_log[i, 2] + 0.5], [state_log[i, 3], state_log[i, 3]], 'b-', linewidth=3, label='Rear Axle')
        ax.plot([state_log[i, 0] - 0.5, state_log[i, 0] + 0.5], [state_log[i, 1], state_log[i, 1]], 'g-', linewidth=3, label='Front Axle')

        ax.set_title(f'Time: {i*dt:.2f}s')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True)
        ax.legend()
        ax.axis('equal')

    anim = animation.FuncAnimation(fig, animate, frames=len(state_log), interval=dt * 1000)
    plt.show()

if __name__ == '__main__':
    simulate_robot()
