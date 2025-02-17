import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class GroundRobotController:
    def __init__(self, length, gains):
        self.L = length # Length of the car
        self.k1, self.k2, self.k3 = gains   # Controller gains

    def compute_control_inputs(self, xr, yr, theta, xd, yd, vd, wd):
        
        # Error along x, y and theta
        xe = np.cos(theta) * (xd - xr) + np.sin(theta) * (yd - yr)
        ye = -np.sin(theta) * (xd - xr) + np.cos(theta) * (yd - yr)
        theta_e = np.arctan2(yd - yr, xd - xr) - theta

        # Velocity as Control inputs
        v = vd * np.cos(theta_e) + self.k1 * xe - self.k2 * ye * theta_e
        omega = wd + vd * (self.k2 * ye + self.k3 * np.sin(theta_e)) - self.k1 * xe * theta_e

        # Convert omega to steering angle delta
        delta = np.arctan((self.L * omega) / (v if abs(v) > 1e-5 else 1e-5))

        # Limit inputs
        v = np.clip(v, -10, 10)
        delta = np.clip(delta, -np.pi/4, np.pi/4)

        return np.clip(v, -10, 10), np.clip(delta, -np.pi/4, np.pi/4)


# Desired trajectory
def desired_path(t):
    return 4 * t, 2 * t

# Car Simulation
def simulate_robot():
    controller = GroundRobotController(length=5, gains=(0.5, 0.5, 0.1))
    dt = 0.02
    t = np.arange(0, 20, dt)
    xd, yd = desired_path(t)

    # Desired velocities
    dxdt, dydt = np.gradient(xd, t), np.gradient(yd, t)
    vd = np.sqrt(dxdt**2 + dydt**2)
    wd = np.gradient(np.arctan2(dydt, dxdt), t)

    xr, yr, theta = -1, -2, 0   # Initial state

    # Arrays to store trajectories
    state_log = [[xr, yr, theta]]

    # Trajectory update in each iteration based on control inputs
    for i in range(1, len(t)):
        v, delta = controller.compute_control_inputs(xr, yr, theta, xd[i], yd[i], vd[i], wd[i])

        # Update state
        xr += v * np.cos(theta) * dt
        yr += v * np.sin(theta) * dt
        theta += (v / 5) * np.tan(delta) * dt

        # Append trajectories
        state_log.append ([xr, yr, theta])

    state_log = np.array(state_log)

    fig, ax = plt.subplots()
    plt.axis('equal')

    def animate(i):
        ax.clear()
        ax.plot(xd, yd, 'r--', label='Desired Path')
        ax.plot(state_log[:i, 0], state_log[:i, 1], 'b-', label='Robot Path')
        ax.plot(state_log[i, 0], state_log[i, 1], 'ko')
        ax.set_title(f'Time: {i*dt:.2f}s')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True)
        ax.legend()
        ax.axis('equal')

    anim = animation.FuncAnimation(fig, animate, frames=len(state_log), interval=dt*1000)
    plt.show()

if __name__ == '__main__':
    simulate_robot()
