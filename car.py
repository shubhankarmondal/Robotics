import numpy as np
import matplotlib.pyplot as plt

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

        return v, delta


# Desired trajectory
def desired_path(t):
    return 4 * t, 2 * t

# Car Simulation
def simulate_robot():
    controller = GroundRobotController(length=5, gains=(0.5, 0.5, 0.1))

    t = np.linspace(0, 20, 1000)
    xd, yd = desired_path(t)

    # Desired velocities
    dxdt = np.gradient(xd, t)
    dydt = np.gradient(yd, t)
    vd = np.sqrt(dxdt**2 + dydt**2)
    wd = np.gradient(np.arctan2(dydt, dxdt), t)

    xr, yr, theta = -1, -2, 0   # Initial state

    # Arrays to store trajectories
    xr_traj, yr_traj = [xr], [yr]
    xf_traj, yf_traj = [xr + 5 * np.cos(theta)], [yr + 5 * np.sin(theta)]

    # Trajectory update in each iteration based on control inputs
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        v, delta = controller.compute_control_inputs(xr, yr, theta, xd[i], yd[i], vd[i], wd[i])

        # Update state
        xr += v * np.cos(theta) * dt
        yr += v * np.sin(theta) * dt
        theta += (v / 5) * np.tan(delta) * dt

        # Append trajectories
        xr_traj.append(xr)
        yr_traj.append(yr)
        xf_traj.append(xr + 5 * np.cos(theta))
        yf_traj.append(yr + 5 * np.sin(theta))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(xd, yd, 'r--', label='Desired Path')
    plt.plot(xr_traj, yr_traj, 'b-', label='Rear Wheel')
    plt.plot(xf_traj, yf_traj, 'g-', label='Front Wheel')
    plt.scatter([-1], [-2], c='k', s=100, label='Start')
    plt.xlabel(r'Position $x$ (m)', fontsize=12)
    plt.ylabel(r'Position $y$ (m)', fontsize=12)
    plt.title('Car Trajectory using Controller', fontsize=14)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("car_closeloop_trajectory.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    simulate_robot()