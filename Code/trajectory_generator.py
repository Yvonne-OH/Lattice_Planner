import numpy as np
import matplotlib.pyplot as plt
import util
import cartesian_frenet_conversion

class TrajectorySolver:
    def solve_lateral_trajectory_frenet(self, delta, initial_state, end_state, step=0.1):
        """
        Solves for the lateral trajectory using a quintic polynomial, which is useful for planning
        paths in autonomous driving where smooth transitions in lateral movement are crucial.

        Parameters:
            delta (float): The total time over which the trajectory is planned.
            initial_state (list): The initial lateral position, velocity, and acceleration.
            end_state (list): The desired end lateral position, velocity, and acceleration.
            step (float): The time step at which to sample the trajectory.

        Returns:
            tuple: Arrays of time points and corresponding lateral positions.
        """
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, delta, delta**2, delta**3, delta**4, delta**5],
            [0, 1, 2*delta, 3*delta**2, 4*delta**3, 5*delta**4],
            [0, 0, 2, 6*delta, 12*delta**2, 20*delta**3]
        ])
        b = np.array([initial_state[0], initial_state[1], initial_state[2], 
                      end_state[0], end_state[1], end_state[2]])
        coefficients = np.linalg.solve(A, b)
        time_points = np.arange(0, delta + step, step)
        lateral_positions = np.polyval(coefficients[::-1], time_points)
        return time_points, lateral_positions

    def solve_longitudinal_trajectory_frenet(self, delta, initial_state, end_state, step=0.1, derivative_order=0):
        """
        Calculates the longitudinal trajectory using a quartic polynomial, often needed for ensuring
        smooth velocity and acceleration profiles in path planning.

        Parameters:
            delta (float): The total time over which the trajectory is planned.
            initial_state (list): The initial position, velocity, and acceleration.
            end_state (list): The end velocity and acceleration (final position is not specified).
            step (float): The time step at which to sample the trajectory.
            derivative_order (int): The order of the derivative to return (0 for position).

        Returns:
            tuple: Arrays of time points and corresponding values of the specified derivative order of positions.
        """
        A = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 1, 2*delta, 3*delta**2, 4*delta**3],
            [0, 0, 2, 6*delta, 12*delta**2]
        ])
        b = np.array([initial_state[0], initial_state[1], initial_state[2], end_state[0], end_state[1]])
        coefficients = np.linalg.solve(A, b)
        time_points = np.arange(0, delta + step, step)
        derivative_coeffs = coefficients[::-1]
        for n in range(derivative_order):
            derivative_coeffs = np.polyder(derivative_coeffs)
        longitudinal_positions = np.polyval(derivative_coeffs, time_points)
        return time_points, longitudinal_positions

def Trajectory_Cluster_Generation(middleLine, linewidth, start_point, vertical_step, initial_condition, end_condition):
    """
    Compute and convert trajectory from Frenet to Cartesian coordinates.

    Parameters:
    - middleLine: Spline curve (center line)
    - linewidth: Width of the lane
    - start_point: Starting point (x, y)
    - vertical_step: Vertical step for trajectory planning
    - initial_condition: Initial conditions [v, a, theta, kappa]
    - end_condition: End conditions [d_offset, v_end]

    Returns:
    - xy_trajectory: List of converted (x, y) coordinates
    """
    # Find the reference point in the Frenet coordinate system based on the matched point
    rx, ry, rtheta, rkappa, rdkappa, rs = util.find_reference_point(middleLine, start_point, sample_num=200, En_test=False)

    # Extract initial and end conditions
    v, a, theta, kappa = initial_condition
    d_offset, v_end = end_condition

    # Compute the vehicle's initial coordinates in the Frenet frame
    x, y = start_point
    s, s_dot, s_double_dot, d, d_prime, d_double_prime = cartesian_frenet_conversion.CartesianFrenetConverter.cartesian_to_frenet(
        rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa)

    # Initialize the trajectory planner
    Trajectory_planner = TrajectorySolver()

    # Define initial lateral and longitudinal conditions in the Frenet frame
    l0 = [d, 0, 0]    # Initial lateral displacement in Frenet frame
    l1 = [d + d_offset, 0, 0]  # Final lateral displacement in Frenet frame

    s0 = [s, s_dot, s_double_dot] # Initial longitudinal displacement in Frenet frame
    s1 = [v_end, 0]               # Final longitudinal velocity (end point)

    # Solve for the lateral and longitudinal trajectory in the Frenet frame
    t, l_plan = Trajectory_planner.solve_lateral_trajectory_frenet(vertical_step, l0, l1)
    t, s_plan = Trajectory_planner.solve_longitudinal_trajectory_frenet(vertical_step, s0, s1, derivative_order=0)

    # Combine the time, lateral, and longitudinal plans into a single array
    Trajectory_plan_frenet = np.column_stack((t, s_plan, l_plan))

    xy_trajectory = []

    for row in Trajectory_plan_frenet:
        time_step = row[0]
        s_value = row[1]
        l_value = row[2]
        
        # Calculate the reference position and yaw angle at the given s value
        rs = s_value
        rx, ry = middleLine.calc_position(rs)
        rtheta = middleLine.calc_yaw(rs)
        
        # Convert the Frenet coordinates to Cartesian coordinates
        x, y, theta, kappa, v, a = cartesian_frenet_conversion.CartesianFrenetConverter.frenet_to_cartesian(
            rs, rx, ry, rtheta, rkappa, rdkappa, [s_value, s_dot, s_double_dot], [l_value, d_prime, d_double_prime])
        
        # Append the converted coordinates to the trajectory list
        xy_trajectory.append((x, y))

    return xy_trajectory


# Test and plot the results using the class
if __name__ == "__main__":
    planner = TrajectorySolver()
    
    # Lateral Trajectory Testing
    plt.figure(figsize=(10, 6))  
    vertical_steps = [10, 40, 60, 80]          # Time intervals for trajectory planning
    l0 = [0, 0, 0]                             # Initial state: zero lateral displacement, velocity, and acceleration
    l1_lat_dis_sample = [-1, -0.5, 0, 0.5, 1]  # Different end lateral displacements
    
    for vertical_step in vertical_steps:
        for l1_lat_dis in l1_lat_dis_sample:
            l1 = [l1_lat_dis, 0, 0]  # End state with varying lateral displacement
            t, y = planner.solve_lateral_trajectory_frenet(vertical_step, l0, l1)
            plt.plot(t, y, label=f'End pos {l1_lat_dis} at time {vertical_step}s')
    
    plt.title('Lateral Trajectories for Different End Positions and Times')
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Position (m)')
    # Placing the legend outside the plot on the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.show()
    
    # Longitudinal Trajectory Testing for Position, Velocity, and Acceleration
    s0 = [0, 10, 2]  # Initial state with nonzero velocity and acceleration
    v_u = 20         # Maximum end velocity sample
    s1_v_sample = [0, 0.25 * v_u, 0.5 * v_u, 0.75 * v_u, v_u]  # Velocity samples
    t_sample = [2, 4, 6, 8]  # Different time intervals
    
    # Setup figure and subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 18), sharex=True)
    
    # Position plot
    for delta in t_sample:
        for v_end in s1_v_sample:
            s1 = [v_end, 0]  # End state with varying velocities
            t, y = planner.solve_longitudinal_trajectory_frenet(delta, s0, s1, derivative_order=0)
            axes[0].plot(t, y, label=f'Vel {v_end} m/s at {delta}s')
    axes[0].set_title('Longitudinal Position')
    axes[0].set_ylabel('Position (m)')
    axes[0].grid(True)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Velocity plot
    for delta in t_sample:
        for v_end in s1_v_sample:
            s1 = [v_end, 0]
            t, y = planner.solve_longitudinal_trajectory_frenet(delta, s0, s1, derivative_order=1)
            axes[1].plot(t, y, label=f'Vel {v_end} m/s at {delta}s')
    axes[1].set_title('Longitudinal Velocity')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].grid(True)
    
    # Acceleration plot
    for delta in t_sample:
        for v_end in s1_v_sample:
            s1 = [v_end, 0]
            t, y = planner.solve_longitudinal_trajectory_frenet(delta, s0, s1, derivative_order=2)
            axes[2].plot(t, y, label=f'Vel {v_end} m/s at {delta}s')
    axes[2].set_title('Longitudinal Acceleration')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].grid(True)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()