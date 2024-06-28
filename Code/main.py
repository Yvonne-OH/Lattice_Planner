from matplotlib import pyplot as plt
import numpy as np
import math

import Cubic_Spline
import util
import cartesian_frenet_conversion
import trajectory_generator

from prettytable import PrettyTable

# Define points for the spline
# 定义样条曲线的点
x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0, 7.5])*10
y = np.array([0.0, 0.0, 0.8, 1.2, 0.6, 0.0])*5
y1 = y + 1.725*np.ones(6)  # 上方道路的y坐标
y2 = y - 1.725*np.ones(6)  # 下方道路的y坐标

# 创建样条实例
middleLine = Cubic_Spline.Spline2D(x, y)  # 中心线
line1 = Cubic_Spline.Spline2D(x, y1)      # 上方道路
line2 = Cubic_Spline.Spline2D(x, y2)      # 下方道路

# 计算样条曲线上的点
s_values = np.linspace(0, max(middleLine.s), 300)          # 生成用于插值的s值
x1, y1 = zip(*[line1.calc_position(s) for s in s_values])  # 上方道路的点
x2, y2 = zip(*[line2.calc_position(s) for s in s_values])  # 下方道路的点
xm, ym = zip(*[middleLine.calc_position(s) for s in s_values])  # 中心线的点

# 设置绘图
plt.figure(figsize=(10, 5))
plt.plot(x1, y1, label='Upper Road', color='black')  # 绘制上方道路
plt.plot(x2, y2, label='Lower Road', color='black')  # 绘制下方道路
plt.plot(xm, ym, label='Middle Road', color='red', linestyle='--')  # 绘制中心线

# Enhancing the plot
plt.title('Road Visualization with Cubic Splines')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensures that one unit in the x-axis is the same length as one unit in the y-axis



start_point=(-2.8,3.5)
x=start_point[0]
y=start_point[1]
v=1
a=0
theta=0
kappa=0
vertical_step=20

"""
according to the matched point, 
compute the init state in Frenet frame 
根据匹配点计算出在frenet坐标系下的规划起点初始状态
"""

#计算匹配点笛卡尔坐标
rx, ry, rtheta, rkappa, rdkappa, rs=util.find_reference_point(middleLine, start_point, sample_num=200,En_test=True)

print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % (rx, ry, rtheta, rkappa, rdkappa, rs))


print("--------")
print(cartesian_frenet_conversion.CartesianFrenetConverter.cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa))

#计算车辆在frenet的坐标
s, s_dot, s_double_dot, d, d_prime, d_double_prime=cartesian_frenet_conversion.CartesianFrenetConverter.cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa)



print("--------")
print(cartesian_frenet_conversion.CartesianFrenetConverter.frenet_to_cartesian(
    rs, rx, ry, rtheta, rkappa, rdkappa, [s, s_dot, s_double_dot], [d, d_prime, d_double_prime]))


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
    Trajectory_planner = trajectory_generator.TrajectorySolver()

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

def plot_trajectories(xy_trajectories, middleLine, line1, line2, Adaptive_zoom=False):
    """
    Plot multiple trajectories in Cartesian coordinates.

    Parameters:
    - xy_trajectories: List of lists of (x, y) coordinates
    - middleLine: Spline curve (center line)
    - line1: Upper boundary spline curve
    - line2: Lower boundary spline curve
    - xlim: Tuple (xmin, xmax) to limit x-axis for zoomed in view (optional)
    - ylim: Tuple (ymin, ymax) to limit y-axis for zoomed in view (optional)
    """
    plt.figure(figsize=(10, 5))
    s_values = np.linspace(0, max(middleLine.s), 300)
    x1, y1 = zip(*[middleLine.calc_position(s) for s in s_values])
    x_up, y_up = zip(*[line1.calc_position(s) for s in s_values])
    x_down, y_down = zip(*[line2.calc_position(s) for s in s_values])
    plt.plot(x1, y1, label='Middle Road', color='red', linestyle='--')
    plt.plot(x_up, y_up, label='Upper Road', color='black')  # Draw upper road boundary
    plt.plot(x_down, y_down, label='Lower Road', color='black')  # Draw lower road boundary

    for i, xy_trajectory in enumerate(xy_trajectories):
        x_traj, y_traj = zip(*xy_trajectory)
        plt.plot(x_traj, y_traj, label=f'Planned Trajectory {i+1}')


    plt.title('Trajectory Conversion from Frenet to Cartesian Coordinates')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    #plt.axis('equal')

    # Apply x and y limits if provided
    if Adaptive_zoom:
        xlim = [min(x_traj)-0.5*(max(x_traj)-min(x_traj)),max(x_traj)+0.5*(max(x_traj)-min(x_traj))]
        ylim = [min(y_traj)-0.5*(max(y_traj)-min(y_traj)),max(y_traj)+0.5*(max(y_traj)-min(y_traj))]
        plt.xlim(xlim)
        plt.ylim(ylim)

    plt.show()
    
# Define initial conditions
start_point = (-2.8, 3.5)
vertical_step = 20
initial_condition = [1, 0, 0, 0]

end_conditions = [
    [0, 2],    # [lateral displacement, final velocity]
    [0.5, 2],
    [1, 2],
    [-0.5, 2],
    [-1, 2]
]

# Compute and convert multiple trajectories
xy_trajectories = []
for end_condition in end_conditions:
    xy_trajectory = Trajectory_Cluster_Generation(middleLine, 0, start_point, vertical_step, initial_condition, end_condition)
    xy_trajectories.append(xy_trajectory)

# Plot the trajectories
plot_trajectories(xy_trajectories, middleLine,line1,line2)


