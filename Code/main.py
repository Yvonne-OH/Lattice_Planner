from matplotlib import pyplot as plt
import numpy as np
import math

import Cubic_Spline
import util
import cartesian_frenet_conversion
import trajectory_generator
import Cost_collision

from prettytable import PrettyTable

# Define points for the spline
# 定义样条曲线的点
x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0, 7.5])*10
y = np.array([0.0, 0.0, 0.8, 1.2, 0.6, 0.0])*5
y1 = y + 1.725*np.ones(6)  # 上方道路的y坐标
y2 = y - 1.725*np.ones(6)  # 下方道路的y坐标

obstacles = [(45, 4.5, 1), (3.5, 3.6, 0.6)]  # 示例障碍物

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



"""
according to the matched point, 
compute the init state in Frenet frame 
根据匹配点计算出在frenet坐标系下的规划起点初始状态
"""

# Define initial conditions
start_point = (-38,-0.0)
vertical_step = 20
initial_condition = [2, 0, 0, 0] #v,a,theta,kappa

end_conditions = [
    [0, 2],    # [lateral displacement, final velocity]
    [0.5, 2],
    [1, 2],
    [-0.5, 2],
    [-1, 2],
    [1.5, 2],
    [-1.5, 2],
    [2, 2],
    [-2, 2]
]

x=start_point[0]
y=start_point[1]
v=initial_condition[0]
a=initial_condition[1]
theta=initial_condition[2]
kappa=initial_condition[3]
#计算匹配点笛卡尔坐标
rx, ry, rtheta, rkappa, rdkappa, rs,distances=util.find_reference_point(middleLine, start_point, sample_num=200,En_simple=False,En_test=True)
#计算车辆在frenet的坐标
s, s_dot, s_double_dot, d, d_prime, d_double_prime=cartesian_frenet_conversion.CartesianFrenetConverter.cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa)
#print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % (rx, ry, rtheta, rkappa, rdkappa, rs))
#print(cartesian_frenet_conversion.CartesianFrenetConverter.cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa))
#print(cartesian_frenet_conversion.CartesianFrenetConverter.frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, [s, s_dot, s_double_dot], [d, d_prime, d_double_prime]))


# Compute and convert multiple trajectories
xy_trajectories = []
for end_condition in end_conditions:
    xy_trajectory = trajectory_generator.Trajectory_Cluster_Generation(middleLine, 0, start_point, vertical_step, initial_condition, end_condition)
    xy_trajectories.append(xy_trajectory)

# Plot the trajectories
cost= Cost_collision.Cost(xy_trajectories, obstacles)
util.plot_trajectories(xy_trajectories, middleLine,line1,line2,obstacles,cost)

# 连续规划循环
for _ in range(100):  # 假设进行10次规划迭代
    # 计算匹配点笛卡尔坐标
    rx, ry, rtheta, rkappa, rdkappa, rs , distance = util.find_reference_point(middleLine, start_point, sample_num=200, En_simple=False,En_test=False)
    # 计算车辆在frenet的坐标
    s, s_dot, s_double_dot, d, d_prime, d_double_prime = cartesian_frenet_conversion.CartesianFrenetConverter.cartesian_to_frenet(
        rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa)

    # 生成并转换多条轨迹
    xy_trajectories = []
    for end_condition in end_conditions:
        xy_trajectory = trajectory_generator.Trajectory_Cluster_Generation(
            middleLine, 0, start_point, vertical_step, initial_condition, end_condition)
        xy_trajectories.append(xy_trajectory)

    # 计算成本
    costs = Cost_collision.Cost(xy_trajectories, obstacles)

    # 找到成本最低的轨迹
    min_cost_index = np.argmin(costs)
    best_trajectory = xy_trajectories[min_cost_index]

    # 打印成本
    table = PrettyTable()
    table.field_names = ["Trajectory", "Cost"]
    for i, cost in enumerate(costs):
        table.add_row([i + 1, cost])
    print(table)

    # 绘制轨迹
    util.plot_trajectories(xy_trajectories, middleLine, line1, line2, obstacles, costs)

    # 更新起始点为最优轨迹的第十个坐标
    if len(best_trajectory) >= 51:
        start_point = best_trajectory[10]
    else:
        print("Trajectory too short for updating start point. Exiting loop.")
        break

    
    x = start_point[0]
    y = start_point[1]
    print(x,y)
