import numpy as np
from math import sqrt, inf, atan2
import matplotlib.pyplot as plt

def Collision_detection(xy_trajectories, obstacles, expansion_factor=2):
    expanded_obstacles = [(x, y, r * expansion_factor) for (x, y, r) in obstacles]

    for xy_trajectory in xy_trajectories:
        for x_traj, y_traj in xy_trajectory:
            for x_obs, y_obs, r_obs in expanded_obstacles:
                distance = sqrt((x_traj - x_obs) ** 2 + (y_traj - y_obs) ** 2)
                if distance <= r_obs:
                    return True
    return False

def calculate_cost(xy_trajectory, time_step, obstacles, expansion_factor):
    xy_trajectory = np.array(xy_trajectory)
    if len(xy_trajectory) < 2:
        return inf

    # 计算速度
    velocities = np.diff(xy_trajectory, axis=0) / time_step
    speeds = np.linalg.norm(velocities, axis=1)

    # 计算加速度
    accelerations = np.diff(velocities, axis=0) / time_step
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)

    # 计算jerk
    jerks = np.diff(accelerations, axis=0) / time_step
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)

    # 计算角速度
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    angular_velocities = np.diff(angles) / time_step

    # 碰撞检测
    if Collision_detection([xy_trajectory], obstacles, expansion_factor):
        return inf

    # 计算成本
    cost = np.mean(acc_magnitudes) + np.mean(jerk_magnitudes) + np.mean(np.abs(angular_velocities))
    return cost

def Cost(xy_trajectories, obstacles, time_step=0.1, expansion_factor=2):
    costs = []
    for xy_trajectory in xy_trajectories:
        cost = calculate_cost(xy_trajectory, time_step, obstacles, expansion_factor)
        costs.append(cost)
    return costs

def plot_trajectories_and_obstacles(xy_trajectories, obstacles, expansion_factor=2):
    fig, ax = plt.subplots()
    
    # 绘制轨迹点
    for xy_trajectory in xy_trajectories:
        x_traj, y_traj = zip(*xy_trajectory)
        ax.plot(x_traj, y_traj, marker='o', linestyle='-', label='Trajectory')

    # 绘制障碍物和扩展障碍物
    for (x, y, r) in obstacles:
        circle = plt.Circle((x, y), r, color='r', fill=False, linestyle='dashed', label='Obstacle')
        expanded_circle = plt.Circle((x, y), r * expansion_factor, color='b', fill=False, linestyle='dotted', label='Expanded Obstacle')
        ax.add_patch(circle)
        ax.add_patch(expanded_circle)

    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories and Obstacles')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 轨迹点示例
    xy_trajectories = [
        [(0, 0), (1, 1), (2, 2)],
        [(3, 3), (4, 4), (5, 5)]
    ]

    # 障碍物示例（圆心和半径）
    obstacles = [(1.5, 1.5, 0.5), (3.5, 3.5, 1.0)]

    # 计算成本
    costs = Cost(xy_trajectories, obstacles, time_step=0.1, expansion_factor=2)
    for i, cost in enumerate(costs):
        print(f"Cost of trajectory {i+1}: {cost}")

    # 绘制图形
    plot_trajectories_and_obstacles(xy_trajectories, obstacles, expansion_factor=2)
