from matplotlib import pyplot as plt
import numpy as np
import math
import bisect
import Cubic_Spline



    

class Lattice_planner:
    """
    lattice路径规划器
    该类用于基于样条曲线生成的道路和障碍物信息来生成车辆的路径。
    """

    def __init__(self, line1, line2, state0=[0,0,0], obstacle=[], sample_num=3, ds=2.0, vw=0.1) -> None:
        """
        初始化路径规划器。
        参数:
        line1, line2: 样条曲线表示的两条道路边界。
        state0: 车辆的初始状态[x, y, yaw]。
        obstacle: 障碍物列表。
        sample_num: 每段中的采样点数。
        ds: 采样间距。
        vw: 车辆宽度。
        """
        self.line1 = line1
        self.line2 = line2
        self.sample_num = sample_num
        self.ds = ds
        self.vw = vw
        self.car = state0
        self.obstacle = obstacle
        self.roadindex = []

    def generate_path(self, target_states, k0):
        """
        生成路径。
        参数:
        target_states: 目标状态列表，每个状态格式为[x, y, yaw, s, km, kf]。
        k0: 初始曲率。
        """
        lookup_table = get_lookup_table()
        result = []

        for state in target_states:
            bestp = search_nearest_one_from_lookuptable(
                state[0], state[1], state[2], lookup_table)
            target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
            init_p = np.array(
                [math.sqrt(state[0] ** 2 + state[1] ** 2), bestp[4], bestp[5]]).reshape(3, 1)
            x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)
            if x is not None:
                print("find good path")
                result.append(
                    [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])
        print("finish path generation")
        return result

    def lane_state_sampling(self, distance):
        """
        采样车道状态。
        参数:
        distance: 总采样距离。
        """
        self.states = []
        distance = min(distance, max(self.line1.s))
        for i in range(1, int(distance/self.ds)+1):
            x1, y1 = self.line1.calc_position(i*self.ds)
            x2, y2 = self.line2.calc_position(i*self.ds)
            yaw1 = self.line1.calc_yaw(i*self.ds)
            yaw2 = self.line2.calc_yaw(i*self.ds)
            self.states.append(self.uniform_sampling((x1, y1), (x2, y2)))
        self.states = np.array(self.states)
    
    def uniform_sampling(self, pos1, pos2):
        """
        均匀采样函数。
        参数:
        pos1, pos2: 采样的两个端点位置。
        """
        states = []
        for i in range(self.sample_num):
            states.append((np.array(pos2)-np.array(pos1))*(i+1)/(self.sample_num+1) + np.array(pos1))
        return np.array(states)

    def plan(self, distance):
        """
        规划路径。
        参数:
        distance: 规划的距离。
        """
        self.lane_state_sampling(distance)
        pointNum = int(distance/self.ds)
        x1, y1 = self.line1.calc_position(pointNum*self.ds)
        x2, y2 = self.line2.calc_position(pointNum*self.ds)
        middle = (y1+y2)/2.0
        matrix = np.zeros(self.states.shape[:2])
        self.index_matrix = np.zeros(self.states.shape[:2])
        for i in range(matrix.shape[1]):
            matrix[-1, i] = abs(middle - self.states[-1, i, 1])
        for i in range(matrix.shape[0]-2, -1, -1):
            for j in range(matrix.shape[1]):
                tmp = []
                for index in range(matrix.shape[1]):
                    tmp.append(self.getCost(self.states[i, j], self.states[i+1, index])+matrix[i+1, index])
                matrix[i, j] = min(tmp)
                self.index_matrix[i, j] = tmp.index(min(tmp))
        print(matrix)
        print(self.index_matrix)
        self.roadindex.append(self.car[:2])
        selected = np.argmin(matrix[0])
        for i in range(matrix.shape[0]):
            self.roadindex.append(self.states[i,selected])
            selected = int(self.index_matrix[i,selected])
        self.roadindex = np.array(self.roadindex)
        self.road = Cubic_Spline.Spline2D(self.roadindex[:,0], self.roadindex[:,1])


    def getCost(self, state1, state2):
        """
        计算从状态1到状态2的成本。
        参数:
        state1, state2: 两个状态。
        """
        cost = np.linalg.norm(state1 - state2)
        offside = 0.5
        for i in self.obstacle:
            if (i[0] - state1[0] + offside) * (i[0] - state2[0] - offside) < 0:
                l1 = np.linalg.norm(state1 - state2)
                l2 = np.linalg.norm(i[:2] - state1)
                l3 = np.linalg.norm(i[:2] - state2)
                theta = math.acos((l2**2+l1**2-l3**2)/(2*l1*l2))
                distance = math.sin(theta)*l2
                if distance < self.vw*1.2:
                    cost += 9999
                else:
                    cost += 0.5/distance
        return cost



