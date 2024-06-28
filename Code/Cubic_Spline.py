# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:29:20 2024

@author: 39829
"""
import numpy as np
import bisect
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math


class Spline:
    """
    三次样条插值类。
    此类用于根据一组点（x, y）生成一个三次样条曲线，并允许计算曲线上任意点的位置及其一阶和二阶导数。
    """

    def __init__(self, x, y):
        """
        初始化函数。
        参数:
        - x: 点的x坐标数组。
        - y: 点的y坐标数组。
        作用:
        - 初始化存储点坐标、计算并存储样条曲线的系数。
        """
        self.b, self.c, self.d, self.w = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # 点的数量
        h = np.diff(x)    # 相邻点之间的x距离

        self.a = [iy for iy in y]  # 系数a直接从输入y值获取

        A = self.__calc_A(h)  # 计算系数矩阵A
        B = self.__calc_B(h)  # 计算向量B
        self.c = np.linalg.solve(A, B)  # 解线性方程组得到系数c

        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        计算给定t（x值）处的y值。
        参数:
        - t: x值。
        返回:
        - y值，如果t在定义的x范围之外，则返回None。
        """
        if t < self.x[0] or t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result

    def calcd(self, t):
        """
        计算在点t处的一阶导数。
        参数:
        - t: x值。
        返回:
        - 一阶导数值，如果t在定义的x范围之外，则返回None。
        """
        if t < self.x[0] or t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        计算在点t处的二阶导数。
        参数:
        - t: x值。
        返回:
        - 二阶导数值，如果t在定义的x范围之外，则返回None。
        """
        if t < self.x[0] or t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result
    
    def calctdd(self, t):
        """
        计算在点t处的三阶导数。
        参数:
        - t: x值。
        返回:
        - 三阶导数值，如果t在定义的x范围之外，则返回None。
        """
        if t < self.x[0] or t > self.x[-1]:
            return None
        i = self.__search_index(t)
        return 6.0 * self.d[i]  # 三次样条的三次导数是常数


    def __search_index(self, x):
        """
        查找数据段的索引。
        保证返回的索引在有效范围内，尤其是处理接近x数组最后一个值时。
        """
        if x >= self.x[-1]:  # 处理x等于或大于x数组最后一个元素的情况
            return len(self.x) - 2  # 返回最后一个区间的索引
        return bisect.bisect_right(self.x, x) - 1


    def __calc_A(self, h):
        """
        计算样条系数c的系数矩阵A。
        参数:
        - h: 相邻点之间的x差值数组。
        返回:
        - 系数矩阵A。
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        """
        计算向量B，用于求解样条系数c。
        参数:
        - h: 相邻点之间的x差值数组。
        返回:
        - 向量B。
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B

class Spline2D:
    """
    二维三次样条曲线类

    此类基于给定的二维点集(x, y)生成二维的三次样条曲线。
    提供位置、曲率和航向角的计算功能。
    """

    def __init__(self, x, y):
        """
        初始化函数。
        参数:
        - x: 水平坐标点数组。
        - y: 垂直坐标点数组。
        作用:
        - 计算每个点对应的累积距离s。
        - 根据s和x、y创建两个一维样条曲线对象。
        """
        self.s = self.__calc_s(x, y)  # 计算累积距离
        self.sx = Spline(self.s, x)  # 对x坐标进行样条插值
        self.sy = Spline(self.s, y)  # 对y坐标进行样条插值

    def __calc_s(self, x, y):
        """
        计算累积距离。
        参数:
        - x: 水平坐标点数组。
        - y: 垂直坐标点数组。
        返回:
        - s: 从起始点到每个点的累积距离数组。
        """
        dx = np.diff(x)  # 相邻点x坐标之差
        dy = np.diff(y)  # 相邻点y坐标之差
        self.ds = np.hypot(dx, dy)  # 计算每段距离
        s = [0]
        s.extend(np.cumsum(self.ds))  # 累计距离
        return s

    def calc_position(self, s):
        """
        计算给定参数s的位置。
        参数:
        - s: 参数值。
        返回:
        - (x, y): 对应s值的坐标位置。
        """
        x = self.sx.calc(s)  # 计算x坐标
        y = self.sy.calc(s)  # 计算y坐标
        return x, y

    def calc_curvature(self, s):
        """
        计算曲率。
        参数:
        - s: 参数值。
        返回:
        - k: 在s处的曲率值。
        """
        dx = self.sx.calcd(s)  # x的一阶导数
        ddx = self.sx.calcdd(s)  # x的二阶导数
        dy = self.sy.calcd(s)  # y的一阶导数
        ddy = self.sy.calcdd(s)  # y的二阶导数
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))  # 曲率公式
        return k
    
    def calc_curvature_derivative(self, s):
        """
        计算曲率的一阶导数。
        参数:
        - s: 参数值。
        返回:
        - dk: 在s处的曲率一阶导数。
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        ddx = self.sx.calcdd(s)
        ddy = self.sy.calcdd(s)
        dddx = self.sx.calctdd(s)  # x的三阶导数
        dddy = self.sy.calctdd(s)  # y的三阶导数
        
        # 曲率一阶导数的计算
        dx2_dy2 = dx**2 + dy**2
        numerator = (dddy * dx - dddx * dy) * (dx2_dy2) - (ddy * dx - ddx * dy) * 3 * (dx * ddx + dy * ddy)
        denominator = (dx2_dy2)**(5 / 2)
        
        dk = numerator / denominator
        return dk



    def calc_yaw(self, s):
        """
        计算航向角。
        参数:
        - s: 参数值。
        返回:
        - yaw: 在s处的航向角。
        """
        dx = self.sx.calcd(s)  # x的一阶导数
        dy = self.sy.calcd(s)  # y的一阶导数
        yaw = math.atan2(dy, dx)  # 使用atan2计算航向角
        return yaw
    


def test_spline_interpolation():
    # 测试数据点
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10])
    y = np.array([0.0, 0.0, 0.8, 1.2, 0.6, 0.0, 1])

    # 创建 Spline 对象
    spline = Spline(x, y)

    # 计算插值结果
    s_values = np.linspace(min(x), max(x), 100)
    y_interp = [spline.calc(xi) for xi in s_values]

    # 绘制原始数据点
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, color='red', zorder=5)

    # 绘制插值曲线
    plt.plot(s_values, y_interp, label='Spline Interpolation', color='blue')

    plt.title('Cubic Spline Interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()



# 测试函数
def test_spline2d():
    # 创建测试数据点
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 0.3, 0.5, 0.2, 0.0, 0.2])
    
    # 初始化 Spline2D 对象
    spline2d = Spline2D(x, y)
    
    # 计算曲线上的点
    s_vals = np.linspace(0, max(spline2d.s), 100)
    positions = [spline2d.calc_position(s) for s in s_vals]
    x_vals, y_vals = zip(*positions)
    
    # 绘制原始点和样条曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(x, y, 'o', label='Original points')
    plt.plot(x_vals, y_vals, '-', label='Spline curve')
    plt.title('2D Cubic Spline Interpolation')
    plt.legend()
    plt.grid(True)
    
    # 计算曲率
    curvatures = [spline2d.calc_curvature(s) for s in s_vals]
    
    # 绘制曲率图
    plt.subplot(212)
    plt.plot(s_vals, curvatures, '-', label='Curvature')
    plt.title('Curvature along Spline')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    test_spline_interpolation()
    test_spline2d()