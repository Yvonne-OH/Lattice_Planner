# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:13:13 2024

@author: 39829
"""

import numpy as np
import matplotlib.pyplot as plt
import Cubic_Spline

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

def find_reference_point(spline, given_point, sample_num=35,En_test=False):
    
    """
    Calculate the reference point on a cubic spline that is closest to a given point.

    This function calculates the point on a provided cubic spline curve that is nearest to a specified point in the plane. It uses linear interpolation between the nearest sampled points on the spline to estimate the exact position, orientation (yaw), and curvature at the closest approach. The function also visually represents the curve, the given point, and the calculated reference point using a plot.

    Parameters:
    spline (Cubic_Spline.Spline2D): The spline object representing the cubic spline curve.
    point (tuple): A tuple (x, y) representing the coordinates of the point to which the closest point on the spline is sought.
    sample_num (int, optional): The number of samples to take along the spline for analysis. Default is 200.

    Returns:
    tuple: Returns a tuple containing the interpolated coordinates (x, y), yaw angle (theta), curvature (k), and the arc length position (s) of the closest point on the spline.

    Raises:
    ValueError: If the closest point is found to be at the very start or the very end of the spline, which may indicate boundary issues in the interpolation process.

    Example:
    >>> middleLine = Cubic_Spline.Spline2D(np.array([-5.0, -2.5, 0.4, 2.0, 6.0, 7.5]), np.array([0.0, 0.0, 0.8, 1.5, 0.6, 0.0]))
    >>> given_point = (1.9, 2.6)
    >>> calculate_reference_point(middleLine, given_point)
    (2.0, 1.5, 0.785, 0.1, 4.5)

    Notes:
    - The function also plots the spline curve, the given point, and the direction vectors at sampled points along the spline to visually assess the alignment and proximity of the closest point.
    - This function depends on the 'Cubic_Spline' module, which should define the Spline2D class and necessary methods like calc_position, calc_yaw, and calc_curvature.
    """
    
    # Calculate samples along the spline and measure distances from the given point
    s_values = np.linspace(0, max(spline.s), sample_num)
    distances = []
    points = []
    
    for s in s_values:
        px, py = spline.calc_position(s)
        theta = spline.calc_yaw(s)
        k = spline.calc_curvature(s)
        k_prime = spline.calc_curvature_derivative(s)
        points.append(np.array((px, py, theta, k, k_prime,s)))
        distance = np.sqrt((px - given_point[0])**2 + (py - given_point[1])**2)
        distances.append(distance)
    
    points = np.array(points)

    # Find the index of the closest point
    min_index = np.argmin(distances)
    
    # Raise an error if the closest point is at the boundary of the spline
    if min_index <= 3:
        raise ValueError("The closest point is the starting point of the spline.")
    if min_index >= sample_num - 4:
        raise ValueError("The closest point is the end point of the spline.")

    # Get indices for linear interpolation
    index_R0 = min_index - 1
    index_R1 = min_index + 1
    
    # Get relevant points for interpolation
    x0, y0, theta0, k0, k_prime0, s0 = points[index_R0]
    xr, yr, thetar, kr, k_primer, sr = points[min_index]
    x1, y1, theta1, k1, k_prime1, s1 = points[index_R1]

    # Calculate vectors v0 = R0R and v1 = R1R
    v0 = np.array([xr - x0, yr - y0])
    v1 = np.array([x1 - xr, y1 - yr])

    # Calculate the projection distance Δs
    delta_s = np.dot(v0, v1) / np.linalg.norm(v1)
    
    w = delta_s / (s1 - s0)  # Linear interpolation weight

    # Compute interpolated point
    x_proj = (1 - w) * x0 + w * x1
    y_proj = (1 - w) * y0 + w * y1
    theta_proj = (1 - w) * theta0 + w * theta1
    k_proj = (1 - w) * k0 + w * k1
    dk_proj = (1 - w) * k_prime0 + w * k_prime1
    s_proj = s0 + delta_s

    # Plot results using quiver for direction
    px = points[:, 0]
    py = points[:, 1]
    theta = points[:, 2]
    u = np.cos(theta)  # X component of direction vectors
    v = np.sin(theta)  # Y component of direction vectors

    if (En_test):
        # Set up plot
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        ax.plot(px, py,label="Reference_line")
        ax.plot(x_proj,y_proj,'o', markerfacecolor='b', markeredgecolor='b', markersize=5,label="Proj_point")
        ax.plot([x0,given_point[0]], [y0,given_point[1]], marker='.',color='c')
        ax.plot([x1,given_point[0]], [y1,given_point[1]], marker='.',color='c')
        ax.plot(px, py,'o', markerfacecolor='none', markeredgecolor='k', markersize=2)
        ax.quiver(px, py, u, v, scale=25, color='m',label="Tangent vector")
        ax.plot([given_point[0],x_proj],[given_point[1],y_proj],color='k',label='Perpendicular line')
        ax.plot([x0,x1],[y0,y1],color='r',label='Auxiliary line')
        ax.plot(given_point[0],given_point[1],'o', markerfacecolor='k', markeredgecolor='k', markersize=5,label="Path_point")
        ax.set_title('Finding the Perpendicular Reference Point on a Curve')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        dot_product = np.dot(np.array([given_point[0]-x_proj, given_point[1]-y_proj]), v1)
        norm_u = np.linalg.norm(np.array([given_point[0]-x_proj, given_point[1]-y_proj]))
        norm_v = np.linalg.norm(v1)
        cos_theta = dot_product / (norm_u * norm_v)
        theta = np.degrees(np.arccos(cos_theta))
        fig.text(0.95, 0.65, f'Angle: {theta:.2f}°')
        
        plt.grid(True)
        plt.show()

    # Return calculated values
    return (x_proj, y_proj, theta_proj, k_proj, dk_proj,s_proj)
        

if __name__=='__main__':
    # Example usage:
    x = np.array([-5.0, -2.5, 0.4, 2.0, 6.0, 7.5])
    y = np.array([0.0, 0.0, 0.8, 1.2, 0.6, 0.0])
    middleLine = Cubic_Spline.Spline2D(x, y)  # x and y should be defined outside this function
    given_point = (0.9, 2.6)
    find_reference_point(middleLine, given_point,En_test=True)
