# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:51:31 2024

@author: 39829
"""
import math
import numpy as np
from prettytable import PrettyTable

class CartesianFrenetConverter:
    @staticmethod
    def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
        dx = x - rx
        dy = y - ry

        cos_theta_r = np.cos(rtheta)
        sin_theta_r = np.sin(rtheta)

        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = np.copysign(np.sqrt(dx**2 + dy**2), cross_rd_nd)

        delta_theta = theta - rtheta
        tan_delta_theta = np.tan(delta_theta)
        cos_delta_theta = np.cos(delta_theta)

        one_minus_kappa_r_d = 1 - rkappa * d
        d_prime = one_minus_kappa_r_d * tan_delta_theta

        kappa_r_d_prime = rdkappa * d + rkappa * d_prime

        d_double_prime = (-kappa_r_d_prime * tan_delta_theta +
                          one_minus_kappa_r_d / cos_delta_theta**2 *
                          (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))

        s = rs
        s_prime = v * cos_delta_theta / one_minus_kappa_r_d

        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        s_double_prime = (a * cos_delta_theta -
                          s_prime**2 * (d_prime * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d

        return s, s_prime, s_double_prime, d, d_prime, d_double_prime

    @staticmethod
    def NormalizeAngle(angle):
        """Normalize the angle to be within the interval [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition):
        assert abs(rs - s_condition[0]) < 1.0e-6  # Ensure reference point s matches
    
        cos_theta_r = np.cos(rtheta)
        sin_theta_r = np.sin(rtheta)
    
        x = rx - sin_theta_r * d_condition[0]
        y = ry + cos_theta_r * d_condition[0]
    
        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
        tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
        delta_theta = np.arctan2(d_condition[1], one_minus_kappa_r_d)
        cos_delta_theta = np.cos(delta_theta)
    
        theta = CartesianFrenetConverter.NormalizeAngle(delta_theta + rtheta)
    
        kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
        kappa = (((d_condition[2] + kappa_r_d_prime * tan_delta_theta) * cos_delta_theta**2) / one_minus_kappa_r_d + rkappa) * cos_delta_theta / one_minus_kappa_r_d
    
        d_dot = d_condition[1] * s_condition[1]
        v = np.sqrt(one_minus_kappa_r_d**2 * s_condition[1]**2 + d_dot**2)
    
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        a = s_condition[2] * one_minus_kappa_r_d / cos_delta_theta + s_condition[1]**2 / cos_delta_theta * (d_condition[1] * delta_theta_prime - kappa_r_d_prime)
    
        return x, y, theta, kappa, v, a
    
    def cartesian_to_frenet_simple(rs, rx, ry, rtheta, x, y):
        """
        Convert Cartesian coordinates to simplified Frenet coordinates (s, d).
    
        Parameters:
        rs (float): Reference s-coordinate on the path.
        rx (float): x-coordinate of the reference point.
        ry (float): y-coordinate of the reference point.
        rtheta (float): Orientation (angle in radians) of the reference path at the reference point.
        x (float): x-coordinate of the point to convert.
        y (float): y-coordinate of the point to convert.
    
        Returns:
        float: s - longitudinal coordinate along the path.
        float: d - lateral offset from the path.
        """
        dx = x - rx
        dy = y - ry
    
        cos_theta_r = np.cos(rtheta)
        sin_theta_r = np.sin(rtheta)
    
        # Calculate the lateral offset (d)
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = np.copysign(np.sqrt(dx**2 + dy**2), cross_rd_nd)
        
        # The longitudinal position s is just the reference s
        s = rs
    
        return s, d
    
    def frenet_to_cartesian1D(rs, rx, ry, rtheta, s_condition, d_condition):
        
        if abs(rs - s_condition[0])>= 1.0e-6:
            print("The reference point s and s_condition[0] don't match")
            
        cos_theta_r = np.cos(rtheta)
        sin_theta_r = np.sin(rtheta)
        
        x = rx - sin_theta_r * d_condition[0]
        y = ry + cos_theta_r * d_condition[0]    
        
        return x, y


    def CalculateTheta(rtheta, rkappa, l, dl):
        """
        Calculate the new orientation (theta) in the Frenet frame.
        
        Parameters:
        rtheta (float): Reference path's orientation.
        rkappa (float): Reference path's curvature at the reference point.
        l (float): Lateral offset from the reference path.
        dl (float): Rate of change of the lateral offset.
    
        Returns:
        float: New orientation in the Frenet frame.
        """
        return CartesianFrenetConverter.NormalizeAngle(rtheta + math.atan2(dl, 1 - l * rkappa))
    
    def CalculateKappa(rkappa, rdkappa, l, dl, ddl):
        """
        Calculate the curvature (kappa) in the Frenet frame.
        
        Parameters:
        rkappa (float): Reference path's curvature.
        rdkappa (float): Derivative of the reference path's curvature.
        l (float): Lateral offset from the reference path.
        dl (float): Rate of change of the lateral offset.
        ddl (float): Second derivative of the lateral offset.
    
        Returns:
        float: New curvature in the Frenet frame.
        """
        denominator = (dl * dl + (1 - l * rkappa) * (1 - l * rkappa))
        if math.fabs(denominator) < 1e-8:
            return 0.0
        denominator = math.pow(denominator, 1.5)
        numerator = (rkappa + ddl - 2 * l * rkappa * rkappa -
                     l * ddl * rkappa + l * l * rkappa * rkappa * rkappa +
                     l * dl * rdkappa + 2 * dl * dl * rkappa)
        return numerator / denominator
    
    def CalculateCartesianPoint(rtheta, rpoint, l):
        x = rpoint.x - l * math.sin(rtheta)
        y = rpoint.y + l * math.cos(rtheta)
        return (x, y)

    def CalculateLateralDerivative(rtheta, theta, l, rkappa):
        return (1 - rkappa * l) * math.tan(theta - rtheta)
    
    def CalculateSecondOrderLateralDerivative(rtheta, theta, rkappa, kappa, rdkappa, l):
        dl = CartesianFrenetConverter.CalculateLateralDerivative(rtheta, theta, l, rkappa)
        theta_diff = theta - rtheta
        cos_theta_diff = math.cos(theta_diff)
        try:
            res = (-(rdkappa * l + rkappa * dl) * math.tan(theta - rtheta) +
                   (1 - rkappa * l) / (cos_theta_diff ** 2) *
                   (kappa * (1 - rkappa * l) / cos_theta_diff - rkappa))
        except ZeroDivisionError:
            res = float('inf')  # Handle division by zero if cos_theta_diff is zero
            print(f"Warning: result is inf when calculate second order lateral derivative.")
        return res


def test_conversion():
    rs, rx, ry, rtheta = 10.0, 0.0, 0.0, np.pi / 4
    rkappa, rdkappa = 0.1, 0.01
    x, y, v, a = -1.0, 1.0, 2.0, 0.0
    theta, kappa = np.pi / 3, 0.11
    
    # Tolerance for numerical comparisons
    tolerance = 1e-4

    # Convert from Cartesian to Frenet and print the results
    s, s_prime, s_double_prime, d, d_prime, d_double_prime = CartesianFrenetConverter.cartesian_to_frenet(
        rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa)
    
    table = PrettyTable(["Parameter", "Value"])
    table.add_rows([
        ["s", s],
        ["s'", s_prime],
        ["s''", s_double_prime],
        ["d", d],
        ["d'", d_prime],
        ["d''", d_double_prime]
    ])
    print("From Cartesian to Frenet:")
    print(table)

    # Convert from Frenet back to Cartesian and print the results
    x_out, y_out, theta_out, kappa_out, v_out, a_out = CartesianFrenetConverter.frenet_to_cartesian(
        rs, rx, ry, rtheta, rkappa, rdkappa, [s, s_prime, s_double_prime], [d, d_prime, d_double_prime])
    
    table = PrettyTable(["Parameter", "Value"])
    table.add_rows([
        ["x", x_out],
        ["y", y_out],
        ["theta", theta_out],
        ["kappa", kappa_out],
        ["v", v_out],
        ["a", a_out]
    ])
    print("From Frenet back to Cartesian:")
    print(table)
    
    results_table = PrettyTable(["Parameter", "Original", "Converted", "Difference", "Pass"])
    parameters = [
        ("x", x, x_out),
        ("y", y, y_out),
        ("theta", theta, theta_out),
        ("kappa", kappa, kappa_out),
        ("v", v, v_out),
        ("a", a, a_out)
    ]

    # Check results and populate table
    for param, original, converted in parameters:
        difference = np.abs(original - converted)
        pass_test = difference < tolerance
        results_table.add_row([param, f"{original:.4f}", f"{converted:.4f}", f"{difference:.4e}", "Yes" if pass_test else "No"])

    # Display results
    print("Conversion Accuracy Test Results:")
    print(results_table)


if __name__ == "__main__":
    test_conversion()
