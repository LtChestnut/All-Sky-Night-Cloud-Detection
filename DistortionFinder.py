# Center - 1534, 1465, 0
# Rigel - 2155, 596, 27.681095603362152, 73.54799889994952
# achernar - 1707, 1662, 76.52109168815042, 168.6965027762954
# fomalhaut - 980, 1712, 58.68916237881435, 285.66055552381385
# Canopus - 2402, 1498, 40.97627484667289, 128.76392525910086

# theta = angle from zenith

# DistFromCenter (pixels) = a0 + a1*theta + a2*theta^2...etc
# a0 = 0 ``

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def polynomial_projection(altitude, *coefficients):
    """Calculate radial distance r for a given altitude using polynomial coefficients, assuming no constant term."""
    altitude_rad = np.radians(altitude)
    theta = np.pi / 2 - altitude_rad
    r = 0
    for i, coeff in enumerate(coefficients):
        r += coeff * (theta ** (i + 1))  # Note i+1 to skip the constant term
    return r

# Example data
altitudes = np.array([27.681095603362152, 76.52109168815042, 
                      58.68916237881435, 40.97627484667289, 
                      80.37011635434025, 17.405146274289972,
                      36.44261585260307, 23.329541998362977,
                      44.11309689787155, 0])  # Altitudes in degrees
x_pixels = np.array([2155, 1707, 980, 2402, 1361, 2582, 1438, 412, 1388, 1350])  # x coordinates in pixels
y_pixels = np.array([596, 1662, 1712, 1498, 1593, 831, 536, 1140, 2315, 0])  # y coordinates in pixels

# Calculate r
center_x = 1534  # Assume center of the image
center_y = 1465  # Assume center of the image
r = np.sqrt((x_pixels - center_x)**2 + (y_pixels - center_y)**2)

# Initial guess for coefficients (quadratic form without the constant term)
initial_guess = [0, 0, 0, 0, 0, 0]  # Adjust this as needed for higher degrees

# Perform the curve fitting
coefficients, covariance = curve_fit(polynomial_projection, altitudes, r, p0=initial_guess)

print("Fitted coefficients:", repr(np.array(coefficients)))

# Generate a range of altitudes for plotting
plot_altitudes = np.linspace(0, 90, 100)
plot_r = polynomial_projection(plot_altitudes, *coefficients)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(altitudes, r, color='red', label='Actual data')
plt.plot(plot_altitudes, plot_r, label='Fitted model', color='blue')
plt.xlabel('Altitude (degrees)')
plt.ylabel('Radial distance (pixels)')
plt.title('Fit Polynomial Projection Model without Constant Term')
plt.legend()
plt.show()
