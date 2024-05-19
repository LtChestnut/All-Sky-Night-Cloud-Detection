import numpy as np
from scipy.optimize import curve_fit

from loadimage import AstroImage
from starcat import *
from utils import *
import config



def main():
    astro_image = AstroImage(config.clear_night)

    astro_image.normalize_image()
    astro_image.stretch_image()
    #astro_image.display_image()
    print(astro_image.loctime)

    cat_table = load_cat(astro_image.obstime)

    alt = cat_table['coords'].alt.degree
    az = cat_table['coords'].az.degree

    stars = generate_stars(cat_table)

    stars = measure_star_brightness(stars, astro_image)

    plot_stars(stars, astro_image, Markers = True)

    # Example data (replace with your actual data)
    magnitude = [star[2] for star in stars]
    measured_brightness = [star[6] for star in stars]

    # Perform curve fitting using least squares
    popt, pcov = curve_fit(stellar_brightness, magnitude, measured_brightness)

    # Extract the optimized parameters
    a_opt, b_opt = popt

    # Generate points for the logarithmic curve
    x_curve = np.linspace(min(magnitude), max(magnitude), 1000)
    y_curve = stellar_brightness(x_curve, a_opt, b_opt)

    # Plot the original data and the logarithmic curve
    plt.scatter(magnitude, measured_brightness, label='Measured Data')
    plt.plot(x_curve, y_curve, color='red', label='Logarithmic Fit')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Logarithmic Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the optimized parameters
    print("Optimized Parameters:")
    print("a =", a_opt)
    print("b =", b_opt)



main()


