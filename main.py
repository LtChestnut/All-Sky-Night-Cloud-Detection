import numpy as np
import scipy.ndimage
from PIL import Image
from scipy.optimize import minimize

from loadimage import AstroImage
from starcat import *
from utils import *
import config


# Replace 'path_to_your_file.fits' with your actual file path
astro_image = AstroImage(config.image_path)
#astro_image.process_image(0.25, 2)
astro_image.normalize_image()
astro_image.stretch_image()
#astro_image.display_image()
print(astro_image.obstime)

#Alt_Az_Map = pixel_to_alt_az(astro_image.get_normalized_image())
# Extract altitude and azimuth data
#display_alt_az_map(Alt_Az_Map)

altaz_coords = load_cat(astro_image.obstime)

alt = altaz_coords.alt.degree
az = altaz_coords.az.degree

plot_select_stars(alt, az, astro_image, 0, 0, 215, 1500)

# Example star data: [( observed_x, observed_y, altiude, azimuth), ...]
star_data = [
    (2155, 596, 68.80322884010984 39.24480239551958),
    (1707, 1662, 69.07797121816508 210.4047519639314),
    (179, 1712, 47.6367506417902 310.8848082687554),
    (2402, 1498, 24.65669218799258 340.5109585713347),
    (1361, 1593, 13.833074934993572 164.6393978954292),
    (2582, 831, 42.62269956656191 176.86873088116383),
    (1438, 536, 12.895717139739613 305.19895362085333),
    (412, 1140, 11.066070524919972 65.89214602529995),
    (1388, 2315, 22.556919666447833 88.91956017967185
     25.401292542317734 271.481631424655)
]


# # Initial guesses for x_offset, y_offset, camera_rotation, radius
# initial_guess = [-15, 30, -37, 1510]

# # Run the optimizer
# result = minimize(
#     objective_function, initial_guess, args=(star_data,),
#     method='Nelder-Mead', options={'disp': True, 'maxiter': 5000, 'maxfev': 50000, 'tol': 1e-8}

# )

# print("Optimized parameters:", result.x)

# plot_select_stars(alt, az, astro_image, result.x[0],result.x[1],result.x[2],result.x[3])

# test_image = astro_image.color_image
# test_image = rgb2gray(test_image)
# sigma = 1.4
# log_image = scipy.ndimage.gaussian_laplace(test_image, sigma=sigma)


# # Create an Image object and save to PNG
# log_image_pil = Image.fromarray(log_image, mode='L')  # 'L' mode for grayscale
# log_image_pil.save('C:/Users/cheha/Documents/School/cosc428-cloud-detector/adjusted_log_image.png')

# # Displaying the original and processed images
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.imshow(astro_image.stretched_image)
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(test_image, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(log_image, cmap='gray')
# plt.title('Laplacian of Gaussian Image')
# plt.axis('off')

# plt.show()