import numpy as np
import cv2
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import config
from scipy.interpolate import griddata
from matplotlib.patches import Circle, Wedge
from PIL import Image
from matplotlib.colors import Normalize

def downsample_image(image, factor):
    """ Downsample the image by the given factor. """
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_LINEAR)

def upsample_data(data, original_shape):
    """ Upsample the data to match the original image shape using interpolation. """
    return cv2.resize(data, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

def pixel_to_alt_az(image):
    """
    Convert pixel coordinates in a downsampled all-sky camera image to altitude and azimuth.

    Parameters:
        image (np.array): The downsampled image from the all-sky camera.
        zenith (tuple): The (x, y) coordinates of the zenith pixel in the full-resolution image.
        FOV (float): The field of view of the camera in degrees.
        downsample_factor (int): The factor by which the image was downsampled.

    Returns:
        np.array: A 3D array where each element contains [altitude, azimuth].
    """
    zenith = config.zenith
    FOV = config.fov
    downsample_factor = config.altaz_DS_Factor
    orig_height, orig_width = image.shape[:2]
    downsampled_image = downsample_image(image, downsample_factor)
    height, width = downsampled_image.shape[:2]
    alt_az_map = np.zeros((height, width, 2))

    # Adjust zenith position for the downsampled image
    zenith_downsampled = (zenith[0] // downsample_factor, zenith[1] // downsample_factor)

    for y in range(height):
        for x in range(width):
            dx = x - zenith_downsampled[0]
            dy = y - zenith_downsampled[1]
            r = np.sqrt(dx**2 + dy**2)
            max_r = np.sqrt(zenith_downsampled[0]**2 + zenith_downsampled[1]**2)  # approx radius to image edge

            theta = (r / max_r) * (FOV / 2)  # in degrees
            altitude = 90 - theta
            
            azimuth = np.degrees(np.arctan2(dy, dx)) + config.camera_rotation
            azimuth = (azimuth + 360) % 360  # normalize azimuth to [0, 360)

            alt_az_map[y, x, 0] = altitude
            alt_az_map[y, x, 1] = azimuth

    alt_az_map = upsample_data(alt_az_map, (orig_height, orig_width))
    return alt_az_map

def alt_az_to_pixel(altitude, azimuth, x_offset, y_offset, camera_rotation, radius):
    rotation_rad = np.deg2rad(camera_rotation)
    # Convert angles from degrees to radians
    altitude_rad = np.deg2rad(altitude)
    azimuth_rad = np.deg2rad(azimuth)

    # Calculate theta for the equidistant projection
    theta = np.pi/2 - altitude_rad  # 90 degrees - altitude

    # Image properties
    center_x = config.zenith[0] + x_offset
    center_y = config.zenith[1] + y_offset
    #Equation from https://paulbourke.net/dome/fisheyecorrect/meike35.png
    r = (2 / np.sqrt(2)) * radius * (0.6475*theta-0.002*theta**2-0.0331*theta**3-0.00010171*theta**4) 

    # Adjust azimuth for camera rotation (considering counter-clockwise as positive direction)
    azimuth_rad = azimuth_rad + rotation_rad

    # Convert polar coordinates (r, azimuth) to Cartesian coordinates (x, y)
    x = center_x + r * np.sin(azimuth_rad)
    y = center_y - r * np.cos(azimuth_rad)  # Negative to adjust for the typical y-coordinate in images

    return int(x), int(y)

def polynomial_projection(theta):
    """ Calculate radial distance r for a given theta using polynomial coefficients. """
    # Calculate r as a polynomial function of theta: r = a0 + a1*theta + a2*theta^2 + ...
    r = np.polyval(config.disortion_coeffs[::-1], theta)
    return r


def display_alt_az_map(Alt_Az_Map):
    altitude = Alt_Az_Map[:, :, 0]
    azimuth = Alt_Az_Map[:, :, 1]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for altitude
    alti_plot = axes[0].imshow(altitude, cmap='viridis', aspect='auto')
    axes[0].set_title('Altitude')
    axes[0].set_xlabel('Pixel X Coordinate')
    axes[0].set_ylabel('Pixel Y Coordinate')
    fig.colorbar(alti_plot, ax=axes[0], orientation='vertical', label='Altitude (degrees)')

    # Plot for azimuth
    azi_plot = axes[1].imshow(azimuth, cmap='cividis', aspect='auto')
    axes[1].set_title('Azimuth')
    axes[1].set_xlabel('Pixel X Coordinate')
    axes[1].set_ylabel('Pixel Y Coordinate')
    fig.colorbar(azi_plot, ax=axes[1], orientation='vertical', label='Azimuth (degrees)')

    # Show plots
    plt.tight_layout()
    plt.show()


def plot_select_stars(alt, az, image,  x_offset, y_offset, camera_rotation, radius):
    
    StarIndex = [
        (3451, 'Antares'),
        (2944, 'Hadar'),
        (2828, 'Spica'),
        (2970, 'Arcturus'),
        (1324, 'Canopus'),
        (271, 'Archenar'),
        (4807, 'Fomalhaut'),
        (2538, 'Denebola'),
        (4178, 'Altair'),
        (4298, 'Dabih'),
        (2365, 'V Hya'),
    ]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image.stretched_image, cmap='gray', origin='lower')
    plt.colorbar()

    #plot North
    x, y = alt_az_to_pixel(0, 0,  x_offset, y_offset, camera_rotation, radius)
    plt.scatter(x, y, s=20, color='green', edgecolor='white')

    print(x, y)


    #plot zenith
    x, y = alt_az_to_pixel(90, 0,  x_offset, y_offset, camera_rotation, radius)
    plt.scatter(x, y, s=20, color='blue', edgecolor='white')

    
    for data in StarIndex:
        #Stars
        x, y = alt_az_to_pixel(alt[data[0]], az[data[0]],  x_offset, y_offset, camera_rotation, radius)
        print(data[1], alt[data[0]], az[data[0]])
        plt.scatter(x, y, s=20, color='red', edgecolor='white')

    plt.title('Overlay of Star Catalog on Image')
    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def least_squares_loss(predicted, observed, delta):
    """Calculate the least squares loss."""
    residual = predicted - observed
    return 0.5 * np.sum(residual ** 2)

def objective_function(params, star_data, delta=1):
    x_offset, y_offset, camera_rotation, radius = params
    total_error = 0
    for observed_x, observed_y, altitude, azimuth in star_data:
        predicted_x, predicted_y = alt_az_to_pixel(altitude, azimuth, x_offset, y_offset, camera_rotation, radius)
        # Apply Huber loss to both x and y components
        error_x = least_squares_loss(predicted_x, observed_x, delta)
        error_y = least_squares_loss(predicted_y, observed_y, delta)
        total_error += error_x + error_y
    return total_error

def plot_stars(stars, image, Markers):
    fig, ax = plt.subplots(figsize=(10, 8))
    #plt.figure(figsize=(10, 8))
    ax.imshow(image.stretched_image, cmap='gray', origin='lower')

    # Initialize a list to keep track of plotted star positions and their magnitudes
    
    # Plot all stars that made it through the check
    for i in range(len(stars)):
        center = [stars[i][0], stars[i][1]]
        plt.scatter(center[0], center[1], s=20, color='red', edgecolor='white')
        if Markers == True:
            circle = Circle(center, config.apertureRadius, color='g', fill=False, linewidth=2)
            ax.add_patch(circle)

            # Define the inner and outer radii for the annulus
            inner_radius = config.annulusInnerRadius
            outer_radius = config.annulusOutterRadius

            # Draw and shade the annulus
            annulus = Wedge(center, outer_radius, 0, 360, width=outer_radius - inner_radius, color='green', fill=True, alpha=0.2)
            ax.add_patch(annulus)

    plt.title('Overlay of Star Catalog on Image')
    plt.show()

def measure_star_brightness(stars, image):
    green_channel = image.color_image[:, :, 1]
    for i in range(len(stars)):
        #positions = (find_brightest_point(green_channel, stars[i][0], stars[i][1], 20))
        positions = stars[i][0], stars[i][1]
        # Create the main apertures and the annuli
        apertures = CircularAperture(positions, r=config.apertureRadius)
        annuli = CircularAnnulus(positions, r_in=config.annulusInnerRadius, r_out=config.annulusOutterRadius)

        # Perform the aperture photometry on both the apertures and the annuli
        photometry_main = aperture_photometry(green_channel, apertures)

        # Extract the pixel values for the annulus and calculate the lower quartile mean
        annulus_mask = annuli.to_mask(method='center')  # Get the single mask
        annulus_data = annulus_mask.multiply(green_channel)  # Apply the mask to the image data
        annulus_data = annulus_data[annulus_mask.data > 0]  # Extract non-zero values
        background_mean = lower_quartile_mean(annulus_data)
        

        # Calculate the total background within the main aperture area
        background_sum = background_mean * apertures.area

        # Subtract the background from the main aperture sum
        photometry_main['aperture_sum'] -= background_sum

        stars[i][6] = photometry_main['aperture_sum'][0]
        stars[i][8] = background_mean

    return stars

def lower_quartile_mean(data):
    sorted_data = np.sort(data.ravel())
    lower_quartile = sorted_data[:len(sorted_data) // 75]
    return np.mean(lower_quartile)

def find_brightest_point(image_data, x_coord, y_coord, search_radius):
    # Extract a sub-array around the initial coordinates
    x_min = max(0, x_coord - search_radius)
    x_max = min(image_data.shape[1], x_coord + search_radius + 1)
    y_min = max(0, y_coord - search_radius)
    y_max = min(image_data.shape[0], y_coord + search_radius + 1)
    sub_image = image_data[y_min:y_max, x_min:x_max]

    # Find the brightest point in the sub-image
    relative_y, relative_x = np.unravel_index(np.argmax(sub_image), sub_image.shape)
    brightest_x = x_min + relative_x
    brightest_y = y_min + relative_y

    return brightest_x, brightest_y


def plot_mag_vs_photometry(stars):
    for i in range(len(stars)):
        plt.scatter(stars[i][2], stars[i][6], s=20, color='red', edgecolor='white')

    
    plt.title('Known magnitude vs measured brightness')
    plt.show()

def stellar_brightness(magnitude, a, b):
    return a * np.power(10, -0.4 * magnitude) + b

def calculate_cloud_percent(stars):
    for i in range(len(stars)):
        estimated_flux = stellar_brightness(stars[i][2], config.flux_a, config.flux_b)
        actual_flux = stars[i][6]

        cloud_percent = max(min(actual_flux / estimated_flux, 1), 0)

        stars[i][7] = cloud_percent

    return stars

def plot_brightnesses(stars):
    # Example data (replace with your actual data)
    x = [star[0] for star in stars]
    y = [star[1] for star in stars]
    brightness = [star[7] for star in stars]

    # Define grid to interpolate over
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate brightness values over grid
    brightness_interp = griddata((x, y), brightness, (xi, yi), method='cubic')

    # Plot the interpolated scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=brightness, cmap='magma', marker='o', edgecolors='none')
    plt.imshow(brightness_interp, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='magma', alpha=0.5)
    plt.colorbar(label='Brightness')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Measured and Interpolated Brightness Values')
    plt.show()

def mask_stars(stars):
    """
    Filters out coordinates that fall on the black part of a mask.

    :param coords: List of (x, y) tuples.
    :param mask_path: Path to the binary mask image.
    :return: List of (x, y) tuples that are on the white part of the mask.
    """
    # Load the mask image
    mask = Image.open(config.mask_path)
    mask = np.array(mask)


    # Filter coordinates
    filtered_coords = [coord for coord in stars if mask[coord[1], coord[0]] == 255]

    return filtered_coords


def plot_cloud_map(stars, astro_image):
    # Display the astronomical image
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Assuming astro_image has attributes for its dimensions or define them if known
    
    extent = [0, config.size, 0, config.size]
    
    ax.imshow((np.clip(astro_image.stretched_image, 0, 1)), 
              cmap='gray', extent=extent)
    
    # Extract data from stars
    x = [star[0] for star in stars]
    y = [star[1] for star in stars]
    brightness = [star[7] for star in stars]

    # Define grid to interpolate over
    xi = np.linspace(min(x), max(x), config.size)  # Match the width of astro_image
    yi = np.linspace(min(y), max(y), config.size) # Match the height of astro_image
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate brightness values over grid
    brightness_interp = griddata((x, y), brightness, (xi, yi), method='cubic', rescale = False)

    # Normalize brightness values for alpha calculation
    # Assuming brightness values are scaled between 0 and 1 for simplicity
    normalized_brightness = Normalize(vmin=0, vmax=1)(brightness_interp)

    # Create an alpha map where alpha is linearly decreasing with brightness
    alpha_map = 1.0 - normalized_brightness  # 1 means fully opaque (at 0 brightness), 0 means fully transparent (at max brightness)

    # Create a red colormap
    red_color = np.zeros((config.size, config.size, 4))
    red_color[:, :, 0] = 1  # Red channel
    red_color[:, :, 3] = alpha_map  # Alpha channel

    # Overlay the red transparency map
    ax.imshow(red_color, extent=(min(x), max(x), min(y), max(y)), interpolation='nearest')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Overlay of Star Catalog and Cloud Map on Image')
    ax.set_xlim([0, config.size])
    ax.set_ylim([0, config.size])

    plt.show()