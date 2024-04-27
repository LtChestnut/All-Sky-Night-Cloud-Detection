import numpy as np
import cv2
import matplotlib.pyplot as plt

import config

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
    r = 2 / np.sqrt(2) * radius * np.sin(theta / 2)
    #r = polynomial_projection(theta)

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
        (4907, 'Fomalhaut'),
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


def huber_loss(predicted, observed, delta):
    """Calculate the Huber loss for a given delta."""
    residual = np.abs(predicted - observed)
    return np.where(residual < delta, 0.5 * residual**2, delta * (residual - 0.5 * delta))

def objective_function(params, star_data, delta=1):
    x_offset, y_offset, camera_rotation, radius = params
    total_error = 0
    for observed_x, observed_y, altitude, azimuth in star_data:
        predicted_x, predicted_y = alt_az_to_pixel(altitude, azimuth, x_offset, y_offset, camera_rotation, radius)
        # Apply Huber loss to both x and y components
        error_x = huber_loss(predicted_x, observed_x, delta)
        error_y = huber_loss(predicted_y, observed_y, delta)
        total_error += error_x + error_y
    return total_error
