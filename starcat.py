import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.utils.iers import IERS_A_URL_MIRROR
from astropy.table import Table
from utils import *

import config

# Load the CSV file
def load_cat(time):
    catalog = pd.read_csv(config.cat_path)
    
    # Observer's location (replace with actual latitude and longitude)
    observer_location = EarthLocation(lat=config.location[0]*u.deg, lon=config.location[1]*u.deg, height=config.location[2]*u.m)

    # Time of the observation
    obs_time = time
    Ra = catalog['RAICRS'].to_numpy()
    Dec = catalog['DEICRS'].to_numpy()
    magnitude = catalog['Vmag']
    name = catalog['HIP']

    # Coordinate transformation setup
    sky_coords = SkyCoord(Ra, Dec, frame='icrs', unit = "deg")
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
    altaz_coords = sky_coords.transform_to(altaz_frame)

    cat_table = Table([name, altaz_coords, magnitude], names=('name', 'coords', 'magnitude'))

    return cat_table

def generate_stars(cat_table):
    x_offset = config.zenith_offset[0]
    y_offset = config.zenith_offset[1]
    camera_rotation = config.camera_rotation
    radius = config.fov
    magnitude_limit = config.magLimit
    # Filter based on altitude greater than 10 degrees
    altitude_mask = cat_table['coords'].alt > config.altLimit * u.deg
    
    # Filter based on magnitude brighter than the magnitude limit
    magnitude_mask = cat_table['magnitude'] < magnitude_limit

    # Combine the masks with logical AND to apply both filters
    combined_mask = altitude_mask & magnitude_mask

    # Apply the filter to the table and return the result
    cat_table = cat_table[combined_mask]

    plotted_stars = []


    for i in range(len(cat_table['name'])):
        current_mag = cat_table['magnitude'][i]
        current_name = cat_table['name'][i]
        alt = cat_table['coords'].alt.degree[i]
        az = cat_table['coords'].az.degree[i]
        measured_brightness = 0
        cloudy_percent = 0
        measured_background = 0
        x, y = alt_az_to_pixel(alt, az, x_offset, y_offset, camera_rotation, radius, )
        
        new_star = [x, y, current_mag, current_name, alt, az, measured_brightness, cloudy_percent, measured_background]

        # Check if the new star is too close to any already plotted star
        for j, (px, py, pmag, pname, palt, paz, pbright, pcloudy_percent, pmeasured_background) in enumerate(plotted_stars):
            if np.sqrt((px - x)**2 + (py - y)**2) < config.starRadius:  # Checking if the distance is less than 50 pixels
                if current_mag < pmag:  # Compare magnitudes (lower is brighter)
                    # Remove the dimmer star and plot the brighter one instead
                    #print(f"Replacing dimmer star at ({px:.2f}, {py:.2f}) with brighter star HIP{cat_table['name'][i]}")
                    plotted_stars[j] = new_star  # Replace with the brighter star
                    break
                else:
                    # Skip plotting the new star if it's dimmer or equal in brightness
                    #print(f"Skipping dimmer or equally bright star HIP{cat_table['name'][i]} at ({x:.2f}, {y:.2f})")
                    break
        else:
            # If no break occurred, plot the new star
            #print(f"Plotting star HIP{cat_table['name'][i]}, which has a Vmag of {current_mag} and an alt/az of {cat_table['coords'].alt.degree[i]:.2f}, {cat_table['coords'].az.degree[i]:.2f}")
            plotted_stars.append(new_star)

    return plotted_stars

