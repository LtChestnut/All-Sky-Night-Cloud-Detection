import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

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

    # Coordinate transformation setup
    sky_coords = SkyCoord(Ra, Dec, frame='icrs', unit = "deg")
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
    altaz_coords = sky_coords.transform_to(altaz_frame)

    return altaz_coords

