import numpy as np
from PIL import Image
from scipy.optimize import minimize


from loadimage import AstroImage
from starcat import *
from utils import *
import config


# Replace 'path_to_your_file.fits' with your actual file path
astro_image = AstroImage(config.clear_night)
#astro_image.process_image(0.25, 2)
astro_image.normalize_image()
astro_image.stretch_image()
astro_image.display_image()
#print(astro_image.loctime)
#print(np.shape(astro_image.get_normalized_image()))



cat_table = load_cat(astro_image.obstime)

alt = cat_table['coords'].alt.degree
az = cat_table['coords'].az.degree



#plot_select_stars(alt, az, astro_image, config.zenith_offset[0], config.zenith_offset[1], 
#                 config.camera_rotation, config.fov  )

stars = generate_stars(cat_table)

stars = measure_star_brightness(stars, astro_image)

#plot_stars(stars, astro_image, Markers = False)

stars = mask_stars(stars)

#plot_mag_vs_photometry(stars)

stars = calculate_cloud_percent(stars)


# plot_brightnesses(stars)
# plot_cloud_map(stars, astro_image)