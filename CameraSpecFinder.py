##This is a small script to solve for the camera parameters using Powell's method. 
##Camera parameters are the rotation, X/Y zenith pixels and the lens FOV in pixels
##Solved information is placed into the config file


import numpy as np
import scipy.ndimage
from PIL import Image
from scipy.optimize import minimize

from loadimage import AstroImage
from starcat import *
from utils import *
import config



astro_image = AstroImage(config.clear_night)

astro_image.normalize_image()
astro_image.stretch_image()

print(astro_image.loctime)

cat_table = load_cat(astro_image.obstime)
alt = cat_table['coords'].alt.degree
az = cat_table['coords'].az.degree



star_data = [
    (1131, 1554, 68.80322884010984, 39.24480239551958), #Antares
    (1888 , 1310, 69.07797121816508, 210.4047519639314), #Hadar
    (1682, 2225, 47.6367506417902, 310.8848082687554), #Spica
    (1167, 2564, 24.65669218799258, 340.5109585713347), #Arcturus
    (2690, 852, 13.575887970003345, 205.25271006081388), #Canopus
    (2008, 242, 13.833074934993572, 164.63939813372568), #Archenar
    (1100, 115, 6.2642251042113894, 125.61764474329657), #Fomalhaut
    (1919, 2730, 12.895717139739613, 305.19895362085333), #Denebola
    (206, 1157, 11.066070524919972, 65.89214602529995), #Altair
    (561, 762, 22.556919666447833, 88.91956017967185), #Dabih
    (2416, 2181, 25.401292542317734, 271.481631424655) #V Hya
]



initial_guess = [  17.61100481,  -39.38376641, -142.80925462, 1510.95693491]

result = minimize(
    objective_function, initial_guess, args=(star_data,),
    method='Powell', options={'disp': True, 'maxiter': 5000, 'maxfev': 50000, 'tol': 1e-8}

)

print("Optimized parameters:", result.x)

plot_select_stars(alt, az, astro_image, result.x[0],result.x[1],result.x[2],result.x[3])