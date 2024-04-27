import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits 
import config


class AstroImage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.color_image = None
        self.normalized_image = None
        self.stretched_image = None

        with fits.open(self.file_path) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header
    
        self.color_image = np.flipud(cv2.cvtColor(np.array(self.data, dtype=np.uint16), cv2.COLOR_BAYER_RG2RGB))
        self.obstime =self.header.get('DATE-OBS')
        self.loctime =self.header.get('DATE-LOC')
        
    
    def normalize_image(self):
        self.normalized_image = cv2.normalize(self.color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    def stretch_image(self):
        m = config.stretchB_factor
        c = config.stretchC_factor
        # Clipping threshold as a percentile of the image data
        self.stretched_image = np.where(self.normalized_image < c, 
                               np.power(self.normalized_image / c, np.log(0.5) / np.log(m)) * c,
                               c)
    
    def display_image(self):
        plt.imshow(np.clip(self.stretched_image, 0, 1))  # Clip to handle any possible overflows
        plt.axis('off')
        plt.show()

    def get_normalized_image(self):
        # Return a copy to prevent modification of the internal state
        return np.copy(self.normalized_image)
    




