##CONFIG FILE

##Camera Parameters
zenith = [1504, 1504]
zenith_offset = [17.38589195,  -41.35864954]
fov = 1192.40321059
camera_rotation = -142.81033535
location = [-43.9833, 170.4667, 1031]
size = 3008
disortion_coeffs = [0, 1370.48875333, -1554.24783798,  4270.9858841 , -6101.88129044,
        3947.88797537,  -932.02935122]
flux_a = 1449799.4772564045 # 1034331.76618567
flux_b = 22101.980314846012

DIR = 'C:/Users/cheha/Documents/School/cosc428-cloud-detector/'

##Script Paramters
cloudy_night = DIR + 'Inputs/cloudy_image.fit'
clear_night = DIR + 'Inputs/TestImage2.fit'
cloudy_night2 = DIR + 'Inputs/Cloudy_image2.fit'
simulated_cloud_night = DIR + 'Inputs/simulated_cloud_image.fit'
cat_path = DIR + 'Inputs/StarCatSmall.csv'
mask_path = DIR + 'Inputs/AllSkyMask.png'
stretchB_factor = 0.1
stretchC_factor = 2
altaz_DS_Factor = 5
altLimit = 5
starRadius = 50
magLimit = 5
apertureRadius = 7
annulusInnerRadius = 10
annulusOutterRadius = 12

