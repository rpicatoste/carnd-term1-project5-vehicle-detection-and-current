
import glob

# Read in cars and notcars
# Big set
#images = glob.glob('*/*/*.png')
#training_image_type = 'png'
# Small set
images = glob.glob('course/*/*/*.jpeg')
training_image_type = 'jpeg'

cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    elif 'vehicles' in image:
        cars.append(image)

# Equalize the samples if different.
sample_size = min(len(cars), len(notcars))
#sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

print('Number of samples used: ', sample_size)

# Classifier parameters

# Parameters of the features selected.
# HOG
color_space = 'HLS' #'YUV' #  Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient          = 11 #9 # HOG orientations
pix_per_cell    = 16 #8 # HOG pixels per cell
cell_per_block  = 2     # HOG cells per block
hog_channel     = 'ALL' # Can be 0, 1, 2, or "ALL"
# Spatial binning
spatial_size = (16, 16) # Spatial binning dimensions
# Color histogram
hist_bins = 16    # Number of histogram bins

spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off



# Search parameters
   
combinations = {}

combinations['color'] = [ (0, 0,   255), 
                          (0, 255,  0), 
                          (255, 255, 0)]

combinations['window_size'] = [ #(70, 70),
                                (140, 140),
                                (230, 230)]

overlap_val = 0.5
combinations['overlap'] = [(overlap_val, overlap_val)] * 3

combinations['y_limit'] = [#(380, 500),
                           (390, 600),
                           (450, 720)]


combinations['num_window'] = list(range( len(combinations['color']) ))