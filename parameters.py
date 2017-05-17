
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
pix_per_cell    = 8 #8  # HOG pixels per      (pix_per_cell, pix_per_cell)
cell_per_block  = 2     # HOG cells per block (cell_per_block, cell_per_block)
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




def test_hog_features():
 
    image_files = [  'course\\vehicles_smallset\\cars3\\1188.jpeg',
                     'course\\vehicles_smallset\\cars3\\1189.jpeg',
                     'course\\non-vehicles_smallset\\notcars3\\extra317_158.jpeg',
                     'course\\non-vehicles_smallset\\notcars3\\extra324_147.jpeg']
#    image_files = ['test_images/test1.jpg']
 
    for image_file in image_files:
        
        image = mpimg.imread( image_file )
    
        cs.get_hog_features(image, 
                            hog_channel = hog_channel,
                            orient = orient, 
                            pix_per_cell = pix_per_cell,
                            cell_per_block = cell_per_block, 
                            feature_vec = True, 
                            plot_figure = True)


if __name__ == '__main__':
    import matplotlib.image as mpimg
    
    import classifier as cs
    # If this module is called directly, train the classifier and check the 
    # features selected.
    test_hog_features()
    
    
    