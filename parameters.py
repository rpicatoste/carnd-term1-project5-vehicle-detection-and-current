
import glob

# Read in cars and notcars filenames
# Big set
#image_filenames_all = glob.glob('*/*/*.png')
#training_image_type = 'png'
# Small set
image_filenames_all = glob.glob('course/*/*/*.jpeg')
training_image_type = 'jpeg'

cars = []
notcars = []
for image_filename in image_filenames_all:
    if 'non-vehicles' in image_filename:
        notcars.append(image_filename)
    elif 'vehicles' in image_filename:
        cars.append(image_filename)

# Equalize the samples if different.
sample_size = min(len(cars), len(notcars))
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

print('Number of samples used: ', sample_size)

# Classifier parameters
# Parameters of the features selected.
# HOG
color_space = 'YUV' # 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
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

video_image_size = (720, 1280, 3)

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

image_files = ['test_images/test1.jpg',
               'test_images/test2.jpg',
               'test_images/test3.jpg',
               'test_images/test4.jpg',
               'test_images/test5.jpg',
               'test_images/test6.jpg']
    
   
if __name__ == '__main__':
    
    import classifier as cs
    import pickle
#    print('Training SVM classifier')
#    cs.check_datasets()    
#    svc, X_scaler = cs.train_classifier(   cars, 
#                                        notcars,
#                                        color_space = color_space, 
#                                        spatial_size = spatial_size, 
#                                        hist_bins = hist_bins, 
#                                        orient = orient, 
#                                        pix_per_cell = pix_per_cell, 
#                                        cell_per_block = cell_per_block, 
#                                        hog_channel = hog_channel, 
#                                        spatial_feat = spatial_feat, 
#                                        hist_feat = hist_feat, 
#                                        hog_feat = hog_feat)
    
    pickle.dump( (svc, X_scaler), open( "trained_svc.p", "wb" ) )
    # If this module is called directly, train the classifier and check the 
    # features selected.
    cs.test_hog_features()
    
    import search_and_classify as sc
    sc.plot_grid()        
    sc.test_search_and_classify()