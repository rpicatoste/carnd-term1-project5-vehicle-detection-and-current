import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label

import parameters as pars
import heatmap as hm
import classifier as cl

# Global variable heat. It will be used to keep the heatmap between frames of 
# the video.
global heat
heat = np.zeros( pars.video_image_size[0:2], dtype = np.float )


# Define a function that takes an image, start and stop positions in both x and 
# y, window size (x and y dimensions), and overlap fraction (for both x and y), 
# and return a list of boxes fullfiling those consitions.
def slide_window(   img, x_start_stop = [None, None], y_start_stop = [None, None], 
                    xy_window = (64, 64), xy_overlap = (0.5, 0.5)):
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    # Initialize a list to append window positions to.
    window_list = []
    # Loop through finding x and y window positions.
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Function to plot the grid of bounding boxes that will be passed to the 
# classifier, using the paramters condigured. It will plot them over an example
# image.
def plot_grid():
  
    image = mpimg.imread( pars.image_files[0] )
    draw_image = np.copy( image )
    
    for ii, window_size, color, overlap, y_limit in zip(pars.combinations['num_window'],
                                                        pars.combinations['window_size'],
                                                        pars.combinations['color'],
                                                        pars.combinations['overlap'],
                                                        pars.combinations['y_limit']):
        
        windows = slide_window( image, 
                                x_start_stop = [None, None], 
                                y_start_stop = y_limit, 
                                xy_window = window_size, 
                                xy_overlap = overlap)
         
        draw_image = draw_boxes(draw_image, windows, color=color, thick=6)    

    plt.figure(figsize=(12,9))
    plt.imshow(draw_image)

    return draw_image        

    
# This function will get an image, a set of windows and a classifier. It will
# apply the classifier to each window of the image, and will return a list of 
# the positive ones.
def search_windows( img, windows, clf, scaler, color_space = 'RGB', 
                    spatial_size = (32, 32), hist_bins = 32, 
                    hist_range = (0, 256), orient = 9, 
                    pix_per_cell = 8, cell_per_block = 2, 
                    hog_channel = 0, spatial_feat = True, 
                    hist_feat = True, hog_feat = True):

    on_windows = []
    
    for window in windows:

        # Resize the window to fit the trained classifier.
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        
        # Extract features for that window using single_img_features()
        features = cl.single_img_features(  test_img, 
                                            color_space = color_space, 
                                            spatial_size = spatial_size, 
                                            hist_bins = hist_bins, 
                                            orient = orient, 
                                            pix_per_cell = pix_per_cell, 
                                            cell_per_block = cell_per_block, 
                                            hog_channel = hog_channel, 
                                            spatial_feat = spatial_feat, 
                                            hist_feat = hist_feat, 
                                            hog_feat = hog_feat)
        
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        
        # Predict using the classifier
        prediction = clf.predict(test_features)
        
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            
    # Returns a list of windows            
    return on_windows
    
# The pipeline for the search and classify will receive an image and apply the
# configures grid to the classifier, and return the lost of boxes with a case 
# detected and an image with them drawn on it.    
def pipeline_search( image, y_limit, window_size, overlap, color ):
    
    image_with_bboxes = np.copy( image )
 
    windows = slide_window( image, 
                            x_start_stop = [None, None], 
                            y_start_stop = y_limit, 
                            xy_window = window_size, 
                            xy_overlap = overlap)
    
    hot_windows = search_windows(   image, windows, cl.svc, cl.X_scaler, 
                                    color_space = pars.color_space, 
                                    spatial_size = pars.spatial_size, 
                                    hist_bins = pars.hist_bins, 
                                    orient = pars.orient, 
                                    pix_per_cell = pars.pix_per_cell, 
                                    cell_per_block = pars.cell_per_block, 
                                    hog_channel = pars.hog_channel, 
                                    spatial_feat = pars.spatial_feat, 
                                    hist_feat = pars.hist_feat, 
                                    hog_feat = pars.hog_feat)                       
        
    image_with_bboxes = draw_boxes(image_with_bboxes, hot_windows, color=color, thick=6) 
    
    return hot_windows, image_with_bboxes
    
# The heat pipeline will receive images, expected to be sequential, and will 
# apply the heatmap on the result. It will hold results from one fram to the 
# next, applying a cooling each time for the detections that move place in the 
# image.
def pipeline_heat( image, return_images = False ):
    
    global heat
    hot_windows_all = []
    # Cool down the heat map
#    print('heat before cooling', np.unique(heat))
    max_heat = 4
    heat[heat>max_heat] = max_heat
    heat[heat > 0] -= 1
    heat[heat < 0] = 0
#    print('heat after cooling', np.unique(heat))
    
    for ii, window_size, color, overlap, y_limit in zip(pars.combinations['num_window'],
                                                        pars.combinations['window_size'],
                                                        pars.combinations['color'],
                                                        pars.combinations['overlap'],
                                                        pars.combinations['y_limit']):
        hot_windows,_  = pipeline_search( image, 
                                          y_limit, 
                                          window_size, 
                                          overlap, 
                                          color )
        # Add heat to each box in box list
        heat = hm.add_heat(heat, hot_windows)
        hot_windows_all = hot_windows_all + hot_windows
        
    # Apply threshold to help remove false positives
#    print('heat before thres', np.unique(heat))
    heat = hm.apply_threshold(heat, 1)
#    print('heat after thres', np.unique(heat))
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    image_labeled = hm.draw_labeled_bboxes(np.copy(image), labels)
    #Restart every cycle
#    heat = np.zeros_like( heat )
    image_with_bboxes = draw_boxes(image, hot_windows_all, color=color, thick=6)
#    plt.figure()
#    plt.imshow(heatmap, cmap='hot')
    
    if return_images:
        return image_labeled, image_with_bboxes, heatmap
    else:
        return image_labeled
    

def test_search_and_classify():
        
    # Reset heat when testing.
    global heat
    heat = np.zeros( pars.video_image_size[0:2], dtype = np.float )
    for image_file in pars.image_files:
                
        image = mpimg.imread( image_file )
        
        # If the training data is extracted from .png images (scaled 0 to 1 by 
        # mpimg) and the image you are searching is a .jpg (scaled 0 to 255).
        if pars.training_image_type == 'png':
            image = image.astype(np.float32)/255
        else:
            pass
        
        draw_img, draw_image, heatmap = pipeline_heat( image, return_images = True )
            
        fig, ax = plt.subplots(3,1, figsize = (13,12) )
        ax[0].imshow(draw_image)
        ax[0].set_title('Original detections')
        
        ax[1].imshow(draw_img)
        ax[1].set_title('Car Positions')
        
        ax[2].imshow(heatmap, cmap='hot')
        ax[2].set_title('Heat Map')
        fig.tight_layout()    
        fig.show()
        fig.waitforbuttonpress()
        