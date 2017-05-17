import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle


from functions import slide_window, draw_boxes
import parameters as pars
import heatmap as hm
import classifier as cl

global heat
video_image_size = (720, 1280, 3)
heat = np.zeros( video_image_size[0:2], dtype = np.float )



# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = cl.single_img_features(  test_img, color_space=color_space, 
                                            spatial_size=spatial_size, hist_bins=hist_bins, 
                                            orient=orient, pix_per_cell=pix_per_cell, 
                                            cell_per_block=cell_per_block, 
                                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
def pipeline_search( image, y_limit, window_size, 
                            overlap, color ):
    draw_image = np.copy( image )
 #print('Searching. Y limits: ', y_limit, '. Window size: ', window_size)
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
        
    draw_image = draw_boxes(draw_image, hot_windows, color=color, thick=6) 
    
    return hot_windows, draw_image
    
    
def pipeline_heat( image, return_images = False ):
    
    global heat
    
    # Cool down the heat map
#    print('heat before cooling', np.unique(heat))
    max_heat = 2
    heat[heat>max_heat] = max_heat
    heat[heat > 0] -= 1
    heat[heat<0] = 0
#    print('heat after cooling', np.unique(heat))
    
    for ii, window_size, color, overlap, y_limit in zip(pars.combinations['num_window'],
                                                        pars.combinations['window_size'],
                                                        pars.combinations['color'],
                                                        pars.combinations['overlap'],
                                                        pars.combinations['y_limit']):
        
        hot_windows, draw_image = pipeline_search( image, y_limit, window_size, 
                            overlap, color )
        
        # Add heat to each box in box list
        heat = hm.add_heat(heat, hot_windows)
        
#    print('heat before thres', np.unique(heat))
    # Apply threshold to help remove false positives
    heat = hm.apply_threshold(heat, 1)
#    print('heat after thres', np.unique(heat))
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = hm.label(heatmap)
    draw_img = hm.draw_labeled_bboxes(np.copy(image), labels)
    #Restart every cycle
#    heat = np.zeros_like( heat )
    
#    plt.figure()
#    plt.imshow(heatmap, cmap='hot')
#     
    
    if return_images:
        return draw_img, draw_image, heatmap
    else:
        return draw_img
    
def plot_grid( image ):
  
    draw_image = np.copy( image )
    
    for ii, window_size, color, overlap, y_limit in zip(pars.combinations['num_window'],
                                                        pars.combinations['window_size'],
                                                        pars.combinations['color'],
                                                        pars.combinations['overlap'],
                                                        pars.combinations['y_limit']):
        
        #print('Searching. Y limits: ', y_limit, '. Window size: ', window_size)
        windows = slide_window( image, 
                                x_start_stop = [None, None], 
                                y_start_stop = y_limit, 
                                xy_window = window_size, 
                                xy_overlap = overlap)
        
         
        draw_image = draw_boxes(draw_image, windows, color=color, thick=6)    

    
    plt.figure(figsize=(12,9))
    plt.imshow(draw_image)

    return draw_image        
  

def test_search_and_classify():
    
    image_files = ['test_images/test1.jpg',
                   'test_images/test2.jpg',
                   'test_images/test3.jpg',
                   'test_images/test4.jpg',
                   'test_images/test5.jpg',
                   'test_images/test6.jpg',
                   'course/bbox-example-image.jpg']
    image_files = ['test_images/test1.jpg']
 
    
    for image_file in image_files:
        
        image = mpimg.imread( image_file )
        # Uncomment the following line if you extracted training data from .png 
        # images (scaled 0 to 1 by mpimg) and the image you are searching is a .jpg 
        # (scaled 0 to 255)
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
        
        
if __name__ == '__main__':
        
    
   
    image_file = 'test_images/test1.jpg'
    image = mpimg.imread( image_file )
    plot_grid( image )
    
    test_search_and_classify()
    