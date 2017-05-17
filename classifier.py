import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

import parameters as pars


# Define a function to return HOG features and visualization
def get_hog_features(feature_image, hog_channel, orient, pix_per_cell, cell_per_block, feature_vec = True, plot_figure = False):

    if hog_channel == 'ALL':
        channels = list(range(feature_image.shape[2]))
    else:
        channels = [hog_channel]
    
    hog_features =[]
    hog_image = [np.zeros_like(feature_image)] * len(channels)
    
    for channel in channels:
        features, hog_image[channel] = hog( feature_image[:,:,channel], 
                                                orient, 
                                                (pix_per_cell, pix_per_cell),  
                                                (cell_per_block, cell_per_block), 
                                                transform_sqrt = True, 
                                                visualise = True, 
                                                feature_vector = True) 
        hog_features.append(features)
        
    
    hog_features = np.ravel(hog_features)        
    
    if plot_figure:
        fig, ax = plt.subplots(2,2, figsize = (12,12) )
        ax[0][0].imshow( feature_image )
        ax[0][0].axis('off')
        ax[0][1].imshow( hog_image[0] )
        ax[0][1].axis('off')
        ax[1][0].imshow( hog_image[1] )
        ax[1][0].axis('off')
        ax[1][1].imshow( hog_image[2] )
        ax[1][1].axis('off')
    
    return hog_features

# Define a function to compute binned color features.
def bin_spatial(img, size = (32, 32)):

    features = cv2.resize(img, size).ravel() 
    
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# Define a function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    # Define an empty list to receive features
    img_features = []
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    
    # Compute requested features
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
        
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
        
    if hog_feat == True:
        hog_features = get_hog_features(feature_image, hog_channel,
                                    orient, pix_per_cell, cell_per_block, feature_vec=True)
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)


# Define a function to extract features from a list of images
def extract_features(image_filenames, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file in image_filenames:
        
        # Read in each one by one
        image = mpimg.imread(file)

        img_features = single_img_features( image, 
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
        features.append(img_features)

    # Return list of feature vectors
    return features
    

# Function to train an SVC classifier with 2 sets of data: cars and not cars.
def train_classifier( cars, notcars, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    
    # Check the time for feature extraction and SVC training
    t=time.time()
    car_features = extract_features(pars.cars, 
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
    
    notcar_features = extract_features( pars.notcars, 
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
    
    X = np.vstack( (car_features, notcar_features) ).astype(np.float64)                        
    # Compute the mean and std to be used for later scaling.
    X_scaler = StandardScaler().fit(X)
    #Perform standardization by centering to zero and scaling to unit variance
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Use a SVC 
    svc = SVC( C = 2.0, kernel = 'rbf')
    svc.fit(X_train, y_train)
    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds for feature extraction and SVC training...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = {:.2%}'.format( round(svc.score(X_test, y_test), 4)) )
    # Check the prediction time for a single sample
    t = time.time()
    
    return svc, X_scaler

  
def check_datasets():
   
    # Verify that the car images are cars and the same for the notcars by 
    # plotting some randomly.

    plot_random_images( pars.cars, 'Car images' )
    plot_random_images( pars.notcars, 'Not car images' )

def plot_random_images( images_list, text = '' ):
    
    fig, ax = plt.subplots(4,4, figsize=(12,12))
    random_list = random.sample(range(0, len(images_list)), 16)
    
    for ii,num in enumerate(random_list):
        image = mpimg.imread(images_list[num])
        ax[int(np.floor(ii/4))][ii%4].imshow(image)
        ax[int(np.floor(ii/4))][ii%4].axis('off')
        
    fig.suptitle(text)

if __name__ == '__main__':
    # If this module is called directly, train the classifier and check the 
    # features selected.
    print('Training SVM classifier')
    
    check_datasets()
    
    svc, X_scaler = train_classifier(   pars.cars, 
                                        pars.notcars,
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
    
    pickle.dump( (svc, X_scaler), open( "trained_svc.p", "wb" ) )
    
else:
    # If this module is imported, load the trained SVC
    svc, X_scaler = pickle.load( open('trained_svc.p', mode='rb') )
    