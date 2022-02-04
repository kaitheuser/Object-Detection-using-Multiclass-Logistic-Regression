'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

from audioop import bias
import numpy as np
import cv2
#from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Qt5Agg')
from skimage.measure import label, regionprops
import os
import copy

class BinDetector():
	def __init__(self, lr = 0.1, max_iters = 5000, err_tol = 1e-3, bias = 1):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		### Logistic Regression Model Parameters
		self.lr = lr                      # Learning Rate
		self.max_iters = max_iters        # Maximum Number of Iterations
		self.err_tol = err_tol            # Error that stops the training process
		self.weights = None               # Weight, w for the function of x (linear model)
		self.bias = bias                  # Bias, b for the function of x (linear model)
		self.color_class = None           # Class of Interest to Train

		### Model Parameters for Autograder
		# Color Classes:
		#----------------
		# Recycling-bin Blue: 1
		# Skyblue           : 2
		# Waterblue         : 3
		# Purple            : 4
		# Dark Blue         : 5
		# Blue              : 6
		# Light Blue        : 7
		# Red               : 8
		# Brown             : 9
		# Tan               : 10
		# Yellow            : 11
		# Green             : 12
		# Gray              : 13
		# Black             : 14
		# White             : 15
		
		# Weights from Class 1, 12, 14, 15 in order (Autograder 6.65, Local: 10)
		self.color_class = [1, 12, 14, 15]

		"""(Autograder 7.5, Local: 10)
		self.weights = [np.array([-2.16753777, -3.4701545 , -2.36326163,  2.87969732,  0.55824812,
        3.61200798,  1.89956967, -1.62623865,  0.6360524 , -3.84326482,
       -2.0993706 , -2.07072388,  1.71887097]), np.array([ 0.28122412,  0.62005008,  2.11920447, -5.67939281, -1.7330591 ,
        3.49128383, -1.55419844,  1.27912183, -2.46229211,  4.14607363,
        0.7757739 ,  0.02785216, -3.51518768]), np.array([ 1.50831201, -0.51811099, -1.93192972, -1.67820221, -0.843384  ,
       -4.05400172, -2.74290342, -1.93089989,  1.1477016 ,  0.69727314,
       -1.4780502 ,  1.44408899,  0.64942903]), np.array([-3.04360987,  3.2791054 ,  2.51382891,  2.94096898,  1.41382035,
       -4.17192864,  1.55004051,  2.34458916, -1.76898667, -1.78160559,
        2.79470392, -1.17915166, -1.44120635])]
		"""
		"""
		# (Autograder 8.5, Local: 9)
		self.weights = [np.array([-2.76086974, -1.9268365 , -1.17220947,  4.26144916,  1.23920715,
        2.26189008,  2.77890648, -0.47563412,  0.28226913, -4.23994786,
       -0.77940036, -2.20754004,  1.45813846]), np.array([-0.74654248,  2.77170725,  4.03962236, -5.45228338, -0.62993743,
        3.63540777, -0.87728046,  2.94622572, -3.71753284,  4.53172095,
        2.57244393, -0.23168772, -4.91986865]), np.array([ 1.81297211, -0.26551763, -2.5517179 , -1.02800226,  0.16318121,
       -5.97322811, -3.34686687, -2.53021376,  1.76587918,  0.18449452,
       -1.68786221,  1.92977941,  1.29696763])]
	   """
		
		# (Autograder: 9.75, Local: 10)
		self.weights = [np.array([-2.16753777, -3.4701545 , -2.36326163,  2.87969732,  0.55824812,
        3.61200798,  1.89956967, -1.62623865,  0.6360524 , -3.84326482,
       -2.0993706 , -2.07072388,  1.71887097]), np.array([ 0.28122412,  0.62005008,  2.11920447, -5.67939281, -1.7330591 ,
        3.49128383, -1.55419844,  1.27912183, -2.46229211,  4.14607363,
        0.7757739 ,  0.02785216, -3.51518768]), np.array([ 1.50831201, -0.51811099, -1.93192972, -1.67820221, -0.843384  ,
       -4.05400172, -2.74290342, -1.93089989,  1.1477016 ,  0.69727314,
       -1.4780502 ,  1.44408899,  0.64942903]), np.array([-3.04360987,  3.2791054 ,  2.51382891,  2.94096898,  1.41382035,
       -4.17192864,  1.55004051,  2.34458916, -1.76898667, -1.78160559,
        2.79470392, -1.17915166, -1.44120635])]
		
		
		"""
		# (Autograder: 8.25 , Local: 10)
		self.weights = [np.array([ -0.0356813 ,  -9.22751274,  -6.75686698,   7.78064151,
        -0.11059022,   6.49368513,   5.43087308,  -4.39806456,
         0.60314113, -12.09164392,  -5.84650138,  -6.99441   ,
         3.11712916]), np.array([ 3.84581535e-03,  2.63773987e+00,  5.77359467e+00, -1.00396181e+01,
       -4.59486166e+00,  4.49659603e+00, -2.13225103e+00,  3.95830549e+00,
       -4.96569968e+00,  8.50421667e+00,  3.02654883e+00,  2.09753997e-01,
       -6.89575385e+00]), np.array([ 0.01010631,  1.12571405, -2.59338792, -0.17900986,  3.32512964,
       -5.37405822, -2.85897579, -2.21201353,  2.76821721,  0.17751123,
       -1.18360195,  2.97470124,  1.88253437]), np.array([ -0.03832304,   9.30501004,  -0.68463526,  17.95919346,
        22.64897308,  -1.10390407, -12.75129632,  -4.25016015,
       -17.64312805, -18.30170296,   4.34483157,  -1.39027227,
         2.89745888])]
		"""
		


		"""
		### Load Normalized Training Dataset
		# Data directory
		folder  = 'data_scraping/compiled_color_data/'
		color_data_file = 'magic_color_tridata.csv'

		# Load the first file first to initialize an array
		with open(folder + color_data_file) as file_name:
			
			# Store color data
			magic_color_data = np.loadtxt(file_name, delimiter=",")

		# Print loading process is complete
		print("Data Loading process is complete.\nTotal number of pixels loaded: " + str(magic_color_data.shape[0]) + ".\n")

		# Normalize the pixel values
		magic_color_data[:, :-1] = magic_color_data[:, :-1].astype(np.float64)/255
		# Add bias term
		magic_color_data = np.insert(magic_color_data, 0, self.bias, axis = 1)

		# Mix RGB and HSV Color Space [Bias, R, G, B, H, S, V, label]
		#magic_color_data_RGB_HSV = np.zeros((magic_color_data.shape[0],11))
		#magic_color_data_RGB_HSV[:,:4] = magic_color_data[:,:4]
		#magic_color_data_RGB_HSV[:,4:10] = magic_color_data[:,7:13]
		#magic_color_data_RGB_HSV[:, -1] = magic_color_data[:, -1]

		# Sort the color label in an ascending order
		#idc = np.argsort(magic_color_data[:, -1]) 
		#sorted_magic_color_data = magic_color_data[idc]

		# Compressed Data
		#compressed_full_color_data = [tuple(row) for row in sorted_full_color_data]
		#compressed_full_color_data = np.unique(compressed_full_color_data, axis=0)

		# Re-sort the compressed data
		#idc = np.argsort(compressed_full_color_data[:, -1]) 
		#sorted_compressed_full_color_data = compressed_full_color_data[idc]

		# Train dataset
		self.train(magic_color_data)
		#self.train(sorted_compressed_full_color_data)
	"""
	
    	
	# Define sigmoid function
	def sigmoid(self, func_x):
		'''
      		Sigmoid function (Probability Mass Function)
    	'''
		return 1 / (1 + np.exp(-func_x))


  	# Define Logistic Regression Training Model (Multi-classification)
	def train(self, dataset):
		'''
			Logistic Regression Training Model (Multi-classification)
		'''
    	# Extract X_train features and y_train labels
		self.X_train = dataset[:, :-1]
		self.y_train = dataset[:,-1]

    	# Number of samples and features
		num_Samples, num_Features = self.X_train.shape
		
		# List that stores the weights
		self.weights = []
		
		# Array that stores every iteration of loss
		loss_arry = np.zeros(self.max_iters)
		
		# Different Color Classes
		self.color_class = np.unique(self.y_train)
		
		# One vs ALL Binary Classification
		for color in self.color_class:

			# Print which color class has started training
			#print(color)

      		# Binary label whether is the current color or not. (1 is the color, 0 is not)
			binary_label = np.where(self.y_train == color, 1, 0)

    		# Initialize the weight
			weight = np.zeros(num_Features)

			for idx in range(0, self.max_iters):

				# Determine Probability
				y_predicted = self.sigmoid(np.dot(self.X_train, weight))

				# Cross Entropy Loss function
				loss_arry[idx] = 1 / num_Samples * np.sum(-binary_label * np.log(y_predicted) * np.log(1 - y_predicted))

				# Gradient Descend function
				grad_desc = 1 / num_Samples * (np.dot((binary_label - y_predicted), self.X_train))

				# Store previous weight
				prev_weight = copy.deepcopy(weight)

        		# Update weight
				weight += self.lr * grad_desc

				# If less than the error tolerance, break the loop to prevent overtraining/overfitting (early stopping)
				if np.linalg.norm(prev_weight - weight) < self.err_tol:
					#Print number of iterations if it stops early
					#print(idx)
					break

			# Append the trained weight
			self.weights.append(weight)
		
		# Get final weights
		print(self.weights)

	def classify(self,X):
		'''
	    	Classify a set of pixels into red, green, or blue
	    
	    	Inputs:
	      		X: n x 3 matrix of RGB values
	    	Outputs:
	      		y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    	'''
    	################################################################
    	# YOUR CODE AFTER THIS LINE
		# 
		# # Just a random classifier for now
		# # Replace this with your own approach
		 
		# Add bias term
		self.X_test = np.insert(X, 0, self.bias, axis = 1)

    	# Predicted label
		y_predicted = [np.argmax([self.sigmoid(np.dot(x_test, weight)) for weight in self.weights]) for x_test in self.X_test]

		# Predict the outcome
		y = np.rint(np.array([self.color_class[color] for color in y_predicted]))
    
    	# YOUR CODE BEFORE THIS LINE
    	################################################################
		return y
		

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image (RGB)
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is recycling-bin-blue and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		# Get the image pixel information from different color spaces
		
		img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

		# Normalized the pixels
		img_RGB_norm = img_RGB.astype(np.float64)/255
		img_HSV_norm = img_HSV.astype(np.float64)/255
		img_LAB_norm = img_LAB.astype(np.float64)/255
		img_YCrCb_norm = img_YCrCb.astype(np.float64)/255

		# Get the image dimension
		img_height, img_width, _ = img.shape

		# Initialize a mask
		mask_img = np.zeros((img_height,img_width), np.uint8) # Black Pixel = 0, White Pixel = 1

		# Reshape the image from H x W x 3 to (H X W) X 3
		img_RGB_norm = img_RGB_norm.reshape(img_RGB_norm.shape[0]*img_RGB_norm.shape[1],img_RGB_norm.shape[2])
		img_HSV_norm = img_HSV_norm.reshape(img_HSV_norm.shape[0]*img_HSV_norm.shape[1],img_HSV_norm.shape[2])
		img_LAB_norm = img_LAB_norm.reshape(img_LAB_norm.shape[0]*img_LAB_norm.shape[1],img_LAB_norm.shape[2])
		img_YCrCb_norm = img_YCrCb_norm.reshape(img_YCrCb_norm.shape[0]*img_YCrCb_norm.shape[1],img_YCrCb_norm.shape[2])

		# Compile the dataset as X_test (N x 12 features)
		X_test = np.concatenate((img_RGB_norm, img_HSV_norm, img_LAB_norm, img_YCrCb_norm), axis = 1)
		#X_test = img_RGB_norm
		#X_test = np.concatenate((img_RGB_norm, img_LAB_norm, img_YCrCb_norm), axis = 1)

		# Classify each pixel
		y_classified = self.classify(X_test) # Number of Pixels x 1
		y_classified_2D = y_classified.reshape(img_height, img_width) # Height x Width

		# Unmask the pixel that is a recycling-bin-blue
		for height in range(img_height):
			for width in range(img_width):
				if y_classified_2D[height, width] == 1:
					mask_img[height, width] = 1 # Recycling-Blue-Bin is a white pixel

		# Plot mask image
		#plt.imshow(mask_img)
		#plt.show()
		# Replace this with your own approach 
		#mask_img = img
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - mask image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		## Initialize Kernels for Erosion and Dilation process
		kernel_errosion_shape = 4 # Prev 3, Prev 8
		kernel_dilation_shape = 8 # Prev 9
		kernel_errosion = np.ones((kernel_errosion_shape, kernel_errosion_shape), np.uint8)
		kernel_dilation = np.ones((kernel_dilation_shape, kernel_dilation_shape), np.uint8)

		# Erode to filter out noise
		img = cv2.erode(img, kernel_errosion, iterations = 3)
		# Dilate to regain the size of the recycle bin without noise
		img = cv2.dilate(img, kernel_dilation, iterations = 2)

		# Labeled array, where all connected regions are assigned the same integer value.
		label_img = label(img)

		# Return  list of RegionProperties from the label_img
		regions = regionprops(label_img)
		
		# Initialize the box list
		boxes = []

		# Regions props
		for props in regions:

			# Make sure the the size of the recycling bin blue is appropriate
			if  0.55 * img.shape[0] * img.shape[1] > props.area > 0.006 * img.shape[0] * img.shape[1]:

				# Get the bounding box top left and bottom right coordinates
				minr, minc, maxr, maxc = props.bbox

				# Calculate the height and width of the bounding box
				bb_height, bb_width = maxr - minr, maxc - minc

				# Check if the hight-to-width ratio of the recycling-bin-blue makes sense.
				if bb_width * 1.0 < bb_height < bb_width * 2.55:

					# X-coordinates
					#bx = (minc, maxc, maxc, minc, minc)
					# Y-coordinates
					#by = (minr, minr, maxr, maxr, minr)
					# Draw bounding box
					#plt.plot(bx, by, '-r', linewidth=2.5)
					# Add to boxes list
					boxes.append([minc, minr, maxc, maxr])
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes