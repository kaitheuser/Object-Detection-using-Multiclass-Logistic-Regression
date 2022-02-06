'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

from audioop import bias
import numpy as np
import cv2
from skimage.measure import label, regionprops
import os
import copy
"""Comment the library below for autograder"""
#from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Qt5Agg')

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
		"""
		# Weights from Class 1, 12, 14, 15 in order 
		#self.color_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

		# Weights from Class 1 to Class 15 in order
		
		self.weights = [np.array([-1.43237565, 									# Bias Weight
									0.39313288, -3.17120339, 2.90329306,  		# RGB Weights
									4.94740535, -1.50282427, 0.28830888,  		# HSV Weights
									-1.90860623,  1.91560939, -4.00319174,		# LAB Weights
									-1.40905774,  0.57831505,  1.72678019]),	# YCrCb Weights
						np.array([-3.09757688, 
									-1.7917831 , 2.11053658, 3.13122623,
									0.47113756, 1.56781302, 1.88648985, 
									1.16396869, -2.88980175, -2.47996631,
									1.05929184, -3.59185286, -0.39027121]), 
						np.array([-1.24808634,
									-0.40655387, -0.46025804, -0.62954517,
									-0.44331041, -0.55198432, -0.68073272,
									-0.50696866, -0.66788934, -0.53122442,
       								-0.46347881, -0.5858138 , -0.72014553]),
						np.array([-1.38228239,
									-0.5160588 , -0.58478363, -0.75368975,
									-0.49486443, -0.54078045, -0.80372253,
									-0.63235822, -0.72853822, -0.59789155,
       								-0.5834381 , -0.64569741, -0.7899122 ]), 
						np.array([-1.29649602,
									-0.50922535, -0.56550942, -0.72070245, 
									-0.46190114, -0.49134405, -0.77076844, 
									-0.61318135, -0.68491458, -0.5645943 ,
       								-0.56632222, -0.61000472, -0.73787705]), 
						np.array([-1.15359345, 
									-0.46019419, -0.48327103, -0.56440328, 
									-0.40074914, -0.38848153, -0.61529322, 
									-0.52038173, -0.59939394, -0.53332057,
       								-0.48557314, -0.56090916, -0.62352389]), 
						np.array([-1.11831114,
									-0.40863372, -0.41208013, -0.55067717, 
									-0.39624735, -0.43611243, -0.60217686,
									-0.4597803 , -0.60920123, -0.48732679,
       								-0.42682313, -0.54837229, -0.63120326]), 
						np.array([-1.20636563, 
									-0.28603288, -0.61455773, -0.79594507, 
									-0.35666301, -0.33856654, -0.61570991,
									-0.54031949, -0.56361273, -0.43170268,
									-0.53690541, -0.42652274, -0.75171721]), 
						np.array([-0.80855115,  
									0.62296732, -0.29964625, -1.3592752 , 
									-1.53744674, -0.63109072, -0.52238834, 
									-0.19714706, -0.3189606 ,  0.29826633,
       								-0.14437873,  0.14193703, -1.09180211]), 
						np.array([-1.48693083,
									1.17701146,  0.37875328, -1.03468078,
									-2.02887633, -0.79451748, -0.08768561,
									0.34468084, -0.80203371,  0.08807505,
        							0.45658277, -0.23213268, -1.58799819]), 
						np.array([-1.26407228, 
									-0.35610304, -0.47056471, -0.81085843, 
									-0.53401977, -0.36072565, -0.66785182, 
									-0.50963975, -0.66992003, -0.4569232 ,
       								-0.47510175, -0.54951733, -0.82395793]), 
						np.array([-0.56349612, 
									-0.28095928,  0.17718288, -1.08336012, 
									-0.73721287, -0.36916126, -0.72321126, 
									-0.03898971, -0.7862093 ,  0.34035694,
       								-0.10307183, -0.41118156, -0.83541228]), 
						np.array([ 0.14899571,  
									1.09763579,  0.55201975, -1.08109452, 
									-1.18516148, -4.42646887, -1.70174256, 
									0.48737518, -0.28081294,  0.9901729 ,
       		 						0.52733485,  0.48115393, -0.83447774]), 
						np.array([ 0.4323592 , 
									-1.04775352, -1.14378416, -1.79882384,  
									0.13155267, -0.8766743 , -2.01843089, 
									-1.37062024, -0.00597533,  0.54695785,
       								-1.19036678,  0.31868721, -0.12725946]), 
						np.array([-1.82686676,  
									0.82806913,  0.54239467, -0.18965986, 
									-1.34549785, -1.52786526, -0.10694402,  
									0.42006767, -1.03301664, -0.50751873,
        							0.54447405, -0.71486512, -1.33126621])]
		"""
		
		
		# Weights from Class 1, 12, 14, 15 in order 
		self.color_class = [1, 12, 14, 15]

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
		#magic_color_data = np.insert(magic_color_data, 0, self.bias, axis = 1)

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

		# Add bias term #
		self.X_train = np.insert(self.X_train, 0, self.bias, axis = 1)

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

					# Add to boxes list
					boxes.append([minc, minr, maxc, maxr])
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes

	### Comment for Autograder ###
	"""
	def draw_bounding_boxes(self, mask_img, boxes, rgb_img):
		'''
		Draw bounding boxes and display the image.
			
		Inputs:
			rgb_img - rgb image
			boxes   - bounding boxes coordinates (top left and bottom right)
			mask_img - mask image
		Outputs:
			None
		'''
		# Get the image shape
		img_height, img_width, _ = rgb_img.shape

		# Initialize a mask
		mask_img_3D = np.zeros((img_height,img_width, 3)) # Black Pixel = 0, White Pixel = 1

		# Unmask the pixel that is a recycling-bin-blue in black and white
		for height in range(img_height):
			for width in range(img_width):
				if mask_img[height, width] == 1:
					mask_img_3D[height, width,:] = 1
            
		# Plot subplots
		fig, ax = plt.subplots(1, 2, figsize = (8, 8))
		
		# Plot mask image without labels
		ax[0].imshow(mask_img_3D)
		ax[0].set_yticklabels([])
		ax[0].set_xticklabels([])
		ax[0].set_xticks([])
		ax[0].set_yticks([])

		# Plot RGB image with bounding boxes
		ax[1].imshow(rgb_img)
		for box in boxes:
			bx = (box[0], box[2], box[2], box[0], box[0])
			by = (box[1], box[1], box[3], box[3], box[1])
			ax[1].plot(bx, by, '-r', linewidth=2.5)
		ax[1].set_yticklabels([])		
		ax[1].set_xticklabels([])
		ax[1].set_xticks([])
		ax[1].set_yticks([])

		# Show plot in tight layout
		plt.tight_layout()
		plt.show()
	"""