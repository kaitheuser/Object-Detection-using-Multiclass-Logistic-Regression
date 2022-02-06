'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
from pixel_classification.generate_rgb_data import read_pixels
import copy
import time

class PixelClassifier():

  def __init__(self, lr = 0.1, max_iters = 5000, err_tol = 1e-3, bias = 1):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    ### Logistic Regression Model Parameters
    self.lr = lr                      # Learning Rate
    self.max_iters = max_iters        # Maximum Number of Iterations
    self.err_tol = err_tol            # Error that stops the training process
    self.weights = None               # Weight, w for the function of x (linear model)
    self.bias = bias                  # Bias, b for the function of x (linear model)
    self.color_class = None           # Class of Interest to Train

    
    ### For autograder
    # Weight and Color Class Parameters
    
    self.weights = [np.array([-1.41444572,  8.36556181, -4.40847181, -4.23174115]), 
                    np.array([-1.15572251, -4.80800263,  8.14250506, -4.27653186]), 
                    np.array([-1.17755687, -4.80183284, -4.29106499,  8.09508523])]
    self.color_class = np.array([1, 2, 3])
    

    """
    ### Load Normalized Training Dataset
    # Data directory
    folder  = 'pixel_classification/data/training'

    # Training images (N x 3)
    X_red_train = read_pixels(folder + '/red')
    X_green_train = read_pixels(folder + '/green')
    X_blue_train = read_pixels(folder + '/blue')
    # Combine the training images (3N x 3)
    X_train = np.concatenate((X_red_train, X_green_train, X_blue_train))
  
    # Training labels (N x 1)
    y_red_train = np.full(X_red_train.shape[0],1)
    y_green_train = np.full(X_green_train.shape[0], 2)
    y_blue_train = np.full(X_blue_train.shape[0],3)
    # Combine the training labels (3N x 1)
    y_train = np.concatenate((y_red_train, y_green_train, y_blue_train)).reshape(-1, 1)

    # Compiled Dataset
    dataset = np.concatenate((X_train, y_train), axis = 1)

    # Train dataset
    self.train(dataset)
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
          #sprint(idx)
          break

      # Append the trained weight
      self.weights.append(weight)


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
    
    # Just a random classifier for now
    # Replace this with your own approach

    # Add bias term
    self.X_test = np.insert(X, 0, self.bias, axis = 1)

    # Predicted label
    y_predicted = [np.argmax([self.sigmoid(np.dot(x_test, weight)) for weight in self.weights]) for x_test in self.X_test]

    # Predict the outcome
    y = np.rint(np.array([self.color_class[color] for color in y_predicted]))
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

