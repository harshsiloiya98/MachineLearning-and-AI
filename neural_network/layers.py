import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0, 0.1, (in_nodes, out_nodes))	
		self.biases = np.random.normal(0, 0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		#raise NotImplementedError
		self.data = np.matmul(X, self.weights)
		self.data += self.biases
		return sigmoid(self.data)
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		#raise NotImplementedError
		dZ = delta * derivative_sigmoid(self.data)
		dW = np.matmul(np.transpose(activation_prev), dZ)
		dB = np.sum(dZ)
		new_delta = np.matmul(dZ, np.transpose(self.weights))
		self.weights -= lr * dW
		self.biases -= lr * dB
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0, 0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0, 0.1, self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		#raise NotImplementedError
		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for i in range(n):
			for j in range(self.out_depth):
				input_matrix = X[i]
				filter_matrix = self.weights[j]
				bias = self.biases[j]
				for row in range(self.out_row):
					for col in range(self.out_col):
						x1 = row * self.stride
						x2 = x1 + self.filter_row
						y1 = col * self.stride
						y2 = y1 + self.filter_col
						input_receptive = input_matrix[:, x1:x2, y1:y2]
						conv_res = np.sum(np.multiply(input_receptive, filter_matrix))
						self.data[i, j, row, col] = conv_res + bias
		return sigmoid(self.data)
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		#raise NotImplementedError
		dZ = delta * derivative_sigmoid(self.data)
		(_, depth, rows, cols) = dZ.shape
		new_delta = np.zeros(activation_prev.shape)
		dW = np.zeros(self.weights.shape)
		dB = np.zeros(self.biases.shape)
		depth = activation_prev.shape[1]
		for i in range(n):
			for j in range(depth):
				input_matrix = activation_prev[i, j]
				filter_matrix = self.weights[j]
				for row in range(rows):
					for col in range(cols):
						x1 = row * self.stride
						x2 = x1 + self.filter_row
						y1 = col * self.stride
						y2 = y1 + self.filter_col
						input_receptive = input_matrix[x1:x2, y1:y2]
						new_delta[:, :, x1:x2, y1:y2] += filter_matrix * dZ[i, j, row, col]
						dW[:, j, :, :] += input_receptive * dZ[i, j, row, col]
						dB[j] += dZ[i, j, row, col]
		self.weights -= lr * dW
		self.biases -= lr * dB
		return new_delta
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		# Stores the outgoing summation of weights * feautres 
		self.data = None

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		#raise NotImplementedError
		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for i in range(n):
			for j in range(self.out_depth):
				input_matrix = X[i, j]
				for row in range(self.out_row):
					for col in range(self.out_col):
						x1 = row * self.stride
						x2 = x1 + self.filter_row
						y1 = col * self.stride
						y2 = y1 + self.filter_col
						input_window = input_matrix[x1:x2, y1:y2]
						self.data[i, j, row, col] = np.mean(input_window)
		return self.data
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		#raise NotImplementedError
		(_, depth, rows, cols) = delta.shape
		new_delta = np.zeros(activation_prev.shape)
		for i in range(n):
			for j in range(depth):
				input_matrix = activation_prev[i, j]
				for row in range(rows):
					for col in range(cols):
						x1 = row
						x2 = x1 + self.filter_row
						y1 = col
						y2 = y1 + self.filter_col
						input_window = input_matrix[x1:x2, y1:y2]
						temp = np.ones((self.filter_row, self.filter_col)) * delta[i, j, row, col] / (self.filter_row * self.filter_col)
						new_delta[i, j, x1:x2, y1:y2] += temp
		return new_delta
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
