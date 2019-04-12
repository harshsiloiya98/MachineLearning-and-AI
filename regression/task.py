import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	preprocessed_X = X
	preprocessed_Y = np.array(Y, dtype = float)
	(N, D) = np.shape(X)
	top = X[0]
	avg_map = {}
	stdev_map = {}
	onehot_map = {}
	for i in range(D):
		if (i > 0):
			col = X[:, i]
			if (type(top[i]) is not str):
				avg_map[i] = np.mean(col)
				stdev_map[i] = np.std(col)
			else:
				labels = np.unique(col)
				onehot_map[i] = one_hot_encode(col, labels)
	for i in range(N):
		for j in range(D):
			if (j == 0):
				preprocessed_X[i][j] = 1.
			elif (type(X[i][j]) is not str):
				preprocessed_X[i][j] = (X[i][j] - avg_map[j]) / stdev_map[j]
			else:
				preprocessed_X[i][j] = onehot_map[j][i]
	tmp = np.hstack(preprocessed_X.flat)
	preprocessed_X = np.reshape(tmp, (N, tmp.size // N))
	return preprocessed_X, preprocessed_Y

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	# returns [D x 1] gradient vector
	error = X.T @ (Y - X @ W)
	gradient = _lambda * W
	gradient = gradient - error
	return 2 * gradient

def ridge_grad_descent(X, Y, _lambda, max_iter = 10000, lr = 0.00017, epsilon = 1e-2):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	(N, D) = X.shape
	#W = np.ones((D, 1), dtype = float)
	#W = np.random.randn(D, 1)
	W = np.linalg.inv(X.T @ X + _lambda * np.eye(X.shape[1])) @ X.T @ Y
	while (max_iter > 0):
		gradient = grad_ridge(W, X, Y, _lambda)
		W += lr * gradient
		if (np.linalg.norm(gradient, ord = 2) <= epsilon):
			break
		max_iter -= 1
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	m = len(lambdas)
	splitted_X = np.vsplit(X, k)
	splitted_Y = np.vsplit(Y, k)
	sse_list = []
	for i in range(m):
		sse_temp = []
		_lambda = lambdas[i]
		for j in range(k):
			X_test = splitted_X[j]
			Y_test = splitted_Y[j]
			splitted_X.pop(j)
			splitted_Y.pop(j)
			X_train = np.vstack(splitted_X)
			Y_train = np.vstack(splitted_Y)
			weights = algo(X_train, Y_train, _lambda)
			sse_v = sse(X_test, Y_test, weights)
			sse_temp.append(sse_v)
			splitted_X.insert(j, X_test)
			splitted_Y.insert(j, Y_test)
		sse_list.append(np.mean(sse_temp))
	return sse_list

def coord_grad_descent(X, Y, _lambda, max_iter = 150):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Lasso Loss Function 
	'''
	(N, D) = X.shape
	W = np.ones((D, 1), dtype = float)
	l1_factor = _lambda / 2
	denominators = np.sum(np.square(X), axis = 0)
	while (max_iter > 0):
		W_prev = W
		for j in range(D):
			X_j = X[:, j]
			denominator_term = denominators[j]
			t1 = X_j.T @ (Y - X @ W)
			t2 = W[j] * denominator_term
			numerator_term = t1[0] + t2[0]
			if (numerator_term > l1_factor):
				numerator_term -= l1_factor
				W[j] = numerator_term / denominator_term
			elif (numerator_term < -1 * l1_factor):
				numerator_term += l1_factor
				W[j] = numerator_term / denominator_term
			else:
				W[j] = 0.
		max_iter -= 1
	return W

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)

	# add code below to test for kfoldcv
	#lambdas = [1, 2000, 4000, 6000, 8000, 10000] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	#scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	#W = ridge_grad_descent(trainX, trainY, 10000)
	#W = coord_grad_descent(trainX, trainY, 12000)
	#print(sse(testX, testX, W))
	#scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	#plot_kfold(lambdas, scores)