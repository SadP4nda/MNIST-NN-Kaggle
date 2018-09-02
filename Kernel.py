import numpy as np
import scipy as sp
from scipy.optimize import fmin_cg, fmin_l_bfgs_b,fmin_bfgs
import pandas
import numpy.random as npr
from ForwardProp import ForwardProp, CostFunc
import time
from BackProp import BackProp
from GradDescent import GradientDescent
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
def RandInit(cols,rows,eps):
	return npr.rand(cols,rows) * 2 * eps - eps
def logicalVectorize(y):
	newY = np.zeros([y.shape[0],10])
	for i in range(y.shape[0]):
		newY[i][y[i]] = 1
	return newY
hidden_layer_nodes = 300
hidden_layer_2_nodes = 300
classification_nodes = 10
if __name__ == '__main__':
	Location = 'train.csv'
	df = pandas.read_csv(Location)

	y = np.asarray(df['label'].values)
	X = df.values
	X = np.delete(X, 0 ,axis = 1)
	print(X)
	X = ((X) * ((1 + 1)/255)) - 1 
	y = logicalVectorize(y)
	X,X_test,y,y_test = train_test_split(X,y)
	print(X)
	print(X_test.shape)
	# arrs = loadmat('ex4data1.mat')
	# y = arrs['y']
	# for i in range(y.size):
	# 	if y[i] == 10:
	# 		y[i] = 0
	# X = arrs['X']
	eps1 = np.sqrt(6)/np.sqrt((hidden_layer_nodes+(X.shape[1]+1)))
	eps2 = np.sqrt(6)/np.sqrt((classification_nodes + (hidden_layer_nodes+1)))
	eps3 = np.sqrt(6)/np.sqrt(((hidden_layer_nodes+1) + hidden_layer_2_nodes))
	Theta1 = RandInit(hidden_layer_nodes,X.shape[1]+1,eps1).reshape(hidden_layer_nodes*(X.shape[1]+1),order='F')
	Theta2 = RandInit(hidden_layer_2_nodes,hidden_layer_nodes+1,eps1).reshape(hidden_layer_2_nodes*(hidden_layer_nodes+1),order='F')
	Theta3 = RandInit(classification_nodes,hidden_layer_nodes+1,eps2).reshape(classification_nodes*(hidden_layer_nodes+1),order='F')
	ThetaVect = np.append(Theta1,Theta2)
	ThetaVect = np.append(ThetaVect,Theta3)
	print(ThetaVect)
	print(y)
	yLabels = np.argmax(y,axis=1)
	args=(X,y,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes,2.5)
	a = time.time()
	grad = fmin_cg(CostFunc,ThetaVect,fprime=BackProp,args=args, maxiter = 1000)
	b = time.time()
	print("Time for Conjugate Gradient Optimization %.6f" % (b-a))
	print(y)
	pred = ForwardProp(grad,X,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes)
	print("Error of Current Gradient %.8f" % (CostFunc(grad, X, y, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes, 0)))
	print(pred)
	print(np.argmax(y,axis=1))
	print(np.argmax(pred,axis=1))
	print(pred[1])
	pred_numbers = np.argmax(pred,axis=1)
	print(np.sum(pred_numbers == yLabels))
	print("Accuracy of predictions on training set %.12f" % (np.sum(pred_numbers == yLabels)/yLabels.shape[0]))
	predTest = ForwardProp(grad,X_test,X_test.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes)
	predTest_numbers = np.argmax(predTest,axis=1)
	yTestLabels = np.argmax(y_test,axis=1)
	print("Accuracy of prediction on test set %.12f" % (np.sum(predTest_numbers == yTestLabels)/yTestLabels.shape[0]))
	Theta1 = grad[0:(1+X.shape[1])*hidden_layer_nodes].reshape((hidden_layer_nodes,X.shape[1]+1),order='F')
	Theta2 = grad[(1+X.shape[1])*hidden_layer_2_nodes:(1+X.shape[1])*hidden_layer_nodes+(1+hidden_layer_nodes)*hidden_layer_2_nodes].reshape(\
		(hidden_layer_2_nodes,hidden_layer_nodes+1),order='F')
	Theta3 = grad[(1+X.shape[1])*hidden_layer_nodes+(1+hidden_layer_nodes)*hidden_layer_2_nodes:].reshape((classification_nodes,hidden_layer_2_nodes+1),order='F')
	T1CSV = pandas.DataFrame(Theta1)
	T1CSV.to_csv("D:\\Coding\\Kaggle\\MNIST\\Theta1.csv")
	T2CSV = pandas.DataFrame(Theta2)
	T2CSV.to_csv("D:\\Coding\\Kaggle\\MNIST\\Theta2.csv")
	T3CSV = pandas.DataFrame(Theta3)
	T3CSV.to_csv("D:\\Coding\\Kaggle\\MNIST\\Theta3.csv")