import matplotlib.pyplot as mpl
import numpy as np
from random import randint
from ForwardProp import CostFunc,ForwardProp
from scipy.optimize import fmin_cg
import numpy as np
import scipy as sp
from scipy.io import loadmat
import pandas
import numpy.random as npr
from ForwardProp import ForwardProp, CostFunc
import time
from BackProp import BackProp
from GradDescent import GradientDescent
from sklearn.model_selection import train_test_split
def LearningCurve(ThetaVect,X,y,input_layer_size,hidden_layer1_size,hidden_layer2_size,classification_nodes,iterations = 1000):
	testErrorSet = []
	CVErrorSet = []
	ThetaVect = ThetaVect[np.newaxis]
	X,cvSet,y,ySetCV = train_test_split(X,y,test_size = .5)
	print(X.shape)
	L = [0,1,2,3,4,5,6,7,8,9,10]
	for i in range(len(L)):
		TestThetaVect = np.copy(ThetaVect)
		args=(X,y,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes,L[i])
		grad = fmin_cg(CostFunc,TestThetaVect,fprime=BackProp,args=args,maxiter = iterations)
		testErrorSet.append(CostFunc(grad, X, y, input_layer_size, hidden_layer1_size, hidden_layer2_size, classification_nodes,0))
		CVErrorSet.append(CostFunc(grad, cvSet , ySetCV, input_layer_size, hidden_layer1_size, hidden_layer2_size, classification_nodes,0))
		print("Error of Current Gradient %.8f" % (CostFunc(grad, X, y, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes, 0)))
	
	mpl.plot(range(0,len(L)),testErrorSet)
	mpl.plot(range(0,len(L)),CVErrorSet)
	mpl.show()
def RandInit(cols,rows,eps):
	return npr.rand(cols,rows) * 2 * eps - eps
def logicalVectorize(y):
	newY = np.zeros([y.shape[0],10])
	for i in range(y.shape[0]):
		newY[i][y[i]] = 1
	return newY
if __name__ == '__main__':
	Location = 'train.csv'
	df = pandas.read_csv(Location)
	hidden_layer_nodes = 400
	hidden_layer_2_nodes = 400
	classification_nodes = 10
	y = np.asarray(df['label'].values)
	X = df.values
	X = np.asarray(np.delete(X, 0 ,axis = 1))
	X = ((X) * ((1 + 1)/255)) - 1 
	y = logicalVectorize(y)
	X,Xinp,y,yinp = train_test_split(X,y,test_size=.022)
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
	
	LearningCurve(ThetaVect, Xinp, yinp, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes)
	"""50/50= 1.506 2.605 100/100 1,0499 2.2086"""