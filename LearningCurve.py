import matplotlib.pyplot as mpl
import numpy as np
from random import randint
from ForwardProp import CostFunc
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
def LearningCurve(ThetaVect,X,y,input_layer_size,hidden_layer1_size,hidden_layer2_size,classification_nodes,L = 0,max_m = 20,iterations = 50):
	Tset = np.empty((0,X.shape[1]),int)
	CVset = np.empty((0,X.shape[1]),int)
	ySetTest = np.empty((0,y.shape[1]),int)
	ySetCV = np.empty((0,y.shape[1]),int)
	testErrorSet = []
	CVErrorSet = []
	ThetaVect = ThetaVect[np.newaxis]
	for _ in range(max_m):
		TCol = randint(0,X.shape[0])
		CVCol = randint(0,X.shape[0])
		CVset = np.append(CVset, X[CVCol][np.newaxis],axis=0)
		ySetCV = np.append(ySetCV,y[CVCol][np.newaxis],axis=0)
		Tset = np.append(Tset,X[TCol][np.newaxis],axis=0)
		ySetTest = np.append(ySetTest,y[TCol][np.newaxis],axis=0)
		TestThetaVect = np.copy(ThetaVect)
		args=(Tset,ySetTest,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes,L)
		grad = fmin_cg(CostFunc,TestThetaVect,fprime=BackProp,args=args)
		testErrorSet.append(CostFunc(grad, Tset, ySetTest, input_layer_size, hidden_layer1_size, hidden_layer2_size, classification_nodes))
		CVErrorSet.append(CostFunc(grad, CVset, ySetCV, input_layer_size, hidden_layer1_size, hidden_layer2_size, classification_nodes))
		print("Error of Current Gradient %.8f" % (CostFunc(grad, X, y, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes, 0)))
	print(testErrorSet[len(testErrorSet)-1])
	print(CVErrorSet[len(CVErrorSet)-1])
	mpl.plot(range(1,max_m+1),testErrorSet)
	mpl.plot(range(1,max_m+1),CVErrorSet)
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
	hidden_layer_nodes = 200
	hidden_layer_2_nodes = 200
	classification_nodes = 10
	y = np.asarray(df['label'].values)
	X = df.values
	X = np.asarray(np.delete(X, 0 ,axis = 1))
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
	y = logicalVectorize(y)
	LearningCurve(ThetaVect, X, y, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes,L=1)
	"""50/50= 1.506 2.605 100/100 1,0499 2.2086"""