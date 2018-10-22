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
import numpy.linalg as npl
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
def ValidationCurve(X,y,input_layer_size,hidden_layer_nodes,hidden_layer2_nodes,classification_nodes,iterations = 100):
	testErrorSet = []
	CVErrorSet = []
	testAcc=[]
	cvAcc=[]

	X,X_test,y,y_test = train_test_split(X,y,test_size = .2)
	print(X.shape)
	L = np.arange(0,11,1)
	print(L)
	#time.sleep(5)
	for i in range(len(L)):
		reg_param = float(L[i]/(X.shape[0]))
		print(reg_param)
		model = keras.Sequential()
		model.add(keras.layers.Dense(hidden_layer_nodes,activation="relu",kernel_regularizer=keras.regularizers.l2(reg_param),use_bias = True,input_shape=(784,)))
		model.add(keras.layers.Dropout(.5))
		model.add(keras.layers.Dense(hidden_layer_2_nodes//1.5,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
		model.add(keras.layers.Dropout(.5))
		model.add(keras.layers.Dense(hidden_layer_2_nodes//2,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
		model.add(keras.layers.Dropout(.5))
		model.add(keras.layers.Dense(hidden_layer_2_nodes//2,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
		model.add(keras.layers.Dropout(.5))
		model.add(keras.layers.Dense(hidden_layer_2_nodes//4,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
		model.add(keras.layers.Dropout(.5))
		model.add(keras.layers.Dense(hidden_layer_2_nodes//8,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
		model.add(keras.layers.Dense(10,kernel_regularizer = keras.regularizers.l2(reg_param),activation="softmax", use_bias = True))
		model.compile(optimizer=keras.optimizers.Adam(.0001),loss="categorical_crossentropy",metrics=["accuracy"])
		model.fit(X,y,epochs=5000,batch_size=X.shape[0],validation_data=(X_test,y_test))
		testEval= model.evaluate(X,y,batch_size=X.shape[0])
		cvEval = model.evaluate(X_test,y_test,batch_size=X_test.shape[0])
		testErrorSet.append(testEval[0])
		CVErrorSet.append(cvEval[0])
		testAcc.append(testEval[1])
		cvAcc.append(cvEval[1])
	print(model.evaluate(X,y,batch_size=X.shape[0]))
	print("L values: " + str(L))
	print("Test Errors for respective L:" + str(testErrorSet))
	print("CV Errors for respective L: " + str(CVErrorSet))
	mpl.plot(L,testErrorSet)
	mpl.plot(L,CVErrorSet)
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
	hidden_layer_nodes = 700
	hidden_layer_2_nodes = 700
	classification_nodes = 10
	y = np.asarray(df['label'].values)
	X = np.asarray(df.drop(columns=['label']).values)
	X = ((X) * ((1 + 1)/255)) - 1
	y = logicalVectorize(y)
	X,Xinp,y,yinp = train_test_split(X,y,test_size=.1)
	ValidationCurve( Xinp, yinp, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes)
	"""50/50= 1.506 2.605 100/100 1,0499 2.2086"""
