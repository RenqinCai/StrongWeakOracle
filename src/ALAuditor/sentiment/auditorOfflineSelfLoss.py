from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads
from numpy import linalg as LA

def sigmoid(x):
	# return 1/(np.exp(-x)+1)
	return 0.5*(np.tanh(x)+1)

def logistic_pred(weights, inputs):
	
	outputs1 = np.matmul(inputs, weights)
	# print(outputs1)
	# outputs = np.matmul(outputs, np.transpose(inputs))
	# print("outputs shape", outputs.shape)
	# outputs2 = np.matmul(inputs, weights[3:])
	# outputs = outputs1*outputs2
	# outputs = np.sum(outputs[3:]inputs, axis=1)
	# print(outputs.shape)
	return sigmoid(outputs1)
	# return sigmoid(np.dot(inputs, weights))

def training_loss(weights, inputs, targets):
	preds = logistic_pred(weights, inputs)

	label_prob = preds*targets+(1-preds)*(1-targets)
	weightParam = 0.00

	return -np.sum(np.log(label_prob))+weightParam*np.sum(np.power(weights, 2))

# inputs = np.array([[0.52, 1.12,  0.77],
#                    [0.88, -1.08, 0.15],
#                    [0.52, 0.06, -1.30],
#                    [0.74, -2.49, 1.39]])
# targets = np.array([True, True, False, True])

def readFeatureLabel(featureLabelFile):
	f = open(featureLabelFile)

	featureMatrix = []
	labelList = []

	for rawLine in f:
		line = rawLine.strip().split("\t")

		lineLen = len(line)

		featureSample = []
		for lineIndex in range(lineLen-1):
			featureVal = float(line[lineIndex])
			featureSample.append(featureVal)

		labelList.append(float(line[lineLen-1]))

		featureMatrix.append(featureSample)

	f.close()

	return featureMatrix, labelList


def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	transferLabelList = []
	targetLabelList = []
	auditorLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		# print(float(line[1]))
		auditorLabelList.append(float(line[0]))
		transferLabelList.append(float(line[1]))
		targetLabelList.append(float(line[2]))

	f.close()

	return auditorLabelList, transferLabelList, targetLabelList


dataName = "electronics"

featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName

featureMatrix, labelList = readFeatureLabel(featureLabelFile)
featureMatrix = np.array(featureMatrix)
labelArray = np.array(labelList)

transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)
transferLabelArray = np.array(transferLabelList)
auditorLabelArray = np.array(auditorLabelList)

inputs = featureMatrix

targets = auditorLabelArray

weights = 0.1*np.ones(len(inputs[0]))

training_gradient_fun = grad(training_loss)

print("initial loss:", training_loss(weights, inputs, targets))

learningRate = 0.1
for i in range(100):
	weights -= training_gradient_fun(weights, inputs, targets)*learningRate

	# print("train loss", training_loss(weights, inputs, targets))
print("final weight", weights)
print("final pred", logistic_pred(weights, inputs))