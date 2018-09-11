"""
use a new model to train the auditor
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

from datetime import datetime

dataName = "electronics"

modelName = "auditor_offline_"+dataName
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp

# def sigmoid(x):
#   	 return (1 / (1 + np.exp(-x)))

def get_name_features(names):

		name = []
		for i in names:
			s = re.findall('(?i)[a-z]{2,}',i)
			name.append(' '.join(s))

		cv = CV(analyzer='char_wb', ngram_range=(3,4))
		fn = cv.fit_transform(name).toarray()

		return fn

def sigmoid(x):
		return 0.5*(np.tanh(x)+1)

def logistic_pred(weights, weakWeights, inputs):

	weightNum = len(weights)
	outputs1 = np.matmul(inputs, weights)
	# print("outputs step1", outputs1)
	outputs = outputs1
	# outputs2 = np.matmul(inputs, weakWeights)
	# outputs = outputs1*outputs2
	# outputs = np.matmul(outputs, np.transpose(inputs))
	# print("outputs shape", outputs.shape)
	# outputs = np.sum(outputs*inputs, axis=1)

	# for labelProbIndex in range(len(outputs)):
	# 	labelProbVal = outputs[labelProbIndex]
		# if labelProbVal == 0.0:
		# 	print("output zero err", labelProbIndex, labelProbVal)

	# print("outputs", outputs)
	# print(outputs.shape)
	return sigmoid(outputs)

def training_loss(weights, weakWeights, inputs, targets):
	preds = logistic_pred(weights, weakWeights, inputs)
	# for labelProbIndex in range(len(preds)):
	# 	labelProbVal = preds[labelProbIndex]
	# 	if labelProbVal == 0.0:
	# 		print("preds zero err", labelProbIndex, labelProbVal)

	# print("training loss step 1", preds)
	label_prob = preds*targets+(1-preds)*(1-targets)
	# print("training loss step 2", label_prob)

	# for labelProbIndex in range(len(label_prob)):
	# 	labelProbVal = label_prob[labelProbIndex]
		# label_prob[labelProbIndex] = label_prob[labelProbIndex]+1e-30
		# if labelProbVal == 0.0:
		# 	print("labelProb zero err", labelProbIndex, labelProbVal, preds[labelProbIndex], targets[labelProbIndex])

	weightParam = 0.0
	loss = -np.sum(np.log(label_prob))+weightParam*np.sum(np.power(weights, 2))
	# print("training loss step 3", loss)
	return loss

class active_learning:

	def __init__(self, fold, rounds, fn, transferLabel, label):

		self.fold = fold
		self.rounds = rounds

		self.fn = fn
		# self.fn = np.hstack((fn, np.ones((len(fn), 1))))
		self.label = label
		self.transferLabel = transferLabel

		self.tao = 0
		self.alpha_ = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.m_auditorWeight = []

		self.ex_id = dd(list)
		self.m_weakOracleWeight = []

	def select_example(self, unlabeled_list):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)
		# print("---------------")
		alpha = 0.1
		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			labelPredictProb = self.m_clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]

			sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)

			maxLabelPredictProb = sortedLabelPredictProb[0]
			subMaxLabelPredictProb = sortedLabelPredictProb[1]

			idScore = maxLabelPredictProb-subMaxLabelPredictProb
			idScore = -idScore

			unlabeledIdScoreMap[unlabeledId] = idScore
		
		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		# print(len(label_train), sum(label_train), "ratio", sum(label_train)*1.0/len(label_train))
		# print("label_train", label_train)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

	def initAuditor(self, featureDim):
		# self.m_auditorWeight = np.random.uniform(-0.1, 0.1, size=[featureDim, featureDim])
		# self.m_auditorWeight = 0.01*np.ones((featureDim, featureDim))
		self.m_auditorWeight = np.random.uniform(-0.1, 0.1, size=featureDim)

		weakOracle = LR(random_state=3)
		weakOracle.fit(self.fn, self.transferLabel)

		self.m_weakOracleWeight = weakOracle.coef_[0]


	def trainAuditor(self, fn_train, label_train):
		training_gradient_fun = grad(training_loss)

		learningRate = 1e-10
		epoch = 200
		for i in range(epoch):
			self.m_auditorWeight -= training_gradient_fun(self.m_auditorWeight, self.m_weakOracleWeight, fn_train, label_train)

			print("weights", self.m_auditorWeight)
			# label_prob[labelProbIndex] = label_prob[labelProbIndex]+1e-30

	def predAuditor(self, fn_test, label_test):
		pred_test = logistic_pred(self.m_auditorWeight, self.m_weakOracleWeight, fn_test)
		pred_test = pred_test > 0.5
		print("pred test", pred_test)
		acc = accuracy_score(label_test, pred_test)

		return acc

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		print("featureNum", len(self.fn[0]))
		# print("non zero feature num", sum(self.fn[0]))

		totalTransferNumList = []
		np.random.seed(3)
		np.random.shuffle(indexList)

		foldNum = 10
		foldInstanceNum = int(totalInstanceNum*1.0/foldNum)
		foldInstanceList = []

		for foldIndex in range(foldNum-1):
			foldIndexInstanceList = indexList[foldIndex*foldInstanceNum:(foldIndex+1)*foldInstanceNum]
			foldInstanceList.append(foldIndexInstanceList)

		foldIndexInstanceList = indexList[foldInstanceNum*(foldNum-1):]
		foldInstanceList.append(foldIndexInstanceList)
		# kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		cvIter = 0
		# random.seed(3)
		totalAccList = [0 for i in range(10)]

		posRatioList = []

		for foldIndex in range(foldNum):
			
			# self.clf = LinearSVC(random_state=3)

			# self.m_clf = LR(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])


			# trainNum = int(totalInstanceNum*0.2)
			# print("trainNum", trainNum)
			
			fn_test = self.fn[test]
			label_test = self.label[test]

			fn_train = self.fn[train]
			label_train = self.label[train]

			testOneNum = np.sum(label_test)
			testNum = len(fn_test)

			posRatio = testOneNum*1.0/testNum
			posRatioList.append(posRatio)

			self.initAuditor(len(fn_train[0]))

			self.trainAuditor(fn_train, label_train)
			acc = self.predAuditor(fn_test, label_test)
			# label_preds = self.m_clf.predict(fn_test)
			# acc = accuracy_score(label_test, label_preds)

			totalAccList[cvIter] = acc

			cvIter += 1      
		
		print("posRatioList", posRatioList, np.mean(posRatioList), np.sqrt(np.var(posRatioList)))

		print("totalAccList", totalAccList, np.mean(totalAccList), np.sqrt(np.var(totalAccList)))

		totalACCFile = modelVersion+".txt"
		f = open(totalACCFile, "w")
		for i in range(10):
			f.write(str(totalAccList[i]))
			# for j in range(totalAlNum):
			# 	f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

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

if __name__ == "__main__":

	### processedBooksElectronics books ---> kitchen
	### processedKitchenElectronics kitchen ---> electronics

	featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName

	featureMatrix, labelList = readFeatureLabel(featureLabelFile)
	featureMatrix = np.array(featureMatrix)
	labelArray = np.array(labelList)

	### processedBooksElectronics books ---> kitchen transferLabel_books--electronics
	### processedKitchenElectronics kitchen ---> electronics transferLabel_electronics--kitchen

	transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
	auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)
	transferLabelArray = np.array(transferLabelList)
	
	auditorLabelArray = np.array(auditorLabelList)
	# print(auditorLabel)
	# exit()
	# label = np.array([float(i.strip()) for i in open('targetAuditorLabel.txt').readlines()])

	# tmp = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
	# label = tmp[:,-1]
	print('class count of true labels of all ex:\n', ct(labelArray))
	print("count of auditor", ct(auditorLabelArray))
	# exit()
	# mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

	auditorLR = LR(random_state=3)
	auditorLR.fit(featureMatrix, auditorLabelArray)
	print(auditorLR.coef_)

	# fn = get_name_features(raw_pt)
	# fold = 10
	# rounds = 100
	# al = active_learning(fold, rounds, featureMatrix, transferLabelArray, auditorLabelArray)

	# al.run_CV()
