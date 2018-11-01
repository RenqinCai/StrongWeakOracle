"""
active learning with a strong oracle and weak labels are given. use theta_{weak} Weaklabels as auditor
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

from datetime import datetime
import os
import copy

from multiprocessing import Pool 

import sklearn.utils.validation
from sklearn.exceptions import NotFittedError

import sys
sys.path.insert(0, "../utils")

from utils import *

from random import randint


random.seed(10)
np.random.seed(10)

def sigmoid(x):
  	  return (1 / (1 + np.exp(-x)))

def get_name_features(names):

		name = []
		for i in names:
			s = re.findall('(?i)[a-z]{2,}',i)
			name.append(' '.join(s))

		cv = CV(analyzer='char_wb', ngram_range=(3,4))
		fn = cv.fit_transform(name).toarray()

		return fn

class _Corpus:

	def __init__(self):
		self.m_category = None
		self.m_multipleClass = None
		self.m_feature = None
		self.m_label = None
		self.m_transferLabel = None
		self.m_auditorLabel = None
		self.m_initialExList = []

	def initCorpus(self, featureMatrix, labelArray, transferLabelArray, auditorLabelArray, initialExList, category, multipleClass):
		self.m_category = category
		print("category", category)
		self.m_multipleClass = multipleClass
		print("multipleClass", multipleClass)
		self.m_feature = featureMatrix
		self.m_label = labelArray
		self.m_transferLabel = transferLabelArray
		self.m_initialExList = initialExList
		self.m_auditorLabel = auditorLabelArray 


class _ActiveClf:
	def __init__(self, category, multipleClass, StrongLabelNumThreshold):
		self.m_activeClf = None

		self.m_strongLabelNumThresh = StrongLabelNumThreshold

		self.m_initialExList = []

		self.m_labeledIDList = []

		self.m_unlabeledIDList = []

		self.m_accList = []

		self.m_train = None

		self.m_test = None

		self.m_multipleClass = multipleClass

		self.m_category = category
		
		self.m_posAuditor = None

		self.m_negAuditor = None

		self.m_posAuditorSampleTrain = []
		self.m_negAuditorSampleTrain = []

		self.m_posAuditorSampleNumList = []
		self.m_negAuditorSampleNumList = []

		self.m_weakLabelPrecisionList = []
		self.m_weakLabelRecallList = []
		self.m_weakLabelAccList = []

		self.m_weakLabelNumList = []

		self.m_cleanStrategy = "slab"

	def initActiveClf(self, initialSampleList, train, test):
		self.m_initialExList = initialSampleList
		self.m_train = train
		self.m_test = test

		if self.m_multipleClass:
			self.m_activeClf = LR(multi_class="multinomial", solver='lbfgs',random_state=3,  fit_intercept=False)
		else:
			self.m_activeClf = LR(random_state=3)

		self.m_posAuditor = LR(random_state=3)

		self.m_negAuditor = LR(random_state=3)

		self.m_slabThreshold = 0.73
	
	def activeTrainClf(self, corpusObj):
		
		"""
		weak training
		"""
		# self.m_activeClf.fit(corpusObj.m_feature[self.m_train], corpusObj.m_transferLabel[self.m_train])


		"""
		strong training
		"""
		# self.m_activeClf.fit(corpusObj.m_feature[self.m_train], corpusObj.m_label[self.m_train])

		"""
		weak training+150 strong labels
		"""
		# sampledTrainNum = len(self.m_train)
		# sampledTrainNum = 150
		# strongSampleTrain = random.sample(self.m_train, sampledTrainNum)
		# weakSampleTrain = list(set(self.m_train) - set(strongSampleTrain))
		# print("weak sample num", len(weakSampleTrain))

		# featureTrain = np.vstack((corpusObj.m_feature[strongSampleTrain], corpusObj.m_feature[weakSampleTrain]))

		# labelTrain = np.hstack((corpusObj.m_label[strongSampleTrain], corpusObj.m_transferLabel[weakSampleTrain]))

		# self.m_activeClf.fit(featureTrain, labelTrain)

		"""
		150 strong lab
		"""
		sampledTrainNum = len(self.m_train)
		sampledTrainNum = 150
		strongSampleTrain = random.sample(self.m_train, sampledTrainNum)
		# weakSampleTrain = list(set(self.m_train) - set(strongSampleTrain))
		# print("weak sample num", len(weakSampleTrain))
		featureTrain = corpusObj.m_feature[strongSampleTrain]
		labelTrain = corpusObj.m_label[strongSampleTrain]

		self.m_activeClf.fit(featureTrain, labelTrain)


		"""test"""
		predLabelTest = self.m_activeClf.predict(corpusObj.m_feature[self.m_test])

		acc = accuracy_score(corpusObj.m_label[self.m_test], predLabelTest)
		# print("acc", acc)
		self.m_accList.append(acc)
	
def loadData(corpusObj, dataName):
	if dataName == "electronics":
		featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/TF/"+dataName

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		transferLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/TF/transferLabel_books--electronics.txt"
		auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)
		transferLabelArray = np.array(transferLabelList)
		auditorLabelArray = np.array(auditorLabelList)

		multipleClassFlag = False
		initialExList = [[397, 1942, 200], [100, 1978, 657], [902, 788, 1370], [1688, 1676, 873], [1562, 1299, 617], [986, 1376, 562], [818, 501, 1922], [600, 1828, 1622], [1653, 920, 1606], [39, 1501, 166]]

		corpusObj.initCorpus(featureMatrix, labelArray, transferLabelArray, auditorLabelArray, initialExList, "text", multipleClassFlag)

def CVALParaWrapper(args):
	return CVALPerFold(*args)

def CVALPerFold(corpusObj, initialSampleList, train, test):
	StrongLabelNumThreshold = 150

	random.seed(10)
	np.random.seed(10)

	# for i in range(StrongLabelNumThreshold):

		# print(i, "random a number", random.random())
		# print(i, "numpy random a number", np.random.random())
	alObj = _ActiveClf(corpusObj.m_category, corpusObj.m_multipleClass, StrongLabelNumThreshold)
	alObj.initActiveClf(initialSampleList, train, test)
	alObj.activeTrainClf(corpusObj)
	
	accList = alObj.m_accList

	resultPerFold = []
	resultPerFold.append(accList)
	resultPerFold.append(alObj.m_posAuditorSampleNumList)
	resultPerFold.append(alObj.m_negAuditorSampleNumList)
	resultPerFold.append(alObj.m_weakLabelAccList)
	resultPerFold.append(alObj.m_weakLabelPrecisionList)
	resultPerFold.append(alObj.m_weakLabelRecallList)
	resultPerFold.append(alObj.m_weakLabelNumList)
	# resultPerFold = copy.deepcopy(alObj.m_accList)
	# resultPerFold = 0
	return resultPerFold

def parallelCVAL(corpusObj, outputSrc, modelVersion):

	totalSampleNum = len(corpusObj.m_label)
	print("number of samples in dataset:", totalSampleNum)
	sampleIndexList = [i for i in range(totalSampleNum)]
	random.shuffle(sampleIndexList)

	foldNum = 10
	perFoldSampleNum = int(totalSampleNum*1.0/foldNum)
	foldSampleList = []

	for foldIndex in range(foldNum-1):
		perFoldSampleList = sampleIndexList[foldIndex*perFoldSampleNum:(foldIndex+1)*perFoldSampleNum]
		foldSampleList.append(perFoldSampleList)

	perFoldSampleList = sampleIndexList[perFoldSampleNum*(foldNum-1):]
	foldSampleList.append(perFoldSampleList)

	totalAccList = [[] for i in range(foldNum)]

	totalWeakLabelAccList = [[] for i in range(foldNum)]
	totalWeakLabelPrecisionList = [[] for i in range(foldNum)]
	totalWeakLabelRecallList = [[] for i in range(foldNum)]
		
	totalWeakLabelNumList = [[] for i in range(foldNum)]

	totalSampleNum4PosAuditorList = [[] for i in range(foldNum)]
	totalSampleNum4NegAuditorList = [[] for i in range(foldNum)]



	poolNum = 10

	results = []
	argsList = [[] for i in range(poolNum)]

	for poolIndex in range(poolNum):
		foldIndex = poolIndex
		train = []
		for preFoldIndex in range(foldIndex):
			train.extend(foldSampleList[preFoldIndex])

		test = foldSampleList[foldIndex]
		for postFoldIndex in range(foldIndex+1, foldNum):
			train.extend(foldSampleList[postFoldIndex])

		argsList[poolIndex].append(corpusObj)
	
		initialSampleList = corpusObj.m_initialExList[foldIndex]
		argsList[poolIndex].append(initialSampleList)

		argsList[poolIndex].append(train)
		argsList[poolIndex].append(test)

	poolObj = Pool(poolNum)
	results = poolObj.map(CVALParaWrapper, argsList)
	poolObj.close()
	poolObj.join()
	# results = map(CVALParaWrapper, argsList)

	for poolIndex in range(poolNum):
		foldIndex = poolIndex
		resultFold = results[foldIndex]
		totalAccList[foldIndex] = resultFold[0]

		totalSampleNum4PosAuditorList[foldIndex] = resultFold[1]
		totalSampleNum4NegAuditorList[foldIndex] = resultFold[2]

		totalWeakLabelAccList[foldIndex] = resultFold[3]
		totalWeakLabelPrecisionList[foldIndex] = resultFold[4]
		totalWeakLabelRecallList[foldIndex] = resultFold[5]

		totalWeakLabelNumList[foldIndex] = resultFold[6]

		# print(len(accList))
		# print(accList)

	# for foldIndex in range(foldNum):
	# 	train = []
	# 	for preFoldIndex in range(foldIndex):
	# 		train.extend(foldSampleList[preFoldIndex])

	# 	test = foldSampleList[foldIndex]
	# 	for postFoldIndex in range(foldIndex+1, foldNum):
	# 		train.extend(foldSampleList[postFoldIndex])

	# 	initialSampleList = corpusObj.m_initialExList[foldIndex]

	# 	alObj = _ActiveClf(corpusObj.m_category, corpusObj.m_multipleClass, StrongLabelNumThreshold)
	# 	alObj.initActiveClf(initialSampleList, train, test)
	# 	alObj.activeTrainClf(corpusObj)

		# totalAccList[foldIndex] = alObj.m_accList
	print("mean, var acc", np.mean(totalAccList), np.sqrt(np.var(totalAccList)))
	writeFile(outputSrc, modelVersion, totalAccList, "acc")
	# writeFile(outputSrc, modelVersion, totalWeakLabelAccList, "weakLabelAcc")
	# writeFile(outputSrc, modelVersion, totalWeakLabelPrecisionList, "weakLabelPrecision")
	# writeFile(outputSrc, modelVersion, totalWeakLabelRecallList, "weakLabelRecall")
	# writeFile(outputSrc, modelVersion, totalWeakLabelNumList, "weakLabelNum")
	# writeFile(outputSrc, modelVersion, totalSampleNum4PosAuditorList, "sampleNum4PosAuditor")
	# writeFile(outputSrc, modelVersion, totalSampleNum4NegAuditorList, "sampleNum4NegAuditor")


if __name__ == '__main__':
	
	timeStart = datetime.now()

	corpusObj = _Corpus()
	dataName = "electronics"
	loadData(corpusObj, dataName)


	modelName = "offlineWeakLabel_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)
	modelVersion = modelName+"_"+timeStamp
	fileSrc = dataName

	# CVAL(corpusObj, fileSrc, modelVersion)
	parallelCVAL(corpusObj, fileSrc, modelVersion)
	timeEnd = datetime.now()
	print("duration", (timeEnd-timeStart).total_seconds())