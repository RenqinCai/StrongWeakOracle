"""
active learning with a strong oracle and weak labels are given. Two auditors 
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

		self.m_cleanStrategy = "twoLR"

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

		self.m_slabThreshold = 0.22

	def select_example(self, corpusObj):

		# selectedID = random.sample(self.m_unlabeledIDList, 1)[0]
		# # print("selectedID", selectedID)
		# return selectedID

		unlabeledIdScoreMap = {}
		unlabeledIdNum = len(self.m_unlabeledIDList)

		for unlabeledIdIndex in range(unlabeledIdNum):
			# print("unlabeledIdIndex", unlabeledIdIndex)
			unlabeledId = self.m_unlabeledIDList[unlabeledIdIndex]
			
			score = self.getClassifierMargin(corpusObj, unlabeledId)

			unlabeledIdScoreMap[unlabeledId] = score

		if len(unlabeledIdScoreMap) == 0:
			print("zero unlabeledId")

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		selectedID = sortedUnlabeledIdList[0]
		score4SelectedID = unlabeledIdScoreMap[selectedID]

		return selectedID

	def getClassifierMargin(self, corpusObj, unlabeledId):
		labelPredictProb = self.m_activeClf.predict_proba(corpusObj.m_feature[unlabeledId].reshape(1, -1))[0]

		labelProbMap = {} ##labelIndex: labelProb
		labelNum = len(labelPredictProb)
		for labelIndex in range(labelNum):
			labelProbMap.setdefault(labelIndex, labelPredictProb[labelIndex])

		sortedLabelIndexList = sorted(labelProbMap, key=labelProbMap.__getitem__, reverse=True)

		maxLabelIndex = sortedLabelIndexList[0]
		subMaxLabelIndex = sortedLabelIndexList[1]

		maxLabelProb = labelProbMap[maxLabelIndex]
		subMaxLabelProb = labelProbMap[subMaxLabelIndex]

		margin = maxLabelProb-subMaxLabelProb

		margin = 0 - margin

		return margin

	def generateCleanDataByTwoLR(self, corpusObj):

		sampledTrainNum = len(self.m_train)

		cleanFeatureTrain = []
		cleanLabelTrain = []

		TP = 0.0
		FP = 0.0
		TN = 0.0
		FN = 0.0

		for trainIndex in range(sampledTrainNum):
			trainID = self.m_train[trainIndex]

			if trainID in self.m_labeledIDList:
				continue

			featureTrain = corpusObj.m_feature[trainID]
			transferLabel = corpusObj.m_transferLabel[trainID]
			trueLabel = corpusObj.m_label[trainID]

			if transferLabel == 1.0:

				predictAuditorLabel = 0.0
				try:
					predictAuditorLabel = self.m_posAuditor.predict(featureTrain.reshape(1, -1))[0]
				except NotFittedError as e:
					predictAuditorLabel = randint(0, 1)

				if predictAuditorLabel == 1.0:
					cleanFeatureTrain.append(featureTrain)
					cleanLabelTrain.append(transferLabel)

					if trueLabel == 1.0:
						TP += 1.0
					else:
						FP += 1.0

				else:
					if trueLabel == 1.0:
						FN += 1.0
					else:
						TN += 1.0

			else:
				predictAuditorLabel = 0.0
				try:
					predictAuditorLabel = self.m_negAuditor.predict(featureTrain.reshape(1, -1))[0]
				except NotFittedError as e:
					predictAuditorLabel = randint(0, 1)

				if predictAuditorLabel == 1.0:
					cleanFeatureTrain.append(featureTrain)
					cleanLabelTrain.append(transferLabel)

					if trueLabel == 0.0:
						TP += 1.0
					else:
						FP += 1.0

				else:
					if trueLabel == 0.0:
						FN += 1.0
					else:
						TN += 1.0
		precision = TP/(TP+FP)
		recall = TP/(TP+FN)
		acc = (TP+TN)/(TP+FP+TN+FN)

		self.m_weakLabelNumList.append(len(cleanLabelTrain))
		cleanFeatureTrain = np.array(cleanFeatureTrain)
		cleanLabelTrain = np.array(cleanLabelTrain)

		self.m_weakLabelPrecisionList.append(precision)
		self.m_weakLabelRecallList.append(recall)
		self.m_weakLabelAccList.append(acc)


		return cleanFeatureTrain, cleanLabelTrain

	def generateCleanDataBySlab(self, corpusObj, slabThreshold, posExpectedFeatureTrain, negExpectedFeatureTrain):
		# print("slab filter")
		sampledTrainNum = len(self.m_train)
		cleanFeatureTrain = []
		cleanLabelTrain = []

		slabDisThreshold = slabThreshold

		featureResidual = posExpectedFeatureTrain-negExpectedFeatureTrain

		poisonScoreList = []

		correctCleanNum = 0.0

		### filter data
		for trainIndex in range(sampledTrainNum):
			trainID = self.m_train[trainIndex]

			if trainID in self.m_labeledIDList:
				continue

			featureTrain = corpusObj.m_feature[trainID]
			transferLabel = corpusObj.m_transferLabel[trainID]
			trueLabel = corpusObj.m_label[trainID]

			featureDis = 0.0
			if transferLabel == 1.0:
				featureDis = featureTrain-posExpectedFeatureTrain

			if transferLabel == 0.0:
				featureDis = featureTrain-negExpectedFeatureTrain

			poisonScore = np.abs(np.dot(featureDis, featureResidual))
			if poisonScore < slabDisThreshold:
				cleanFeatureTrain.append(featureTrain)
				cleanLabelTrain.append(transferLabel)

				if transferLabel == trueLabel:
					correctCleanNum += 1.0

			poisonScoreList.append(poisonScore)

			# if transferLabel == trueLabel:
			# 	cleanFeatureTrain.append(self.fn[trainID])
			# 	cleanLabelTrain.append(transferLabel)
		# print("poisonScoreList", np.mean(poisonScoreList), np.median(poisonScoreList), np.sqrt(np.var(poisonScoreList)), "min, max", np.min(poisonScoreList), np.max(poisonScoreList))
		# print("correctCleanNum", correctCleanNum, "cleanNum", len(cleanLabelTrain), correctCleanNum*1.0/len(cleanLabelTrain), sampledTrainNum)
		
		cleanFeatureTrain = np.array(cleanFeatureTrain)
		cleanLabelTrain = np.array(cleanLabelTrain)

		return cleanFeatureTrain, cleanLabelTrain	

	def updateTwoLRAuditor(self, corpusObj, idx, trueLabel, transferLabel):
		if transferLabel == 1.0:
			self.m_posAuditorSampleTrain.append(idx)

			if len(set(corpusObj.m_auditorLabel[self.m_posAuditorSampleTrain])) > 1:
				self.m_posAuditor.fit(corpusObj.m_feature[self.m_posAuditorSampleTrain], corpusObj.m_auditorLabel[self.m_posAuditorSampleTrain])

		if transferLabel == 0.0:
			self.m_negAuditorSampleTrain.append(idx)
			if len(set(corpusObj.m_auditorLabel[self.m_negAuditorSampleTrain])) > 1:
				self.m_negAuditor.fit(corpusObj.m_feature[self.m_negAuditorSampleTrain], corpusObj.m_auditorLabel[self.m_negAuditorSampleTrain])

		self.m_posAuditorSampleNumList.append(len(self.m_posAuditorSampleTrain))
		self.m_negAuditorSampleNumList.append(len(self.m_negAuditorSampleTrain))

	def activeTrainClf(self, corpusObj):
		
		label_init = corpusObj.m_label[self.m_initialExList]
		print("initExList\t", self.m_initialExList, label_init)
		strongLabelNumIter = 3
		
		self.m_labeledIDList.extend(self.m_initialExList)
		self.m_unlabeledIDList = list(set(self.m_train)-set(self.m_labeledIDList))

		feature_train_iter = []
		label_train_iter = []

		feature_train_iter = corpusObj.m_feature[self.m_labeledIDList]
		label_train_iter = corpusObj.m_label[self.m_labeledIDList]
		self.m_activeClf.fit(feature_train_iter, label_train_iter)

		accList = []
		weakLabelACCList = []
		weakLabelPrecisionList = []
		weakLabelRecallList = []

		for idx in self.m_initialExList:
			feature = corpusObj.m_feature[idx]
			trueLabel = corpusObj.m_label[idx]
			transferLabel = corpusObj.m_transferLabel[idx]

			self.updateTwoLRAuditor(corpusObj, idx, trueLabel, transferLabel)

		print("strong label num threshold", self.m_strongLabelNumThresh)

		while strongLabelNumIter < self.m_strongLabelNumThresh:
			# print("random a number", random.random())
			# print("numpy random a number", np.random.random())
			# print("strongLabelNumIter", strongLabelNumIter)

			idx = self.select_example(corpusObj) 
			# print(strongLabelNumIter, "idx", idx)
			self.m_labeledIDList.append(idx)
			self.m_unlabeledIDList.remove(idx)
			
			cleanFeatureTrain = None
			cleanLabelTrain = None

			feature = corpusObj.m_feature[idx]
			trueLabel = corpusObj.m_label[idx]
			transferLabel = corpusObj.m_transferLabel[idx]


			self.updateTwoLRAuditor(corpusObj, idx, trueLabel, transferLabel)

			# if self.m_category == "synthetic":
			# 	cleanFeatureTrain, cleanLabelTrain = self.generateCleanDataBySphere(train, slabThreshold)
				
			if self.m_cleanStrategy == "slab":
				cleanFeatureTrain, cleanLabelTrain = self.generateCleanDataBySlab(corpusObj, self.m_slabThreshold, posExpectedFeatureTrain/posLabelNum, negExpectedFeatureTrain/negLabelNum)

			if self.m_cleanStrategy == "twoLR":
				cleanFeatureTrain, cleanLabelTrain = self.generateCleanDataByTwoLR(corpusObj)

			# print("corpusObj.m_feature[self.m_labeledIDList]", corpusObj.m_feature[self.m_labeledIDList].shape)
			# print(cleanFeatureTrain.shape)
			feature_train_iter = np.vstack((cleanFeatureTrain, corpusObj.m_feature[self.m_labeledIDList]))
			label_train_iter = np.hstack((cleanLabelTrain, corpusObj.m_label[self.m_labeledIDList]))
			# print("clean data num", len(cleanLabelTrain))
			self.m_activeClf.fit(feature_train_iter, label_train_iter)

			predLabelTest = self.m_activeClf.predict(corpusObj.m_feature[self.m_test])

			acc = accuracy_score(corpusObj.m_label[self.m_test], predLabelTest)
			# print("acc", acc)
			self.m_accList.append(acc)
			strongLabelNumIter += 1
	
def loadData(corpusObj, dataName):
	if dataName == "electronics":
		featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
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

# def CVAL(corpusObj, outputSrc, modelVersion):
# 	StrongLabelNumThreshold = 150
# 	totalSampleNum = len(corpusObj.m_label)
# 	print("number of samples in dataset:", totalSampleNum)
# 	sampleIndexList = [i for i in range(totalSampleNum)]
# 	random.shuffle(sampleIndexList)

# 	foldNum = 10
# 	perFoldSampleNum = int(totalSampleNum*1.0/foldNum)
# 	foldSampleList = []

# 	for foldIndex in range(foldNum-1):
# 		perFoldSampleList = sampleIndexList[foldIndex*perFoldSampleNum:(foldIndex+1)*perFoldSampleNum]
# 		foldSampleList.append(perFoldSampleList)

# 	perFoldSampleList = sampleIndexList[perFoldSampleNum*(foldNum-1):]
# 	foldSampleList.append(perFoldSampleList)

# 	totalAccList = [[] for i in range(foldNum)]

# 	for foldIndex in range(foldNum):
# 		train = []
# 		for preFoldIndex in range(foldIndex):
# 			train.extend(foldSampleList[preFoldIndex])

# 		test = foldSampleList[foldIndex]
# 		for postFoldIndex in range(foldIndex+1, foldNum):
# 			train.extend(foldSampleList[postFoldIndex])

# 		initialSampleList = corpusObj.m_initialExList[foldIndex]

# 		alObj = _ActiveClf(corpusObj.m_category, corpusObj.m_multipleClass, StrongLabelNumThreshold)
# 		alObj.initActiveClf(initialSampleList, train, test)
# 		alObj.activeTrainClf(corpusObj)

# 		totalAccList[foldIndex] = alObj.m_accList

# 	writeFile(outputSrc, modelVersion, totalAccList)

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

	writeFile(outputSrc, modelVersion, totalAccList, "acc")
	writeFile(outputSrc, modelVersion, totalWeakLabelAccList, "weakLabelAcc")
	writeFile(outputSrc, modelVersion, totalWeakLabelPrecisionList, "weakLabelPrecision")
	writeFile(outputSrc, modelVersion, totalWeakLabelRecallList, "weakLabelRecall")
	writeFile(outputSrc, modelVersion, totalWeakLabelNumList, "weakLabelNum")
	writeFile(outputSrc, modelVersion, totalSampleNum4PosAuditorList, "sampleNum4PosAuditor")
	writeFile(outputSrc, modelVersion, totalSampleNum4NegAuditorList, "sampleNum4NegAuditor")


if __name__ == '__main__':
	
	timeStart = datetime.now()

	corpusObj = _Corpus()
	dataName = "electronics"
	loadData(corpusObj, dataName)


	modelName = "twoLRAuditors_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)
	modelVersion = modelName+"_"+timeStamp
	fileSrc = dataName

	# CVAL(corpusObj, fileSrc, modelVersion)
	parallelCVAL(corpusObj, fileSrc, modelVersion)
	timeEnd = datetime.now()
	print("duration", (timeEnd-timeStart).total_seconds())