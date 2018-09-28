"""
offline learning via considering weak labels as data poison and remove poisoned data to train a better classifier, compare distance of features via slab threshold, randomly select instances to flip their labels. 
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
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

from datetime import datetime
import os

"""
electronics, sensor_rice
"""


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

class active_learning:

	def __init__(self, fold, rounds, fn, label, transferlabel, category, multipleClass):

		self.m_category = category
		print("category", category)
		self.m_multipleClass = multipleClass
		print("multipleClass", multipleClass)

		self.fold = fold
		self.rounds = rounds

		self.fn = np.array(fn)
		self.label = np.array(label)
		self.transferLabel = np.array(transferlabel)

		self.m_lambda = 0.01
		self.m_selectA = 0
		self.m_selectAInv = 0
		self.m_selectCbRate = 0.002 ###0.005
		self.m_clf = 0

		self.m_initialExList = []

	def setInitialExList(self, initialExList):
		self.m_initialExList = initialExList


	def select_example(self, unlabeled_list):
		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)

		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			labelPredictProb = self.m_clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]

			sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)
			maxLabelPredictProb = sortedLabelPredictProb[0]
			subMaxLabelPredictProb = sortedLabelPredictProb[1]
			marginProb = maxLabelPredictProb-subMaxLabelPredictProb
			idScore = -marginProb

			unlabeledIdScoreMap[unlabeledId] = idScore

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def init_confidence_bound(self, featureDim):
		self.m_selectA = self.m_lambda*np.identity(featureDim)
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

	def update_select_confidence_bound(self, exId):
		# print("updating select cb", exId)
		self.m_selectA += np.outer(self.fn[exId], self.fn[exId])
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

	def get_select_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.fn[exId], self.m_selectAInv), self.fn[exId]))

		return CB

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
	
		return acc

	def pretrainSelectInit(self, train, foldIndex):
		
		initList = self.m_initialExList[foldIndex]
		
		print("initList", initList)

		return initList

	def generateCleanData(self, train, slabThreshold):
		sampledTrainNum = len(train)
		cleanFeatureTrain = []
		cleanLabelTrain = []

		posExpectedFeatureTrain = [0.0 for i in range(len(self.fn[0]))]
		negExpectedFeatureTrain = [0.0 for i in range(len(self.fn[0]))]

		posLabelNum = 0.0
		negLabelNum = 0.0

		
		flipTrainNum = 150
		### obtain sampled train instances
		flipLabelsTrain = random.sample(train, flipTrainNum)

		for trainIndex in range(sampledTrainNum):
			trainID = train[trainIndex]
			feature = self.fn[trainID]

			trueLabel = self.label[trainID]
			transferLabel = self.transferLabel[trainID]

			if trainID in flipLabelsTrain:
				if trueLabel == 1:
					posExpectedFeatureTrain += feature
					posLabelNum += 1.0
				else:
					negExpectedFeatureTrain += feature
					negLabelNum += 1.0
			else:
				if transferLabel == 1:
					posExpectedFeatureTrain += feature
					posLabelNum += 1.0
				else:
					negExpectedFeatureTrain += feature
					negLabelNum += 1.0
		"""
		obtain expected feature for pos and neg classes
		"""
		# for trainIndex in range(sampledTrainNum):
		# 	trainID = train[trainIndex]

		# 	feature = self.fn[trainID]

		# 	trueLabel = self.label[trainID]
		# 	transferLabel = self.transferLabel[trainID]

		# 	if trueLabel == 1:
		# 		posExpectedFeatureTrain += feature
		# 		posLabelNum += 1.0
		# 	else:
		# 		negExpectedFeatureTrain += feature
		# 		negLabelNum += 1.0

			# if transferLabel == 1:
			# 	posExpectedFeatureTrain += feature
			# 	posLabelNum += 1.0
			# else:
			# 	negExpectedFeatureTrain += feature
			# 	negLabelNum += 1.0

		posExpectedFeatureTrain /= posLabelNum
		negExpectedFeatureTrain /= negLabelNum

		### obtain threshold

		slabDisThreshold = slabThreshold
		sphereDisThreshold = 0.0

		featureResidual = posExpectedFeatureTrain-negExpectedFeatureTrain

		featureDisList = []

		correctCleanNum = 0.0

		### filter data
		for trainIndex in range(sampledTrainNum):
			trainID = train[trainIndex]
			featureTrain = self.fn[trainID]
			transferLabel = self.transferLabel[trainID]
			trueLabel = self.label[trainID]

			featureDis = 0.0
			if transferLabel == 1.0:
				featureDis = featureTrain-posExpectedFeatureTrain

			if transferLabel == 0.0:
				featureDis = featureTrain-negExpectedFeatureTrain


			featureDis = np.abs(np.dot(featureDis, featureResidual))
			if featureDis < slabDisThreshold:
				cleanFeatureTrain.append(self.fn[trainID])
				cleanLabelTrain.append(transferLabel)

				if transferLabel == trueLabel:
					correctCleanNum += 1.0

			featureDisList.append(featureDis)

			# if transferLabel == trueLabel:
			# 	cleanFeatureTrain.append(self.fn[trainID])
			# 	cleanLabelTrain.append(transferLabel)
		
		# print("featureDisList", np.mean(featureDisList), np.median(featureDisList), np.sqrt(np.var(featureDisList)))
		# print("correctCleanNum", correctCleanNum, "cleanNum", len(cleanLabelTrain), correctCleanNum*1.0/len(cleanLabelTrain))
		cleanFeatureTrain = np.array(cleanFeatureTrain)
		cleanLabelTrain = np.array(cleanLabelTrain)

		return cleanFeatureTrain, cleanLabelTrain

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		print("featureNum", len(self.fn[0]))
		# print("non zero feature num", sum(self.fn[0]))

		totalTransferNumList = []
		# np.random.seed(3)
		# np.random.shuffle(indexList)

		random.shuffle(indexList)

		foldNum = 10
		foldInstanceNum = int(totalInstanceNum*1.0/foldNum)
		foldInstanceList = []

		for foldIndex in range(foldNum-1):
			foldIndexInstanceList = indexList[foldIndex*foldInstanceNum:(foldIndex+1)*foldInstanceNum]
			foldInstanceList.append(foldIndexInstanceList)

		foldIndexInstanceList = indexList[foldInstanceNum*(foldNum-1):]
		foldInstanceList.append(foldIndexInstanceList)
		# kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		
		slabThresholdList = np.arange(0.05, 0.3, 0.01)
		for slabThreshold in slabThresholdList:
			# print("*************************slabThreshold: %f****************"%slabThreshold)
			cvIter = 0
			# random.seed(3)
			totalAccList = [0 for i in range(10)]

			coefList = [0 for i in range(10)]

			for foldIndex in range(foldNum):

				if self.m_multipleClass:
					self.m_clf = LR(multi_class="multinomial", solver='lbfgs',random_state=3,  fit_intercept=False)
				else:
					self.m_clf = LR(random_state=3)

				train = []
				for preFoldIndex in range(foldIndex):
					train.extend(foldInstanceList[preFoldIndex])

				test = foldInstanceList[foldIndex]
				for postFoldIndex in range(foldIndex+1, foldNum):
					train.extend(foldInstanceList[postFoldIndex])

				trainNum = int(totalInstanceNum*0.9)
				
				# print(test)
				fn_test = self.fn[test]

				label_test = self.label[test]

				sampledTrainNum = len(train)
				# sampledTrainNum = 150
				train_sampled = random.sample(train, sampledTrainNum)

				fn_train = self.fn[train_sampled]
				label_train = self.label[train_sampled]
				transferLabel_train = self.transferLabel[train_sampled]

				cleanFeatureTrain, cleanLabelTrain = self.generateCleanData(train_sampled, slabThreshold)
				# print("clean data num", len(cleanLabelTrain))
				self.m_clf.fit(cleanFeatureTrain, cleanLabelTrain)

				# print("data num", len(transferLabel_train))
				# self.m_clf.fit(fn_train, transferLabel_train)

				# self.m_clf.fit(fn_train, label_train)

				coefList[cvIter] = self.m_clf.coef_

				label_preds = self.m_clf.predict(fn_test)
				acc = accuracy_score(label_test, label_preds)

				totalAccList[cvIter] = acc

				cvIter += 1      
			
			# totalACCFile = modelVersion+".txt"
			# f = open(totalACCFile, "w")
			# for i in range(10):
			# 	f.write(str(totalAccList[i]))
			# 	# for j in range(totalAlNum):
			# 	# 	f.write(str(totalAccList[i][j])+"\t")
			# 	f.write("\n")
			# f.close()

			# coefFile = modelVersion+"_coef.txt"
			# f = open(coefFile, "w")
			# for i in range(10):
			# 	coef4Classifier = coefList[i]
			# 	coefNum = len(coef4Classifier)

			# 	for coefIndex in range(coefNum):
			# 		f.write(str(coef4Classifier[coefIndex])+"\t")
			# 	f.write("\n")

			# f.close()

			print(np.mean(totalAccList), np.sqrt(np.var(totalAccList)))

def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	auditorLabelList = []
	transferLabelList = []
	trueLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		auditorLabelList.append(float(line[0]))
		transferLabelList.append(float(line[1]))
		trueLabelList.append(float(line[2]))

	f.close()

	return auditorLabelList, transferLabelList, trueLabelList

def readFeatureLabelCSV(csvFile):
    f = open(csvFile)

    firstLine = False

    featureMatrix = []
    label = []

    firstLine = f.readline()
    
    posFeatureMatrix = []
    posLabel = []
    negFeatureMatrix = []
    negLabel = []

    for rawLine in f:
        line = rawLine.strip().split(",")
        lineLen = len(line)

        featureList = []
        for lineIndex in range(lineLen-1):
            featureVal = float(line[lineIndex])
            featureList.append(featureVal)

#         featureMatrix.append(featureList)
        if line[lineLen-1] == "FALSE":
            negFeatureMatrix.append(featureList)
            negLabel.append(0.0)
        else:
            posFeatureMatrix.append(featureList)
            # print(line[lineLen-1])
            posLabel.append(1.0)
    
    negFeatureMatrix = random.sample(negFeatureMatrix, len(posLabel))
    negLabel = random.sample(negLabel, len(posLabel))
    
    featureMatrix = np.vstack((negFeatureMatrix, posFeatureMatrix))
    label = np.hstack((negLabel, posLabel))
    
    f.close()

    return featureMatrix, label

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

def readSensorData():
	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../../dataset/sensorType/sdh_soda_rice/rice_names').readlines()]
	tmp = np.genfromtxt('../../dataset/sensorType/rice_hour_sdh', delimiter=',')
	label = tmp[:,-1]

	fn = get_name_features(raw_pt)

	featureMatrix = fn
	labelList = label

	return featureMatrix, labelList

if __name__ == "__main__":

	dataName = "electronics"

	modelName = "offline_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

	modelVersion = modelName+"_"+timeStamp
	fileSrc = dataName

	"""
	 	processedKitchenElectronics
	"""
	if dataName == "electronics":
		featureLabelFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		transferLabelFile = "../../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
		auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)
		transferLabelArray = np.array(transferLabelList)
	
		initialExList = [[397, 1942, 200], [100, 1978, 657], [902, 788, 1370], [1688, 1676, 873], [1562, 1299, 617], [986, 1376, 562], [818, 501, 1922], [600, 1828, 1622], [1653, 920, 1606], [39, 1501, 166]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, labelArray,  transferLabelArray, "sentiment_electronics", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()


	"""
	 	sensor type
	"""
	if dataName == "sensor_rice":
		featureMatrix, labelList = readSensorData()

		transferLabelFile0 = "../../dataset/sensorType/sdh_soda_rice/transferLabel_sdh--rice.txt"
		auditorLabelList0, transferLabelList0, trueLabelList = readTransferLabel(transferLabelFile0)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(trueLabelList)

		initialExList = [[470, 352, 217],  [203, 280, 54], [267, 16, 190], [130, 8, 318], [290, 96, 418], [252, 447, 55],  [429, 243, 416], [240, 13, 68], [115, 449, 226], [262, 127, 381]]

		fold = 10
		rounds = 150

		multipleClassFlag = True
		al = active_learning(fold, rounds, featureMatrix, labelArray, "sensor", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()

	"""
	 	synthetic data
	"""
	if dataName == "simulation":
		featureLabelFile = "../../dataset/synthetic/simulatedFeatureLabel_500_20_2.txt"

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		transferLabelFile0 = "../../dataset/synthetic/simulatedTransferLabel_500_20_2.txt"
		auditorLabelList0, transferLabelList0, trueLabelList = readTransferLabel(transferLabelFile0)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(trueLabelList)

		initialExList = []
		initialExList = [[42, 438, 9],  [246, 365, 299], [282, 329, 254], [114, 158, 255], [161, 389, 174], [283, 86, 90],  [75, 368, 403], [48, 481, 332], [356, 289, 176], [364, 437, 156]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, labelArray, "synthetic", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()