"""
active learning with random initialization and CB query strategy
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
# dataName = "sensor_rice"



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

	def __init__(self, fold, rounds, fn, label, category, multipleClass):

		self.m_category = category
		print("category", category)
		self.m_multipleClass = multipleClass
		print("multipleClass", multipleClass)

		self.fold = fold
		self.rounds = rounds

		self.fn = np.array(fn)
		self.label = np.array(label)

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
			# print("unlabeledId\t", unlabeledId)
			labelPredictProb = self.m_clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]

			selectCB = self.get_select_confidence_bound(unlabeledId)

			idScore = selectCB
			
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
		# print("acc\t", acc)
		# print debug
		return acc

	def pretrainSelectInit(self, train, foldIndex):
		
		initList = self.m_initialExList[foldIndex]
		
		print("initList", initList)

		return initList


	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		totalTransferNumList = []
		# np.random.seed(3)
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
		cvIter = 0
		totalAccList = [[] for i in range(10)]
		totalNewClassFlagList = [[] for i in range(10)]
		for foldIndex in range(foldNum):

			if self.m_multipleClass:
				self.m_clf = LR(multi_class="multinomial", solver='lbfgs',random_state=3,  fit_intercept=False)
			else:
				self.m_clf = LR(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			fn_train = self.fn[train]

			test = foldInstanceList[foldIndex]
		
			fn_test = self.fn[test]
			label_test = self.label[test]

			featureDim = len(fn_train[0])
			self.init_confidence_bound(featureDim)
			
			initExList = []
			initExList = self.pretrainSelectInit(train, foldIndex)
			
			fn_init = self.fn[initExList]
			label_init = self.label[initExList]

			print("initExList\t", initExList, label_init)
			queryIter = 3
			labeledExList = []
			unlabeledExList = []
			
			labeledExList.extend(initExList)
			unlabeledExList = list(set(train)-set(labeledExList))

			while queryIter < rounds:
				fn_train_iter = []
				label_train_iter = []

				fn_train_iter = self.fn[labeledExList]
				label_train_iter = self.label[labeledExList]

				self.m_clf.fit(fn_train_iter, label_train_iter) 

				idx = self.select_example(unlabeledExList) 
				self.update_select_confidence_bound(idx)
				# print(queryIter, "idx", idx, self.label[idx])
				# self.update_select_confidence_bound(idx)

				labeledExList.append(idx)
				unlabeledExList.remove(idx)

				acc = self.get_pred_acc(fn_test, label_test, labeledExList)
				totalAccList[cvIter].append(acc)
				queryIter += 1

			cvIter += 1      
			
		totalACCFile = modelVersion+"_acc.txt"
		totalACCFile = os.path.join(fileSrc, totalACCFile)


		f = open(totalACCFile, "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()


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

	dataName = "sensor_rice"

	modelName = "activeLearning_CB_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

	modelVersion = modelName+"_"+timeStamp
	fileSrc = dataName

	"""
	 	processedKitchenElectronics
	"""
	if dataName == "electronics":

		featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)
		
		###processedKitchenElectronics transferLabel_electronics--kitchen.txt
		transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
		auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		transferLabelArray = np.array(transferLabelList)
		auditorLabelArray = np.array(auditorLabelList)

		initialExList = []
		initialExList = [[397, 1942, 200], [1055, 144, 873], [865, 1702, 1769], [1156, 906, 1964], [1562, 1299, 617], [1431, 1033, 1823], [1063, 1313, 1183], [817, 1631, 426], [360, 1950, 1702], [1921, 822, 1528]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, auditorLabelArray, "sentiment_electronics", multipleClassFlag)

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

		auditorLabelArray = np.array(auditorLabelList0)

		initialExList = []
		initialExList = [[470, 352, 217],  [203, 280, 54], [267, 16, 190], [3, 362, 268], [328, 307, 194], [77, 413, 380],  [119, 170, 420], [312, 310, 6],  [115, 449, 226], [297, 87, 46]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, auditorLabelArray, "sensor", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()