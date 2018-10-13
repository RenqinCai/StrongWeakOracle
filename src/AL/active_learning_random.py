"""
active learning with random initialization and random query strategy
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
		
		selectedID = random.sample(unlabeled_list, 1)[0]

		return selectedID

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
	
		return acc

	def pretrainSelectInit(self, train, foldIndex):

		if len(self.m_initialExList) == 0:
			sampledLabels = []
			sampledInstances = []
			while len(set(sampledLabels)) < 2:
				sampledInstances = random.sample(train, 3)
				sampledLabels = self.label[sampledInstances]

			initList = sampledInstances

		# initList = initTotalList[foldIndex]

			print("initList", initList)

			return initList
		else:
			initList = self.m_initialExList[foldIndex]
		
			print("initList", initList)

			return initList

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		totalTransferNumList = []
		random.shuffle(indexList)

		foldNum = 10
		foldInstanceNum = int(totalInstanceNum*1.0/foldNum)
		foldInstanceList = []

		for foldIndex in range(foldNum-1):
			foldIndexInstanceList = indexList[foldIndex*foldInstanceNum:(foldIndex+1)*foldInstanceNum]
			foldInstanceList.append(foldIndexInstanceList)

		foldIndexInstanceList = indexList[foldInstanceNum*(foldNum-1):]
		foldInstanceList.append(foldIndexInstanceList)

		cvIter = 0
		totalAccList = [[] for i in range(10)]
		for foldIndex in range(foldNum):
			# self.clf = LinearSVC(random_state=3)
			# self.clf = LR(fit_intercept=False)
			# self.clf = LR(random_state=3)

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
			
		
			initExList = []
			initExList = self.pretrainSelectInit(train, foldIndex)
			# initExList = [316, 68, 495]
			
			# random.seed(110)
			# initExList = random.sample(train, 3)
			fn_init = self.fn[initExList]
			label_init = self.label[initExList]

			print("initExList\t", initExList, label_init)
			queryIter = 3
			labeledExList = []
			unlabeledExList = []
			###labeled index
			labeledExList.extend(initExList)
			unlabeledExList = list(set(train)-set(labeledExList))

			while queryIter < rounds:
				fn_train_iter = []
				label_train_iter = []

				fn_train_iter = self.fn[labeledExList]
				label_train_iter = self.label[labeledExList]

				self.m_clf.fit(fn_train_iter, label_train_iter) 

				idx = self.select_example(unlabeledExList) 
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

def readFeatureFile(featureFile, labelIndex):
	f = open(featureFile)

	featureMatrix = []
	labelList = []

	for rawLine in f:
		line = rawLine.strip().split("\t")

		lineLen = len(line)

		featureSample = []
		for lineIndex in range(lineLen):
			featureVal = float(line[lineIndex])
			featureSample.append(featureVal)
		
		labelList.append(labelIndex)

		featureMatrix.append(featureSample)

	f.close()

	return featureMatrix, labelList


if __name__ == "__main__":

	dataName = "20News"

	modelName = "activeLearning_random_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

	modelVersion = modelName+"_"+timeStamp
	fileSrc = dataName

	"""
	 	processedKitchenElectronics
	"""

	if dataName == "electronics":
		featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		initialExList = [[397, 1942, 200], [100, 1978, 657], [902, 788, 1370], [1688, 1676, 873], [1562, 1299, 617], [986, 1376, 562], [818, 501, 1922], [600, 1828, 1622], [1653, 920, 1606], [39, 1501, 166]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, labelArray, "sentiment_electronics", multipleClassFlag)

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

	if dataName == "20News":
		featureFile = "../../dataset/20News/baseball"
		labelIndex = 0
		featureMatrix0, labelList0 = readFeatureFile(featureFile, labelIndex)
		print(len(labelList0))

		featureFile = "../../dataset/20News/politicsMisc"
		labelIndex = 1
		featureMatrix1, labelList1 = readFeatureFile(featureFile, labelIndex)
		print(len(labelList1))
		
		featureMatrix = featureMatrix0+featureMatrix1
		labelList = labelList0+labelList1

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		initialExList = []
		initialExList = [[1411, 435, 1390], [1564, 216, 576], [563, 213, 1746], [4, 1162, 1593], [47, 1754, 71], [360, 1512, 1128], [86, 873, 1126], [551, 1540, 437], [1175, 579, 82], [678, 1575, 1306]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, labelArray, "synthetic", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()