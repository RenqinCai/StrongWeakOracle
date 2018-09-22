"""
train KNN on train dataset. The feature is the instance and the output is the weak labels.
During testing, use KNN to output predicted labels and compare the predicted labels with the weak labels to see whether the weak label is trustful or not.
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

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.model_selection import train_test_split

from datetime import datetime

sourceDataName = "books"
targetDataName = "electronics"

adjFeatureList = [1, 3126, 31, 71, 73, 76, 93, 103, 105, 110, 939, 154, 162, 173, 3067, 191, 204, 2496, 242, 263, 264, 268, 297, 308, 311, 327, 2005, 340, 1638, 371, 372, 383, 389, 393, 1849, 399, 409, 415, 430, 444, 457, 460, 466, 469, 478, 514, 537, 543, 547, 556, 561, 567, 604, 630, 646, 647, 1897, 711, 1291, 778, 809, 855, 870, 1940, 906, 912, 950, 952, 959, 960, 973, 986, 988, 995, 1017, 1033, 3236, 1048, 1052, 1054, 1090, 1126, 1136, 1151, 1185, 1218, 1246, 557, 1270, 1275, 1297, 1299, 1300, 1308, 1312, 1314, 1329, 1331, 527, 1357, 1380, 1394, 3296, 1438, 1459, 3370, 1471, 1475, 1500, 1545, 1546, 3214, 3255, 2594, 1581, 1943, 1589, 2072, 1608, 1610, 1618, 1660, 1678, 3457, 1741, 1747, 1766, 1786, 3365, 1812, 270, 1821, 1823, 1830, 1832, 1833, 1853, 1866, 1938, 1891, 1898, 1904, 1918, 1977, 1980, 2778, 1995, 1656, 2033, 1003, 1436, 1452, 2093, 2103, 2105, 1825, 2122, 2134, 2137, 2161, 3247, 2178, 2182, 2200, 2206, 2207, 2210, 2222, 2235, 2239, 2242, 3479, 2252, 2271, 2279, 2284, 2301, 3448, 1336, 2331, 2342, 2382, 2404, 2412, 1720, 2434, 2436, 2443, 2849, 2456, 2470, 301, 2489, 2497, 2503, 2514, 497, 2533, 2548, 2573, 2576, 2578, 3443, 3678, 1238, 2685, 2697, 1076, 2733, 2760, 2766, 2768, 2794, 2798, 2809, 1026, 2844, 2862, 2589, 2871, 2883, 2802, 3325, 19, 2938, 2940, 2955, 2957, 2975, 2988, 3002, 3018, 3042, 3568, 3052, 3192, 158, 175, 3088, 3097, 3105, 3405, 3117, 3120, 3148, 3180, 3194, 842, 866, 801, 3239, 3251, 3253, 3269, 1382, 3318, 3341, 3355, 3356, 3357, 1803, 3367, 223, 1042, 3401, 2057, 3408, 3417, 3426, 2274, 2329, 3454, 3466, 3473, 3537, 3541, 3564, 3581, 3667, 3603, 3613, 3672]

modelName = "passiveAuditorActiveLearner_"+sourceDataName+"_"+targetDataName
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp

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

	def __init__(self, fold, rounds, target_fn, target_label, transfer_label, auditor_label):

		self.fold = fold
		self.rounds = rounds

		# self.source_fn = source_fn
		# self.source_label = source_label

		self.target_fn = target_fn
		self.target_label = target_label

		self.transfer_label = transfer_label
		self.auditor_label = auditor_label

		self.tao = 0
		self.alpha_ = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.m_strongClf = None
		self.m_weakClf = None

		self.ex_id = dd(list)

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.target_fn[labeled_list]
		label_train = self.target_label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.transfer_label)
		print("target totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		print("featureNum", len(self.target_fn[0]))
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
		cvIter = 0
		# random.seed(3)
		totalAccList = [0 for i in range(10)]

		coefList = [0 for i in range(10)]

		auditorKNNAccList = []
		targetKNNAccList = []
		transferKNNAccList = []

		auditorKNNWeakAccList = []
		auditorKNNStrongAccList = []


		# KNNNeighborsList = [1, 3, 5, 10, 20]
		KNNNeighborsList = [3]
		for KNNNeighbors in KNNNeighborsList:
			print("==================%d==============="%KNNNeighbors)
			for foldIndex in range(foldNum):
				print("###############%d#########"%foldIndex)
				# print("foldIndex", foldIndex)
				train = []
				for preFoldIndex in range(foldIndex):
					train.extend(foldInstanceList[preFoldIndex])

				test = foldInstanceList[foldIndex]
				for postFoldIndex in range(foldIndex+1, foldNum):
					train.extend(foldInstanceList[postFoldIndex])
	 
				
				"""
				fit a knn clf on strong oracle's labels
				"""
				KNNClf = KNN(n_neighbors=KNNNeighbors)
				KNNClf.fit(self.target_fn[train][:, adjFeatureList], self.target_label[train])

				predTargetLabels = KNNClf.predict(self.target_fn[test][:, adjFeatureList])
				neighborsTrainIndexList = KNNClf.kneighbors(self.target_fn[test][:, adjFeatureList])[1]
				# print(neighborsIDList)
				# exit()


				testNum = len(test)
				for testIndex in range(testNum):
					ID = test[testIndex]
					neighborsIDList = []
					for neightborsTrainIndex in neighborsTrainIndexList[testIndex]:
						neighborsIDList.append(train[neightborsTrainIndex])

					print(ID, neighborsIDList, predTargetLabels[testIndex], self.target_label[ID])
				
				acc = accuracy_score(self.target_label[test], predTargetLabels)
				print("target KNN clf", acc)
				targetKNNAccList.append(acc)


				"""
				fit a knn clf on weak oracle's labels
				"""
				KNNClf = KNN(n_neighbors=KNNNeighbors)
				KNNClf.fit(self.target_fn[train][:, adjFeatureList], self.transfer_label[train])

				predTransferLabels = KNNClf.predict(self.target_fn[test][:, adjFeatureList])
				acc = accuracy_score(self.transfer_label[test], predTransferLabels)
				print("transfer KNN clf", acc)
				transferKNNAccList.append(acc)



			print("auditor knn acc", np.mean(auditorKNNAccList), np.sqrt(np.var(auditorKNNAccList)))
		

			print("auditor strong knn acc", np.mean(auditorKNNStrongAccList), np.sqrt(np.var(auditorKNNStrongAccList)))
			print("auditor weak knn acc", np.mean(auditorKNNWeakAccList), np.sqrt(np.var(auditorKNNWeakAccList)))

			print("target knn acc", np.mean(targetKNNAccList), np.sqrt(np.var(targetKNNAccList)))
			print("transfer knn acc", np.mean(transferKNNAccList), np.sqrt(np.var(transferKNNAccList)))

			print("******************************************")
		
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
	auditorLabelList = []
	targetLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)
		
		auditorLabelList.append(float(line[0]))
		transferLabelList.append(float(line[1]))
		targetLabelList.append(float(line[2]))

	f.close()

	return auditorLabelList, transferLabelList, targetLabelList

def readFeatureLabelCSV(csvFile):
	f = open(csvFile)

	firstLine = False

	featureMatrix = []
	label = []

	firstLine = f.readline()

	for rawLine in f:
		line = rawLine.strip().split(",")
		lineLen = len(line)

		featureList = []
		for lineIndex in range(lineLen-1):
			featureVal = float(line[lineIndex])
			featureList.append(featureVal)

		featureMatrix.append(featureList)
		if line[lineLen-1] == "FALSE":
			label.append(0.0)
		else:
			# print(line[lineLen-1])
			label.append(1.0)

	f.close()

	return featureMatrix, label

if __name__ == "__main__":

	dataName = "electronics"
	featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

	featureMatrix, labelList = readFeatureLabel(featureLabelFile)

	transferLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/transferLabel_books--electronics.txt"
	auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile)

	# getNoiseRatio(trueLabelList, transferLabelList)
	featureMatrix = np.array(featureMatrix)

	auditorLabel = np.array(auditorLabelList)
	transferLabel = np.array(transferLabelList)
	trueLabel = np.array(trueLabelList)

	fold = 1
	rounds = 100
	al = active_learning(fold, rounds, featureMatrix, trueLabel, transferLabel, auditorLabel)

	al.run_CV()
