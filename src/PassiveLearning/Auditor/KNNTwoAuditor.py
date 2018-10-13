"""
train KNN on train dataset. For training, the label is strong label, feature.
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

		posIndexList = []
		negIndexList = []

		accWeakPos = 0.0
		accWeakNeg = 0.0

		for i in range(totalInstanceNum):
			if self.transfer_label[i] == 1.0:
				posIndexList.append(i)

				if self.target_label[i] == 1.0:
					accWeakPos += 1.0
			else:
				if self.target_label[i] == 0.0:
					accWeakNeg += 1.0
				negIndexList.append(i)

		# indexList = [i for i in range(totalInstanceNum)]

		print("featureNum", len(self.target_fn[0]))
		# print("non zero feature num", sum(self.fn[0]))

		totalTransferNumList = []
		# np.random.seed(3)
		# np.random.shuffle(indexList)

		random.shuffle(posIndexList)
		random.shuffle(negIndexList)

		posInstanceNum = len(posIndexList)
		print("acc weak pos", accWeakPos/posInstanceNum)

		negInstanceNum = len(negIndexList)
		print("acc weak neg", accWeakNeg/negInstanceNum)

		print("weak pos label num", posInstanceNum)
		print("weak neg label num", negInstanceNum)

		foldNum = 10
		foldInstanceNum = int(posInstanceNum*1.0/foldNum)
		foldInstanceList = []
		for foldIndex in range(foldNum-1):
			foldIndexInstanceList = negIndexList[foldIndex*foldInstanceNum:(foldIndex+1)*foldInstanceNum]
			foldInstanceList.append(foldIndexInstanceList)

		foldIndexInstanceList = negIndexList[foldInstanceNum*(foldNum-1):]
		foldInstanceList.append(foldIndexInstanceList)
		# kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		cvIter = 0
		# random.seed(3)
		totalAccList = [0 for i in range(10)]

		coefList = [0 for i in range(10)]

		auditorKNNStrongAccList = []
		auditorKNNWeakAccList = []

		auditorKNNAccList = []
		targetKNNAccList = []
		transferKNNAccList = []


		KNNNeighbors = 20
		for foldIndex in range(foldNum):
			print("###############%d#########"%foldIndex)
			# print("foldIndex", foldIndex)
			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			# print("train num", len(train))
			# print("test num", len(test))
 	
			# KNNClf.fit(self.target_fn[train], self.target_label[train])
			# 
			"""
			fit a knn clf on strong labels and work as auditor
			"""
			KNNClf = KNN(n_neighbors=KNNNeighbors)
			KNNClf.fit(self.target_fn[train], self.target_label[train])

			predTargetLabels = KNNClf.predict(self.target_fn[test])
			# print(predTargetLabels)
			predAuditorLabels = predTargetLabels == self.transfer_label[test]

			acc = accuracy_score(self.auditor_label[test], predAuditorLabels)
			# print("auditor KNN clf with strong labels", acc)
			auditorKNNStrongAccList.append(acc)

			# KNNClf.fit(self.target_fn[train], self.transfer_label[train])
			# 
			"""
			fit a knn clf on strong labels and work as auditor
			"""
			KNNClf = KNN(n_neighbors=KNNNeighbors)
			KNNClf.fit(self.target_fn[train], self.transfer_label[train])

			predTargetLabels = KNNClf.predict(self.target_fn[test])
			predAuditorLabels = predTargetLabels == self.transfer_label[test]

			acc = accuracy_score(self.auditor_label[test], predAuditorLabels)
			# print("auditor KNN clf with weak labels", acc)
			auditorKNNWeakAccList.append(acc)


			# KNNClf.fit(self.target_fn[train], self.auditor_label[train])
			# 
			"""
			fit a knn clf on auditor labels
			"""
			KNNClf = KNN(n_neighbors=KNNNeighbors)
			KNNClf.fit(self.target_fn[train], self.auditor_label[train])

			predAuditorLabels = KNNClf.predict(self.target_fn[test])
			acc = accuracy_score(self.auditor_label[test], predAuditorLabels)
			# print("auditor KNN clf", acc)
			auditorKNNAccList.append(acc)


			"""
			fit a knn clf on strong oracle's labels
			"""
			KNNClf = KNN(n_neighbors=KNNNeighbors)
			KNNClf.fit(self.target_fn[train], self.target_label[train])

			predTargetLabels = KNNClf.predict(self.target_fn[test])
			acc = accuracy_score(self.target_label[test], predTargetLabels)
			# print("target KNN clf", acc)
			targetKNNAccList.append(acc)


			"""
			fit a knn clf on weak oracle's labels
			"""
			KNNClf = KNN(n_neighbors=KNNNeighbors)
			KNNClf.fit(self.target_fn[train], self.transfer_label[train])

			predTransferLabels = KNNClf.predict(self.target_fn[test])
			acc = accuracy_score(self.transfer_label[test], predTransferLabels)
			# print("transfer KNN clf", acc)
			transferKNNAccList.append(acc)

		print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
		print("auditor knn acc", np.mean(auditorKNNAccList), np.sqrt(np.var(auditorKNNAccList)))
		print("auditor strong knn acc", np.mean(auditorKNNStrongAccList), np.sqrt(np.var(auditorKNNStrongAccList)))
		print("auditor weak knn acc", np.mean(auditorKNNWeakAccList), np.sqrt(np.var(auditorKNNWeakAccList)))
		
		print("target knn acc", np.mean(targetKNNAccList), np.sqrt(np.var(targetKNNAccList)))
		print("transfer knn acc", np.mean(transferKNNAccList), np.sqrt(np.var(transferKNNAccList)))


		
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
	featureLabelFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

	featureMatrix, labelList = readFeatureLabel(featureLabelFile)

	transferLabelFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics/transferLabel_books--electronics.txt"
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
