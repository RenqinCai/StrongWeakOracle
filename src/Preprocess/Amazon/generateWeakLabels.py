"""
generate transferred labels from weak oracle.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
import os
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

from sklearn.model_selection import train_test_split

from datetime import datetime

# sourceDataName = "electronics"
# targetDataName = "kitchen"

sourceDataName = "books"
targetDataName = "electronics"

dataName = sourceDataName+"--"+targetDataName
modelName = "activeLearning_offline_transfer_"+dataName
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

	def __init__(self, fold, rounds, source_fn, source_label, target_fn, target_label):

		self.fold = fold
		self.rounds = rounds

		self.source_fn = source_fn
		self.source_label = source_label

		self.target_fn = target_fn
		self.target_label = target_label

		self.tao = 0
		self.alpha_ = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.ex_id = dd(list)


	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.target_fn[labeled_list]
		label_train = self.target_label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		# print(len(label_train), sum(label_train), "ratio", sum(label_train)*1.0/len(label_train))
		# print("label_train", label_train)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc


	def run_CV(self, dataDir, sourceName, targetName):

		cvIter = 0
		
		totalInstanceNum = len(self.target_label)
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

		self.m_clf = LR(random_state=3)
		self.m_clf.fit(self.source_fn, self.source_label)

		target_preds = self.m_clf.predict(self.target_fn)
		acc = accuracy_score(self.target_label, target_preds)

		print("transfer acc", acc)	

		transferLabelFileName = "transferLabel_"+sourceName+"--"+targetName+".txt"
		transferLabelFileName = os.path.join(dataDir, transferLabelFileName)
		f = open(transferLabelFileName, "w")

		f.write("auditorLabel"+"\t"+"transferLabel"+"\t"+"trueLabel\n")
		for instanceIndex in range(totalInstanceNum):
			transferLabel = target_preds[instanceIndex]
			trueLabel = self.target_label[instanceIndex]

			if transferLabel == trueLabel:
				f.write("1.0"+"\t")
			else:
				f.write("0.0"+"\t")

			f.write(str(transferLabel)+"\t"+str(trueLabel))
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

	###processedKitchenElectronics electronics ---> kitchen

	###processedBooksElectronics books ---> electronics
	sourceFeatureLabelFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics/"+sourceDataName
	sourceFeatureMatrix, sourceLabelList = readFeatureLabel(sourceFeatureLabelFile)

	sourceLabel = np.array(sourceLabelList)
	sourceFeatureMatrix = np.array(sourceFeatureMatrix)

	print('class count of true source labels of all ex:\n', ct(sourceLabel))

	targetFeatureLabelFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics/"+targetDataName
	targetFeatureMatrix, targetLabelList = readFeatureLabel(targetFeatureLabelFile)

	targetLabel = np.array(targetLabelList)
	targetFeatureMatrix = np.array(targetFeatureMatrix)

	print('class count of true target labels of all ex:\n', ct(targetLabel))

	fold = 1
	rounds = 100
	al = active_learning(fold, rounds, sourceFeatureMatrix, sourceLabel, targetFeatureMatrix, targetLabel)

	dataDir = "../../../dataset/processed_acl/processedBooksKitchenElectronics/"

	al.run_CV(dataDir, "books", "electronics")
