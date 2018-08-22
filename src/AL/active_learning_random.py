"""
active learning with random initialization and least confidence query strategy
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

dataName = "electronics"

modelName = "activeLearning_random_"+dataName
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

	def __init__(self, fold, rounds, fn, label):

		self.fold = fold
		self.rounds = rounds

		self.fn = np.array(fn)
		self.label = np.array(label)

		self.tao = 0
		self.alpha_ = 1

		self.ex_id = dd(list)

		self.m_lambda = 0.01
		self.m_selectA = 0
		self.m_selectAInv = 0
		self.m_selectCbRate = 0.002 ###0.005
		self.clf = 0

	def select_example(self, unlabeled_list):
		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)

		# selectedID = unlabeled_list[0]
		
		selectedID = random.sample(unlabeled_list, 1)[0]

		return selectedID

		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			labelPredictProb = self.clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]

			selectCB = self.get_select_confidence_bound(unlabeledId)
			
			idScore = selectCB
			unlabeledIdScoreMap[unlabeledId] = idScore

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.clf.fit(fn_train, label_train)
		fn_preds = self.clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
	
		return acc

	def pretrainSelectInit(self, train):

		posTrain = []
		negTrain = []

		for trainIndex in range(len(train)):
			if self.label[train[trainIndex]] == 1.0:
				posTrain.append(train[trainIndex])
			else:
				negTrain.append(train[trainIndex])

		initList = []

		random.seed(10)

		initList += random.sample(posTrain, 2)
		initList += random.sample(negTrain, 1)

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
			self.clf = LR(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			trainNum = int(totalInstanceNum*0.9)
			
			fn_test = self.fn[test]
			label_test = self.label[test]
			print("testing", ct(label_test))

			fn_train = self.fn[train]
			
			initExList = []
			initExList = self.pretrainSelectInit(train)
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

				self.clf.fit(fn_train_iter, label_train_iter) 

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
		f = open(totalACCFile, "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	transferLabelList = []
	targetLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		# print(float(line[1]))

		transferLabelList.append(float(line[1]))
		targetLabelList.append(float(line[2]))

	f.close()

	return transferLabelList, targetLabelList

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

if __name__ == "__main__":

	# featureDim = 100
	# sampleNum = 400
	# classifierNum = 2

	# featureLabelFile = "../simulatedFeatureLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"

	### processedBooksElectronics
	featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName

	f = open(featureLabelFile)
	featureMatrix = []
	label = []
	for rawLine in f:
		featureLine = rawLine.strip().split("\t")
		featureNum = len(featureLine)
		featureList = []
		for featureIndex in range(featureNum-1):
			featureVal = float(featureLine[featureIndex])
			featureList.append(featureVal)

		labelVal = float(featureLine[featureNum-1])

		featureMatrix.append(featureList)
		label.append(labelVal)
	f.close()

	
	fold = 10
	rounds = 100
	al = active_learning(fold, rounds, featureMatrix, label)

	al.run_CV()
