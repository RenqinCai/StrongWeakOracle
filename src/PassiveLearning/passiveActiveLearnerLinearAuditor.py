"""
train a weak oracle on the source domain data and a strong oracle on the target domain data. Use the difference of oracle's weight to predict the output of the auditor. Use the output of dot product to predict the auditor label
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
	
		self.m_weakClf = LR(random_state=3)
		self.m_weakClf.fit(self.source_fn, self.source_label)
		
		# source_fn_train, source_fn_test, source_label_train, source_label_test = train_test_split(self.target_fn, self.source_label, random_state=3, test_size=0.1)
		# self.m_clf.fit(self.source_fn, self.source_label)
		

		# self.m_clf = LR(fit_intercept=False)

		# self.m_clf = LR(random_state=3, fit_intercept=False)

		# train = []
		# foldIndex = 1
		# for preFoldIndex in range(foldIndex):
		# 	train.extend(foldInstanceList[preFoldIndex])
		# for postFoldIndex in range(foldIndex+1, foldNum):
		# 	train.extend(foldInstanceList[postFoldIndex])

		# test = foldInstanceList[foldIndex]

		# self.m_strongClf = LR(random_state=3)
		# self.m_strongClf.fit(self.target_fn[train], self.target_label[train])

		# predTargetLabels = self.m_weakClf.predict(self.target_fn)

		# self.m_transferWeakClf = LR(random_state=3)
		# self.m_transferWeakClf.fit(self.target_fn[train], predTargetLabels[train])

		# auditorLabels = (self.target_label == predTargetLabels).reshape(-1, 1)
		# # print(test)
		

		# strongPred = np.dot(self.target_fn[test], self.m_strongClf.coef_.reshape(-1, 1))+self.m_strongClf.intercept_
		# weakPred = np.dot(self.target_fn[test], self.m_transferWeakClf.coef_.reshape(-1, 1))+self.m_transferWeakClf.intercept_

		# # print(strongPred, weakPred)

		# predAuditorLabels = strongPred*weakPred

		# predAuditorLabels = predAuditorLabels > 0
		# correctPredLabels = (auditorLabels[test] == predAuditorLabels)
			
		# acc = np.sum(correctPredLabels)*1.0/len(correctPredLabels)
		# print("ACC", acc)

		train = []
		foldIndex = 1
		for preFoldIndex in range(foldIndex):
			train.extend(foldInstanceList[preFoldIndex])
		for postFoldIndex in range(foldIndex+1, foldNum):
			train.extend(foldInstanceList[postFoldIndex])

		test = foldInstanceList[foldIndex]

		self.m_strongClf = LR(random_state=3)
		self.m_strongClf.fit(self.target_fn, self.target_label)

		predTargetLabels = self.m_weakClf.predict(self.target_fn)
		print(predTargetLabels.shape)
		print(self.target_label.shape)

		self.m_transferWeakClf = LR(random_state=3)
		self.m_transferWeakClf.fit(self.target_fn, predTargetLabels)

		auditorLabels = (self.target_label == predTargetLabels)

		coefDiff = self.m_strongClf.coef_ - self.m_transferWeakClf.coef_
		predAuditorLabels = np.dot(self.target_fn, coefDiff.reshape(-1, 1))
		# print("intercept", self.m_strongClf.intercept_, self.m_weakClf.intercept_)
		predAuditorLabels += self.m_strongClf.intercept_ - self.m_weakClf.intercept_
		predAuditorLabels = np.abs(predAuditorLabels)

		maxAcc = 0
		maxDelta = 0
		deltaList = [i for i in np.arange(0.0, 25.0, 0.005)]
		for delta in deltaList:
		# delta = 0.005
				# if delta %2 == 0:
				# 	print("delta", delta)
			# delta = 3.0
			predAuditorLabelsDelta = (predAuditorLabels < delta)
			# print("predAuditorLabels", predAuditorLabelsDelta)
			
			accLinear = accuracy_score(auditorLabels, predAuditorLabelsDelta)

			if accLinear > maxAcc:
				maxAcc = accLinear
				maxDelta = delta

		# print(test)
		
		# linearAuditor = LR(random_state=3)
		# linearAuditor.fit(self.target_fn, auditorLabels)
		# predAuditorLabels = linearAuditor.predict(self.target_fn)
		# print(strongPred, weakPred)

		# predAuditorLabels = strongPred-weakPred

		# # predAuditorLabels = predAuditorLabels > 0
		# correctPredLabels = (auditorLabels == predAuditorLabels)
			
		# acc = np.sum(correctPredLabels)*1.0/len(correctPredLabels)
		print(maxDelta, "ACC", maxAcc)
		
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

	sourceFeatureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+sourceDataName
	sourceFeatureMatrix, sourceLabelList = readFeatureLabel(sourceFeatureLabelFile)

	sourceLabel = np.array(sourceLabelList)
	sourceFeatureMatrix = np.array(sourceFeatureMatrix)

	print('class count of true source labels of all ex:\n', ct(sourceLabel))

	targetFeatureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+targetDataName
	targetFeatureMatrix, targetLabelList = readFeatureLabel(targetFeatureLabelFile)

	targetLabel = np.array(targetLabelList)
	targetFeatureMatrix = np.array(targetFeatureMatrix)

	print('class count of true target labels of all ex:\n', ct(targetLabel))

	fold = 1
	rounds = 100
	al = active_learning(fold, rounds, sourceFeatureMatrix, sourceLabel, targetFeatureMatrix, targetLabel)

	al.run_CV()
