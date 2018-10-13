"""
train a weak oracle on the source domain data and a strong oracle on the target domain data. Use the difference of oracle's weight to predict the output of the auditor
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

		transferLabel = self.m_weakClf.predict(self.target_fn)

		for foldIndex in range(foldNum):
			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])
 	
		# source_fn_train, source_fn_test, source_label_train, source_label_test = train_test_split(self.source_fn, self.source_label, random_state=3, test_size=0.1)
		# self.m_clf.fit(self.source_fn, self.source_label)
			

		# self.m_clf = LR(fit_intercept=False)

		# self.m_clf = LR(random_state=3, fit_intercept=False)

			self.m_strongClf = LR(random_state=3)
			self.m_strongClf.fit(self.target_fn[train], self.target_label[train])

			predLabel = self.m_strongClf.predict(self.target_fn[test])

			accLR = accuracy_score(predLabel, self.target_label[test])
			print("strong accLR", accLR)

			self.m_weakTransferClf = LR(random_state=3)
			self.m_weakTransferClf.fit(self.target_fn[train], transferLabel[train])

			predTargetLabels = self.m_weakTransferClf.predict(self.target_fn[test])

			accLR = accuracy_score(predTargetLabels, transferLabel[test])
			print("weak accLR", accLR)

		# 	auditorLabels = (self.target_label[test] == transferLabel[test]).reshape(-1, 1)
		# # print(test)
		
		# 	coefDiff = self.m_strongClf.coef_ - self.m_weakClf.coef_
		# 	predAuditorLabels = np.dot(self.target_fn[test], coefDiff.reshape(-1, 1))
		# 	# print("intercept", self.m_strongClf.intercept_, self.m_weakClf.intercept_)
		# 	predAuditorLabels += self.m_strongClf.intercept_ - self.m_weakClf.intercept_
		# 	predAuditorLabels = np.abs(predAuditorLabels)

		# # print(predAuditorLabels.shape)

		# 	print(np.min(predAuditorLabels), np.max(predAuditorLabels))
		
		# 	deltaList = [i for i in np.arange(0.0, 25.0, 0.005)]
		# 	maxAcc = 0

		# 	print("pos", sum(auditorLabels)*1.0/len(auditorLabels))
		# # print(deltaList)
		# 	maxDelta = -1
		# 	for delta in deltaList:
		# # delta = 0.005
		# 		# if delta %2 == 0:
		# 		# 	print("delta", delta)
		# 	# delta = 3.0
		# 		predAuditorLabelsDelta = (predAuditorLabels < delta)
		# 		# print("predAuditorLabels", predAuditorLabelsDelta)
				
		# 		accLinear = accuracy_score((self.target_label[test] == transferLabel[test]), predAuditorLabelsDelta)

		# 		# correctPredLabels = (auditorLabels == predAuditorLabelsDelta)
		# 		# print(auditorLabels)
		# 		# print(correctPredLabels)

		# 		# print(np.sum(correctPredLabels), len(correctPredLabels))
		# 		# print(correctPredLabels)
		# 		# acc = np.sum(correctPredLabels)*1.0/len(correctPredLabels)
					
		# 		if accLinear > maxAcc:
		# 			maxAcc = accLinear
		# 			maxDelta = delta
		# 			# print("sum", np.sum(predAuditorLabelsDelta))
		# 			# print("predAuditorLabels", predAuditorLabelsDelta)
		# 	print(maxDelta, "maxAcc\t", maxAcc)

		# 	self.m_auditor = LR(random_state=3)
		# 	self.m_auditor.fit(self.target_fn[train], (self.target_label[train] == transferLabel[train]))

		# 	auditorLabelsLR = self.m_auditor.predict(self.target_fn[test])

		# 	accLR = accuracy_score((self.target_label[test] == transferLabel[test]), auditorLabelsLR)
		# 	print("accLR", accLR)

		# 	strongPred = np.dot(self.target_fn[test], self.m_strongClf.coef_.reshape(-1, 1))+self.m_strongClf.intercept_
		# 	weakPred = np.dot(self.target_fn[test], self.m_weakTransferClf.coef_.reshape(-1, 1))+self.m_weakTransferClf.intercept_

		# 	auditorLabelsDotProduct = strongPred*weakPred > 0
		# 	accDotProduct = accuracy_score((self.target_label[test] == transferLabel[test]), auditorLabelsDotProduct)
		# 	print("accDotProduct", accDotProduct)
				
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
	print("")

	targetFeatureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+targetDataName
	targetFeatureMatrix, targetLabelList = readFeatureLabel(targetFeatureLabelFile)

	targetLabel = np.array(targetLabelList)
	targetFeatureMatrix = np.array(targetFeatureMatrix)

	print('class count of true target labels of all ex:\n', ct(targetLabel))

	fold = 1
	rounds = 100
	al = active_learning(fold, rounds, sourceFeatureMatrix, sourceLabel, targetFeatureMatrix, targetLabel)

	al.run_CV()
