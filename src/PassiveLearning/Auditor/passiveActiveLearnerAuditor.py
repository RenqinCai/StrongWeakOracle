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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

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

	def __init__(self, fold, rounds, target_fn, target_label, transferLabel, auditorLabel):

		self.fold = fold
		self.rounds = rounds

		# self.source_fn = source_fn
		# self.source_label = source_label

		self.target_fn = target_fn
		self.target_label = target_label

		self.m_transferLabel = transferLabel
		self.m_auditorLabel = auditorLabel

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

		
		totalAccListLR = []
		totalAccListThetaDiff = []
		totalAccListThetaStrongWeak = []
		totalAccListThetaStrongThetaWeak = []
		totalAccListThetaWeakWeak = []
		totalAccListTwoLR = []

		totalAccListLRExtra = []
		totalAccListLRExtra1 = []

		totalPrecisionListLR = []
		totalPrecisionListThetaDiff = []
		totalPrecisionListThetaStrongWeak = []
		totalPrecisionListThetaStrongThetaWeak = []
		totalPrecisionListThetaWeakWeak = []
		totalPrecisionListTwoLR = []

		totalPrecisionListLRExtra = []
		totalPrecisionListLRExtra1 = []

		totalRecallListLR = []
		totalRecallListThetaDiff = []
		totalRecallListThetaStrongWeak = []
		totalRecallListThetaStrongThetaWeak = []
		totalRecallListThetaWeakWeak = []
		totalRecallListTwoLR = []

		totalRecallListLRExtra = []
		totalRecallListLRExtra1 = []

		totalACCListStrongLR =[]
		totalACCListWeakLR =[]

		for foldIndex in range(foldNum):
			print("###############%s#########"%foldIndex)
			# print("foldIndex", foldIndex)
			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			sampledTrainNum = len(train)
			sampledTrainNum = 150
			train_sampled = random.sample(train, sampledTrainNum)
			train = train_sampled

			self.m_strongClf = LR(random_state=3)
			self.m_strongClf.fit(self.target_fn[train], self.target_label[train])

			self.m_weakTransferClf = LR(random_state=3)
			self.m_weakTransferClf.fit(self.target_fn[train], self.m_transferLabel[train])

		# print(test)

			predLabel = self.m_strongClf.predict(self.target_fn[test])
			accLR = accuracy_score(predLabel, self.target_label[test])

			print("strong oracle", accLR)
			totalACCListStrongLR.append(accLR)

			predTargetLabels = self.m_weakTransferClf.predict(self.target_fn[test])
			accLR = accuracy_score(predTargetLabels, self.target_label[test])
			print("weak oracle", accLR)
			totalACCListWeakLR.append(accLR)

			"""
			auditor as theta_{strong}X-theta_{weak}X
			"""
		
			coefDiff = self.m_strongClf.coef_ - self.m_weakTransferClf.coef_
			predAuditorLabels = np.dot(self.target_fn[test], coefDiff.reshape(-1, 1))
			# print("intercept", self.m_strongClf.intercept_, self.m_weakClf.intercept_)
			predAuditorLabels += self.m_strongClf.intercept_ - self.m_weakTransferClf.intercept_
			predAuditorLabels = np.abs(predAuditorLabels)

			print(np.min(predAuditorLabels), np.max(predAuditorLabels))
		
			deltaList = [i for i in np.arange(0.0, 25.0, 0.005)]
			maxAcc = 0

			print("pos", sum(self.m_auditorLabel)*1.0/len(self.m_auditorLabel))
			maxDelta = -1
			for delta in deltaList:
		# delta = 0.005
				# if delta %2 == 0:
				# 	print("delta", delta)
			# delta = 3.0
				predAuditorLabelsDelta = (predAuditorLabels < delta)
				# print("predAuditorLabels", predAuditorLabelsDelta)
				
				accLinear = accuracy_score(self.m_auditorLabel[test], predAuditorLabelsDelta)

				if accLinear > maxAcc:
					maxAcc = accLinear
					maxDelta = delta
					# print("sum", np.sum(predAuditorLabelsDelta))
					# print("predAuditorLabels", predAuditorLabelsDelta)
			print(maxDelta, "theta_strong-theta_weak maxAcc\t", maxAcc)
			predAuditorLabelsDelta = (predAuditorLabels < maxDelta)
			accLinear = accuracy_score(self.m_auditorLabel[test], predAuditorLabelsDelta)
			totalAccListThetaDiff.append(accLinear)

			precision = precision_score(self.m_auditorLabel[test], predAuditorLabelsDelta)
			recall = recall_score(self.m_auditorLabel[test], predAuditorLabelsDelta)
			totalPrecisionListThetaDiff.append(precision)
			totalRecallListThetaDiff.append(recall)
			"""
			auditor as logistic regression
			"""

			self.m_auditor = LR(random_state=3)
			self.m_auditor.fit(self.target_fn[train], self.m_auditorLabel[train])
			auditorLabelsLR = self.m_auditor.predict(self.target_fn[test])
			accLR = accuracy_score(self.m_auditorLabel[test], auditorLabelsLR)
			print("accLR", accLR)
			totalAccListLR.append(accLR)

			# print(self.m_auditorLabel[test], auditorLabelsLR)
			precision = precision_score(self.m_auditorLabel[test], auditorLabelsLR)
			recall = recall_score(self.m_auditorLabel[test], auditorLabelsLR)
			totalPrecisionListLR.append(precision)
			totalRecallListLR.append(recall)

			print("precision, recall", precision, recall)
			# exit()

			"""
			auditor as theta_{strong}X*weakLabels
			"""

			strongPred = np.dot(self.target_fn[test], self.m_strongClf.coef_.reshape(-1, 1))+self.m_strongClf.intercept_
			weakPred = np.dot(self.target_fn[test], self.m_weakTransferClf.coef_.reshape(-1, 1))+self.m_weakTransferClf.intercept_

			auditorLabelsDotProduct = strongPred*(2*self.m_transferLabel[test].reshape(-1, 1)-1)>0

			accDotProduct = accuracy_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			print("theta_strong * weakLabels", accDotProduct)
			totalAccListThetaStrongWeak.append(accDotProduct)

			precision = precision_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			recall = recall_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			totalPrecisionListThetaStrongWeak.append(precision)
			totalRecallListThetaStrongWeak.append(recall)

			"""
			auditor as theta_{strong}X*theta_{weak}X
			"""

			auditorLabelsDotProduct = strongPred*weakPred>0

			accDotProduct = accuracy_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			print("theta_strong X * theta_weakX", accDotProduct)
			totalAccListThetaStrongThetaWeak.append(accDotProduct)

			precision = precision_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			recall = recall_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			totalPrecisionListThetaStrongThetaWeak.append(precision)
			totalRecallListThetaStrongThetaWeak.append(recall)

			"""
			auditor as theta_{weak}X*WeakLabels
			"""

			auditorLabelsDotProduct = (2*self.m_transferLabel[test].reshape(-1, 1)-1)*weakPred>0

			accDotProduct = accuracy_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			print("theta_weak X * WeakLabels", accDotProduct)
			totalAccListThetaWeakWeak.append(accDotProduct)

			precision = precision_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			recall = recall_score(self.m_auditorLabel[test], auditorLabelsDotProduct)
			totalPrecisionListThetaWeakWeak.append(precision)
			totalRecallListThetaWeakWeak.append(recall)


			"""
			train two auditors
			"""
			self.m_auditor_pos = LR(random_state=3)
			self.m_auditor_neg = LR(random_state=3)

			posWeakLabelSampleListTrain = []
			negWeakLabelSampleListTrain = []
			for trainIndex in range(len(train)):
				trainID = train[trainIndex]
				if self.m_transferLabel[trainID] == 1.0:
					posWeakLabelSampleListTrain.append(trainID)
				else:
					negWeakLabelSampleListTrain.append(trainID)

			posWeakLabelSampleListTest = []
			negWeakLabelSampleListTest = []
			for testIndex in range(len(test)):
				testID = test[testIndex]
				if self.m_transferLabel[testID] == 1.0:
					posWeakLabelSampleListTest.append(testID)
				else:
					negWeakLabelSampleListTest.append(testID)

			self.m_auditor_pos.fit(self.target_fn[posWeakLabelSampleListTrain], self.m_auditorLabel[posWeakLabelSampleListTrain])
			auditorLabelsLRPos = self.m_auditor_pos.predict(self.target_fn[posWeakLabelSampleListTest])

			self.m_auditor_neg.fit(self.target_fn[negWeakLabelSampleListTrain], self.m_auditorLabel[negWeakLabelSampleListTrain])
			auditorLabelsLRNeg = self.m_auditor_neg.predict(self.target_fn[negWeakLabelSampleListTest]) 

			auditorLabelsTest = np.hstack((self.m_auditorLabel[posWeakLabelSampleListTest], self.m_auditorLabel[negWeakLabelSampleListTest]))
			auditorLabelsPred = np.hstack((auditorLabelsLRPos, auditorLabelsLRNeg))
			# accLR = accuracy_score(self.m_auditorLabel[posWeakLabelSampleListTest], auditorLabelsLRPos)
			# print("accLR pos", accLR)
			# totalAccListLR.append(accLR)
			
			accLR = accuracy_score(auditorLabelsPred, auditorLabelsTest)
			print("accLR pos/neg", accLR)
			totalAccListTwoLR.append(accLR)
			
			# print(self.m_auditorLabel[test], auditorLabelsLR)
			precision = precision_score(auditorLabelsPred, auditorLabelsTest)
			recall = recall_score(auditorLabelsPred, auditorLabelsTest)
			totalPrecisionListTwoLR.append(precision)
			totalRecallListTwoLR.append(recall)

			print("precision, recall", precision, recall)
			
			"""
			auditor as logistic regression use weak label as feature
			"""
			# self.m_weakTransferClf_extra = LR(random_state=3)

			# print(self.target_fn[train].shape, self.m_transferLabel[train].shape)
			# auditorTrainFeature = np.hstack((self.m_transferLabel[train].reshape(-1, 1), self.target_fn[train]))

			# self.m_weakTransferClf_extra.fit(auditorTrainFeature, self.m_transferLabel[train])

			# auditorTestFeature = np.hstack((self.m_transferLabel[test].reshape(-1, 1), self.target_fn[test]))

			# # predStrongLabels = np.dot(self.target_fn[test], self.m_weakTransferClf_extra.coef_ .reshape(-1, 1))+self.m_weakTransferClf_extra.intercept_

			# predStrongLabels = self.m_weakTransferClf_extra.predict(auditorTestFeature)

			# predAuditorLabels = (predStrongLabels == self.m_transferLabel[test])
			# accDotProduct = accuracy_score(self.m_auditorLabel[test], predAuditorLabels)
			# print("add extra logistic weakLabels", accDotProduct)
			# totalAccListLRExtra.append(accDotProduct)


			"""
			auditor as logistic regression use weak label as feature
			"""
			# self.m_weakTransferClf_extra = LR(random_state=3)

			# print(self.target_fn[train].shape, self.m_transferLabel[train].shape)
			# auditorTrainFeature = np.hstack((self.m_transferLabel[train].reshape(-1, 1), self.target_fn[train]))

			# self.m_weakTransferClf_extra.fit(auditorTrainFeature, self.m_transferLabel[train])

			# auditorTestFeature = np.hstack((self.m_transferLabel[test].reshape(-1, 1), self.target_fn[test]))

			# predStrongLabels = np.dot(auditorTestFeature, self.m_weakTransferClf_extra.coef_ .reshape(-1, 1))+self.m_weakTransferClf_extra.intercept_

			# print("min, mean, max", np.min(predStrongLabels), np.mean(predStrongLabels), np.max(predStrongLabels))
			# thresholdList = [i for i in np.arange(0, 4.8, 0.005)]

			# maxAcc = 0

			# # print(self.m_auditorLabel[test])
			# maxThreshold = -1
			# for threshold in thresholdList:
			# 	predAuditorLabels = (np.abs(predStrongLabels) < threshold)
			# 	# print("predAuditorLabels", predAuditorLabelsDelta)
				
			# 	accThreshold = accuracy_score(self.m_auditorLabel[test], predAuditorLabels)

			# 	if accThreshold > maxAcc:
			# 		maxAcc = accThreshold
			# 		maxThreshold = threshold
					
			# print(maxThreshold, "extra threshold\t", maxAcc)
			
			# # predAuditorLabels = (predStrongLabels == self.m_transferLabel[test])
			# # accDotProduct = accuracy_score(self.m_auditorLabel[test], predAuditorLabels)
			# # print("extra threshold logistic weakLabels", accDotProduct)
			# totalAccListLRExtra1.append(maxAcc)

			"""
			auditor as logistic regression PCA
			"""

			# pca = PCA(n_components=10)
			# auditorTrainFeature = pca.fit_transform(self.target_fn[train])

			# poly = PolynomialFeatures(2, include_bias=False)
			# auditorTrainFeature = poly.fit_transform(auditorTrainFeature)

			# # auditorTrainFeature = np.vstack((auditorTrainFeature, self.target_fn[train]))

			# self.m_auditor = LR(random_state=3)
			# self.m_auditor.fit(auditorTrainFeature, (self.target_label[train] == transferLabel[train]))


			# auditorTestFeature = pca.fit_transform(self.target_fn[test])
			# auditorTestFeature = poly.fit_transform(auditorTestFeature)

			# auditorLabelsLR = self.m_auditor.predict(auditorTestFeature)

			# accLR = accuracy_score(self.m_auditorLabel[test], auditorLabelsLR)
			# print("extra accLR", accLR)

			print("***********************************")

		print("totalACCListStrongLR %.3f +/- %.3f"%(np.mean(totalACCListStrongLR), np.sqrt(np.var(totalACCListStrongLR))))

		print("totalACCListWeakLR %.3f +/- %.3f"%(np.mean(totalACCListWeakLR), np.sqrt(np.var(totalACCListWeakLR))))

		print("totalAccListLR %.3f +/- %.3f"%(np.mean(totalAccListLR), np.sqrt(np.var(totalAccListLR))))
		print("totalPrecisionListLR %.3f +/- %.3f"%(np.mean(totalPrecisionListLR), np.sqrt(np.var(totalPrecisionListLR))))
		print("totalRecallListLR %.3f +/- %.3f"%(np.mean(totalRecallListLR), np.sqrt(np.var(totalRecallListLR))))

		print("totalAccListThetaDiff %.3f +/- %.3f"%(np.mean(totalAccListThetaDiff), np.sqrt(np.var(totalAccListThetaDiff))))
		print("totalPrecisionListThetaDiff %.3f +/- %.3f"%(np.mean(totalPrecisionListThetaDiff), np.sqrt(np.var(totalPrecisionListThetaDiff))))
		print("totalRecallListThetaDiff %.3f +/- %.3f"%(np.mean(totalRecallListThetaDiff), np.sqrt(np.var(totalRecallListThetaDiff))))

		print("totalAccListThetaStrongWeak %.3f +/- %.3f"%(np.mean(totalAccListThetaStrongWeak), np.sqrt(np.var(totalAccListThetaStrongWeak))))
		print("totalPrecisionListThetaStrongWeak %.3f +/- %.3f"%(np.mean(totalPrecisionListThetaStrongWeak), np.sqrt(np.var(totalPrecisionListThetaStrongWeak))))
		print("totalRecallListThetaStrongWeak %.3f +/- %.3f"%(np.mean(totalRecallListThetaStrongWeak), np.sqrt(np.var(totalRecallListThetaStrongWeak))))

		print("totalAccListThetaStrongThetaWeak %.3f +/- %.3f"%( np.mean(totalAccListThetaStrongThetaWeak), np.sqrt(np.var(totalAccListThetaStrongThetaWeak))))
		print("totalPrecisionListThetaStrongThetaWeak %.3f +/- %.3f"%( np.mean(totalPrecisionListThetaStrongThetaWeak), np.sqrt(np.var(totalPrecisionListThetaStrongThetaWeak))))
		print("totalRecallListThetaStrongThetaWeak %.3f +/- %.3f"%( np.mean(totalRecallListThetaStrongThetaWeak), np.sqrt(np.var(totalRecallListThetaStrongThetaWeak))))

		print("totalAccListThetaWeakWeak %.3f +/- %.3f"%( np.mean(totalAccListThetaWeakWeak), np.sqrt(np.var(totalAccListThetaWeakWeak))))
		print("totalPrecisionListThetaWeakWeak %.3f +/- %.3f"%( np.mean(totalPrecisionListThetaWeakWeak), np.sqrt(np.var(totalPrecisionListThetaWeakWeak))))
		print("totalRecallListThetaWeakWeak %.3f +/- %.3f"%( np.mean(totalRecallListThetaWeakWeak), np.sqrt(np.var(totalRecallListThetaWeakWeak))))

		print("totalAccListTwoLR %.3f +/- %.3f"%(np.mean(totalAccListTwoLR), np.sqrt(np.var(totalAccListTwoLR))))
		print("totalPrecisionListTwoLR %.3f +/- %.3f"%(np.mean(totalPrecisionListTwoLR), np.sqrt(np.var(totalPrecisionListTwoLR))))
		print("totalRecallListTwoLR %.3f +/- %.3f"%(np.mean(totalRecallListTwoLR), np.sqrt(np.var(totalRecallListTwoLR))))


		# print("totalAccListLRExtra %.3f +/- %.3f"%(np.mean(totalAccListLRExtra), np.sqrt(np.var(totalAccListLRExtra))))

		# print("totalAccListLRExtra1 %.3f +/- %.3f"%(np.mean(totalAccListLRExtra1), np.sqrt(np.var(totalAccListLRExtra1))))

				
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
		auditorLabelArray = np.array(auditorLabelList)

		print('class count of target labels of all ex:\n', ct(labelArray))

		fold = 1
		rounds = 100
		al = active_learning(fold, rounds, featureMatrix, labelArray, transferLabelArray, auditorLabelArray)

		al.run_CV()

	if dataName == "20News":
		featureFile = "../../../dataset/20News/baseball"
		labelIndex = 0
		featureMatrix0, labelList0 = readFeatureFile(featureFile, labelIndex)
		print(len(labelList0))

		featureFile = "../../../dataset/20News/politicsMisc"
		labelIndex = 1
		featureMatrix1, labelList1 = readFeatureFile(featureFile, labelIndex)
		print(len(labelList1))
		
		featureMatrix = featureMatrix0+featureMatrix1
		labelList = labelList0+labelList1

		transferLabelFile0 = "../../../dataset/20News/transferLabel_hockey_religionMisc--baseball_politicsMisc.txt"
		auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile0)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(trueLabelList)

		transferLabelArray = np.array(transferLabelList)
		auditorLabelArray = np.array(auditorLabelList)

		initialExList = []
		initialExList = [[42, 438, 9],  [246, 365, 299], [145, 77, 45], [353, 369, 299], [483, 337, 27], [489, 468, 122],  [360, 44, 412], [263, 284, 453], [449, 3, 261], [244, 200, 47]]

		fold = 10
		rounds = 150


		al = active_learning(fold, rounds, featureMatrix, labelArray, transferLabelArray, auditorLabelArray)

		al.run_CV()

	if dataName == "simulation":
		featureLabelFile = "../../../dataset/synthetic/simulatedFeatureLabel_500_20_2.txt"

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		transferLabelFile0 = "../../../dataset/synthetic/simulatedTransferLabel_500_20_2.txt"
		auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile0)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(trueLabelList)

		transferLabelArray = np.array(transferLabelList)
		auditorLabelArray = np.array(auditorLabelList)

		TPWeakLabels = transferLabelArray*2-1.0 == auditorLabelArray
		TPWeakLabelNum = np.sum(TPWeakLabels)
		weakLabelPrecision = TPWeakLabelNum*1.0/np.sum(transferLabelArray)
		print("weakLabelPrecision", weakLabelPrecision)

		TPWeakLabels = transferLabelArray*2-1.0 == auditorLabelArray
		TPWeakLabelNum = np.sum(TPWeakLabels)
		weakLabelRecall = TPWeakLabelNum*1.0/np.sum(labelArray)
		print("weakLabelRecall", weakLabelRecall)

		weakLabelAcc = np.sum(auditorLabelArray)*1.0/len(auditorLabelArray)
		print("weakLabelAcc", weakLabelAcc)

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, labelArray, transferLabelArray, auditorLabelArray)

		al.run_CV()
