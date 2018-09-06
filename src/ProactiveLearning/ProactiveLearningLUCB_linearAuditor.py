"""
proactive learning LUCB with random initialization, judge classifier to judge whether transfer learning is correct. 
"""
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
import numpy as np

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

modelName = "LUCB_linearAuditor_"+dataName
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp
# random.seed(3)

def get_name_features(names):

		name = []
		for i in names:
			s = re.findall('(?i)[a-z]{2,}',i)
			name.append(' '.join(s))

		cv = CV(analyzer='char_wb', ngram_range=(3,4))
		fn = cv.fit_transform(name).toarray()

		return fn

def sigmoid(x):
  	  return (1 / (1 + np.exp(-x)))

class _ProactiveLearning:

	def __init__(self, fold, rounds, featureMatrix, label, transferLabel):

		self.m_fold = fold
		self.m_rounds = rounds

		self.m_targetNameFeature = np.array(featureMatrix)
		self.m_targetLabel = np.array(label)

		self.m_transferLabel = np.array(transferLabel)

		self.m_randomForest = 0
		self.m_tao = 0

		self.m_alpha = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.002 ##0.05

		self.m_judgeClassifier = 0
		self.m_clf = 0

		self.m_weakLabeledIDList = []
		self.m_strongLabeledIDList = []
		self.m_unlabeledIDList = []

	def select_example(self, unlabeled_list):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)

		instanceLabelIndexMap = {} ##id:labelIndex
		labelDensityMap = {} ###labelIndex:densityRatio

		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			# print("unlabeledId\t", unlabeledId)
			labelPredictProb = self.m_clf.predict_proba(self.m_targetNameFeature[unlabeledId].reshape(1, -1))[0]

			labelIndexMap = {} ##labelIndex: labelProb
			labelNum = len(labelPredictProb)
			for labelIndex in range(labelNum):
				labelIndexMap.setdefault(labelIndex, labelPredictProb[labelIndex])

			sortedLabelIndexList = sorted(labelIndexMap, key=labelIndexMap.__getitem__, reverse=True)

			maxLabelIndex = sortedLabelIndexList[0]
			subMaxLabelIndex = sortedLabelIndexList[1]

			selectCB = self.get_select_confidence_bound(unlabeledId)

			coefDiff = 0
			# if labelNum == 2:
			# 	# coefDiff = np.dot(self.clf.coef_, self.fn[unlabeledId])
			# 	print("error")
			# 	print(debug)
			# else:
			# 	maxCoef = self.m_clf.coef_[maxLabelIndex]
			# 	subMaxCoef = self.m_clf.coef_[subMaxLabelIndex]
			# 	coefDiff = np.dot(maxCoef, self.m_targetNameFeature[unlabeledId])-np.dot(subMaxCoef, self.m_targetNameFeature[unlabeledId])
			
			coefDiff = np.abs(np.dot(self.m_clf.coef_, self.m_targetNameFeature[unlabeledId]))

			LCB = coefDiff-2*0.002*selectCB
			idScore = 1-LCB
			unlabeledIdScoreMap[unlabeledId] = idScore

			# if unlabeledId not in instanceLabelIndexMap:
			# 	instanceLabelIndexMap.setdefault(unlabeledId, maxLabelIndex)

			# if maxLabelIndex not in labelDensityMap:
			# 	labelDensityMap.setdefault(maxLabelIndex, 0.0)

			# labelDensityMap[maxLabelIndex] += 1.0

			# print(labelPredictProb)
			# sortedLabelPredictProb = sorted(labelPredictProb)
			# sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)
			# print(sortedLabelPredictProb)
			# maxLabelPredictProb = sortedLabelPredictProb[0]
			# subMaxLabelPredictProb = sortedLabelPredictProb[1]
			# print("maxLabelPredictProb\t", maxLabelPredictProb)
			# idScore = 1-(maxLabelPredictProb-subMaxLabelPredictProb)
			# print("idScore\t", idScore)
			# unlabeledIdScoreMap[unlabeledId] = idScore

		# for maxLabelIndex in labelDensityMap:
		# 	labelDensityMap[maxLabelIndex] /= unlabeledIdNum

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		# print("****************")
		# print(sortedUnlabeledIdList[:10])

		# labelIndexTop10IDList = []
		# labelIndexDensityList = []

		# for instanceIndex in range(10):
		# 	maxLabelIndex = instanceLabelIndexMap[sortedUnlabeledIdList[instanceIndex]]
		# 	labelIndexTop10IDList.append(maxLabelIndex)
		# 	labelIndexDensityList.append(labelDensityMap[maxLabelIndex])
		# print("predict labelIndex", labelIndexTop10IDList, labelIndexDensityList)
		# print("===============")

		return sortedUnlabeledIdList[0]

	def get_pred_acc(self, targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter):

		# targetNameFeatureTrain = self.m_targetNameFeature[labeledIdList]
		# targetLabelTrain = self.m_targetLabel[labeledIdList]
		
		self.m_clf.fit(targetNameFeatureIter, targetLabelIter)
		targetLabelPreds = self.m_clf.predict(targetNameFeatureTest)

		acc = accuracy_score(targetLabelTest, targetLabelPreds)
		# print("acc\t", acc)
		# print debug
		return acc

	def get_base_learners(self):
		self.m_randomForest = RFC(n_estimators=100, criterion='entropy', random_state=3)

		self.m_randomForest.fit(self.m_sourceDataFeature, self.m_sourceLabel)

	def init_confidence_bound(self, featureDim, labeledExList, unlabeledExList):
		self.m_weakLabeledIDList = []
		self.m_strongLabeledIDList = labeledExList
		# self.m_unlabeledIDList = unlabeledExList

		self.m_selectA = self.m_lambda*np.identity(featureDim)
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

		self.m_judgeA = self.m_lambda*np.identity(featureDim)
		self.m_judgeAInv = np.linalg.inv(self.m_judgeA)

	def update_select_confidence_bound(self, exId):
		# print("updating select cb", exId)
		# self.m_unlabeledIDList.remove(exId)
		self.m_selectA += np.outer(self.m_targetNameFeature[exId], self.m_targetNameFeature[exId])
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

	def update_judge_confidence_bound(self, exId):
		# print("updating judge cb", exId)
		self.m_judgeA += np.outer(self.m_targetNameFeature[exId], self.m_targetNameFeature[exId])
		self.m_judgeAInv = np.linalg.inv(self.m_judgeA)

	def get_select_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.m_targetNameFeature[exId], self.m_selectAInv), self.m_targetNameFeature[exId]))

		return CB

	def get_judge_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.m_targetNameFeature[exId], self.m_judgeAInv), self.m_targetNameFeature[exId]))

		return CB

	def get_judgeClassifier_prob(self, judgeParam, feature, CB, judgeRound):
		rawProb = np.dot(judgeParam, np.transpose(feature))
		judgeProbThreshold = 0.65

		cbProb = sigmoid(rawProb-self.m_cbRate*CB)
		# print("cbProb\t", cbProb)
		if cbProb > judgeProbThreshold:
			return True
		else:
			return False

	def get_transfer_flag(self, transferFeatureList, transferFlagList, exId, judgeRound):
		# predLabel = self.m_randomForest.predict(self.m_targetDataFeature[exId].reshape(1, -1))[0]
		predLabel = self.m_transferLabel[exId]

		if len(np.unique(transferFlagList)) > 1:
			self.m_judgeClassifier.fit(np.array(transferFeatureList), np.array(transferFlagList))
		else:
			return False, predLabel

		CB = self.get_judge_confidence_bound(exId)

		transferFlag = self.get_judgeClassifier_prob(self.m_judgeClassifier.coef_, self.m_targetNameFeature[exId].reshape(1, -1), CB, judgeRound)

		if transferFlag:
			return True, predLabel
		else:
			return False, predLabel

	def getAuditorMetric(self, transferFeatureList, transferFlagList, transferFeatureTest, transferLabelTest, targetLabelTest):
		acc = 0.0
		if len(np.unique(transferFlagList)) > 1:
			self.m_judgeClassifier.fit(np.array(transferFeatureList), np.array(transferFlagList))

			auditorLabelTest = (transferLabelTest==targetLabelTest)

			predictAuditorLabelTest = self.m_judgeClassifier.predict(transferFeatureTest)

			acc = accuracy_score(auditorLabelTest, predictAuditorLabelTest)

		return acc
		
	def pretrainSelectInit(self, train):

		posTrain = []
		negTrain = []

		for trainIndex in range(len(train)):
			if self.m_targetLabel[train[trainIndex]] == 1.0:
				posTrain.append(train[trainIndex])
			else:
				negTrain.append(train[trainIndex])

		initList = []

		random.seed(10)

		initList += random.sample(posTrain, 2)
		initList += random.sample(negTrain, 1)

		# initList += posTrain[:2]
		# initList += negTrain[:1]

		print("initList", initList)

		return initList

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.m_targetLabel)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		totalTransferNumList = []
		np.random.seed(3)
		np.random.shuffle(indexList)

		foldNum = 10
		foldInstanceNum = int(totalInstanceNum*1.0/foldNum)
		foldInstanceList = []

		for foldIndex in range(foldNum-1):
			foldIndexInstanceList = indexList[foldIndex*foldInstanceNum:(foldIndex+1)*foldInstanceNum]
			foldInstanceList.append(foldIndexInstanceList)

		foldIndexInstanceList = indexList[foldInstanceNum*(foldNum-1):]
		foldInstanceList.append(foldIndexInstanceList)
		# kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		# random.seed(3)
		totalAccList = [[] for i in range(10)]
		humanAccList = [[] for i in range(10)]

		# self.get_base_learners()

		correctTransferRatioList = []
		totalTransferNumList = []
		correctUntransferRatioList = []

		totalAuditorPrecisionList = []
		totalAuditorRecallList = []
		totalAuditorAccList = []

		for foldIndex in range(foldNum):
			
			# self.clf = LinearSVC(random_state=3)

			self.m_clf = LR(random_state=3)
			self.m_judgeClassifier = LR(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			trainNum = int(totalInstanceNum*0.9)

			targetNameFeatureTrain = self.m_targetNameFeature[train]
			targetLabelTrain = self.m_targetLabel[train]
			# targetDataFeatureTrain = self.m_targetDataFeature[train]

			targetNameFeatureTest = self.m_targetNameFeature[test]
			targetLabelTest = self.m_targetLabel[test]

			transferLabelTest = self.m_transferLabel[test]
			# targetDataFeatureTest = self.m_targetDataFeature[test]

			# sourceUniqueClass = np.unique(self.m_sourceLabel)

			initExList = []
			initExList = self.pretrainSelectInit(train)
			# random.seed(101)
			# initExList = random.sample(train, 3)

			targetNameFeatureInit = self.m_targetNameFeature[initExList]
			targetLabelInit = self.m_targetLabel[initExList]

			print("initExList\t", initExList, targetLabelInit)

			queryIter = 0
			labeledExList = []
			unlabeledExList = []
			###labeled index
			labeledExList.extend(initExList)
			unlabeledExList = list(set(train)-set(labeledExList))

			activeLabelNum = 3.0
			transferLabelNum = 0.0
			transferFeatureList = []
			transferFlagList = []

			featureDim = len(targetNameFeatureTrain[0])
			self.init_confidence_bound(featureDim, labeledExList, unlabeledExList)

			targetNameFeatureIter = targetNameFeatureInit
			targetLabelIter = targetLabelInit

			correctTransferLabelNum = 0.0
			wrongTransferLabelNum = 0.0
			correctUntransferLabelNum = 0.0
			wrongUntransferLabelNum = 0.0

			# auditorPrecisionList = []
			# auditorRecallList = []
			auditorAccList = []

			while activeLabelNum < rounds:

				# targetNameFeatureIter = self.m_targetNameFeature[labeledExList]
				# targetLabelIter = self.m_targetLabel[labeledExList]

				self.m_clf.fit(targetNameFeatureIter, targetLabelIter) 

				exId = self.select_example(unlabeledExList) 
				self.update_select_confidence_bound(exId)

				# print(idx)
				activeLabelFlag = False
				transferLabelFlag, transferLabel = self.get_transfer_flag(transferFeatureList, transferFlagList, exId, activeLabelNum)

				exLabel = -1
				if transferLabelFlag:
					self.m_weakLabeledIDList.append(exId)
					print("queryIter\t", queryIter)
					transferLabelNum += 1.0
					activeLabelFlag = False
					
					exLabel = transferLabel
					targetNameFeatureIter = np.vstack((targetNameFeatureIter, self.m_targetNameFeature[exId]))
					targetLabelIter = np.hstack((targetLabelIter, exLabel))
					# targetNameFeatureIter.append(self.m_targetNameFeature[exId])
					# targetLabelIter.append(exLabel)

					if exLabel == self.m_targetLabel[exId]:
						correctTransferLabelNum += 1.0
					else:
						wrongTransferLabelNum += 1.0
						print("query iteration", queryIter, "error transfer label\t", exLabel, "true label", self.m_targetLabel[exId])
				else:
					self.m_strongLabeledIDList.append(exId)
					self.update_judge_confidence_bound(exId)
					activeLabelNum += 1.0
					activeLabelFlag = True

					exLabel = self.m_targetLabel[exId]
					targetNameFeatureIter = np.vstack((targetNameFeatureIter, self.m_targetNameFeature[exId]))
					targetLabelIter = np.hstack((targetLabelIter, exLabel))
					# targetNameFeatureIter.append(self.m_targetNameFeature[exId])
					# targetLabelIter.append(exLabel)

					if transferLabel == exLabel:
						correctUntransferLabelNum += 1.0
						transferFlagList.append(1.0)
						transferFeatureList.append(self.m_targetNameFeature[exId])
					else:
						wrongUntransferLabelNum += 1.0
						transferFlagList.append(0.0)
						transferFeatureList.append(self.m_targetNameFeature[exId])


					# auditorPrecision = 0.0
					# if correctTransferLabelNum+wrongTransferLabelNum > 0.0:
					# 	auditorPrecision = correctTransferLabelNum*1.0/(correctTransferLabelNum+wrongTransferLabelNum)

					auditorAcc = self.getAuditorMetric(transferFeatureList, transferFlagList, targetNameFeatureTest, transferLabelTest, targetLabelTest)
					print("auditorAcc", auditorAcc)

					# auditorRecall = 0.0
					# if correctTransferLabelNum+correctUntransferLabelNum > 0.0:
					# 	auditorRecall = correctTransferLabelNum*1.0/(correctTransferLabelNum+correctUntransferLabelNum)

					# auditorAcc = (correctTransferLabelNum+wrongUntransferLabelNum)*1.0/(correctTransferLabelNum+wrongUntransferLabelNum+correctUntransferLabelNum+wrongTransferLabelNum)

					# auditorPrecisionList.append(auditorPrecision)
					# auditorRecallList.append(auditorRecall)
					auditorAccList.append(auditorAcc)

				labeledExList.append(exId)
				unlabeledExList.remove(exId)

				acc = self.get_pred_acc(targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter)
				totalAccList[cvIter].append(acc)
				if activeLabelFlag:
					humanAccList[cvIter].append(acc)
				queryIter += 1

			# totalAuditorPrecisionList.append(auditorPrecisionList)
			# totalAuditorRecallList.append(auditorRecallList)
			totalAuditorAccList.append(auditorAccList)

			# correctUntransferRatio = correctUntransferLabelNum*1.0
			# correctUntransferRatioList.append(correctUntransferRatio)
			# print("correctUntransferRatio\t", correctUntransferRatio)

			# correctTransferRatio = correctTransferLabelNum*1.0/transferLabelNum
			# print("transferLabelNum\t", transferLabelNum, "correct transfer ratio\t", correctTransferRatio)
			# correctTransferRatioList.append(correctTransferRatio)
			transferLabelNum = len(self.m_weakLabeledIDList)
			totalTransferNumList.append(transferLabelNum)

			cvIter += 1      
		
		print("transfer num\t", np.mean(totalTransferNumList), np.sqrt(np.var(totalTransferNumList)))
		# print("correct ratio\t", np.mean(correctTransferRatioList), np.sqrt(np.var(correctTransferRatioList)))
		# print("untransfer correct ratio\t", np.mean(correctUntransferRatioList), np.sqrt(np.var(correctUntransferRatioList)))

		# AuditorPrecisionFile = modelVersion+"_auditor_precision.txt"
		# writeFile(totalAuditorPrecisionList, AuditorPrecisionFile)

		# AuditorRecallFile = modelVersion+"_auditor_recall.txt"
		# writeFile(totalAuditorRecallList, AuditorRecallFile)

		AuditorAccFile = modelVersion+"_auditor_acc.txt"
		writeFile(totalAuditorAccList, AuditorAccFile)

		totalACCFile = modelVersion+"_acc.txt"
		writeFile(totalAccList, totalACCFile)

		humanACCFile = modelVersion+"_human_acc.txt"
		writeFile(humanAccList, humanACCFile)

def writeFile(valueList, fileName):
	f = open(fileName, "w")
	for i in range(10):
		num4Iter = len(valueList[i])
		for j in range(num4Iter):
			f.write(str(valueList[i][j])+"\t")
		f.write("\n")
	f.close()

def data_analysis(sourceLabelList, targetLabelList):
	sourceLabelNum = len(sourceLabelList)
	sourceLabelMap = {}
	for sourceLabelIndex in range(sourceLabelNum):
		sourceLabelVal = int(sourceLabelList[sourceLabelIndex])

		if sourceLabelVal not in sourceLabelMap.keys():
			sourceLabelMap.setdefault(sourceLabelVal, 0.0)
		sourceLabelMap[sourceLabelVal] += 1.0

	sortedLabelList = sorted(sourceLabelMap.keys())

	# sortedLabelList = sorted(sourceLabelMap, key=sourceLabelMap.__getitem__, reverse=True)

	print("====source label distribution====")
	for label in sortedLabelList:
		print(label, sourceLabelMap[label], "--",)

	print("\n")
	targetLabelNum = len(targetLabelList)
	targetLabelMap = {}
	for targetLabelIndex in range(targetLabelNum):
		targetLabelVal = int(targetLabelList[targetLabelIndex])

		if targetLabelVal not in targetLabelMap.keys():
			targetLabelMap.setdefault(targetLabelVal, 0.0)
		targetLabelMap[targetLabelVal] += 1.0
	
	sortedLabelList = sorted(targetLabelMap.keys())

	# sortedLabelList = sorted(targetLabelMap, key=targetLabelMap.__getitem__, reverse=True)

	print("====target label distribution====")
	for label in sortedLabelList:
		print(label, targetLabelMap[label],"--",)
	print("\n")

def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	transferLabelList = []
	auditorLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		auditorLabelList.append(float(line[0]))
		transferLabelList.append(float(line[1]))

	f.close()

	return auditorLabelList, transferLabelList 

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

if __name__ == "__main__":

	# raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('./selectedNameFeature4Label_5types.txt').readlines()]

	# raw_pt = [i.strip().split('\t')[:-1] for i in open().readlines()]

	# f = open('../simulatedFeatureLabel_500_100_2.txt')
	featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName
	featureMatrix, labelList = readFeatureLabel(featureLabelFile)

	featureMatrix = np.array(featureMatrix)
	labelArray = np.array(labelList)

	transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
	auditorLabelList, transferLabelList = readTransferLabel(transferLabelFile)

	transferLabelArray = np.array(transferLabelList)
	print("number of types", len(set(labelArray)))
	print 'class count of true labels of all ex:\n', ct(transferLabelArray)

	fold = 10
	rounds = 150

	al = _ProactiveLearning(fold, rounds, featureMatrix, labelArray, transferLabelArray)

	al.run_CV()
