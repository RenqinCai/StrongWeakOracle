"""
proactive learning with random initialization, an auditor to judge whether transfer learning is correct. Reuse previous wrong transferred instances
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

"""
electronics, sensor_rice
"""

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

	def __init__(self, fold, rounds, featureMatrix, label, transferLabel, category, multipleClass):

		self.m_fold = fold
		self.m_rounds = rounds

		self.m_category = category
		print("category", category)
		self.m_multipleClass = multipleClass
		print("multipleClass", multipleClass)

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

		self.m_auditor = 0
		self.m_clf = 0

		self.m_weakLabeledIDList = []
		self.m_strongLabeledIDList = []
		self.m_unlabeledIDList = []

	def select_example(self, unlabeled_list, auditorScoreFlag):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)

		instanceLabelIndexMap = {} ##id:labelIndex
		labelDensityMap = {} ###labelIndex:densityRatio

		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			
			# idScore = self.getLUCB(unlabeledId)
			idScoreClassifier = self.getClassifierMargin(unlabeledId)

			idScoreAuditor = 0.0
			if auditorScoreFlag:
				idScoreAuditor = self.getAuditorMargin(unlabeledId)

			## fine tune
			scoreWeightClassifier = 0.0
			unlabeledIdScoreMap[unlabeledId] = scoreWeightClassifier*idScoreClassifier + (1-scoreWeightClassifier)*idScoreAuditor

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def getClassifierMargin(self, unlabeledId):
		labelPredictProb = self.m_clf.predict_proba(self.m_targetNameFeature[unlabeledId].reshape(1, -1))[0]

		labelProbMap = {} ##labelIndex: labelProb
		labelNum = len(labelPredictProb)
		for labelIndex in range(labelNum):
			labelProbMap.setdefault(labelIndex, labelPredictProb[labelIndex])

		sortedLabelIndexList = sorted(labelProbMap, key=labelProbMap.__getitem__, reverse=True)

		maxLabelIndex = sortedLabelIndexList[0]
		subMaxLabelIndex = sortedLabelIndexList[1]

		maxLabelProb = labelProbMap[maxLabelIndex]
		subMaxLabelProb = labelProbMap[subMaxLabelIndex]

		margin = maxLabelProb-subMaxLabelProb

		margin = 0 - margin

		return margin

	def getAuditorMargin(self, unlabeledId):
		
		labelPredictProb = self.m_auditor.predict_proba(self.m_targetNameFeature[unlabeledId].reshape(1, -1))[0]

		labelProbMap = {} ##labelIndex: labelProb
		labelNum = len(labelPredictProb)
		for labelIndex in range(labelNum):
			labelProbMap.setdefault(labelIndex, labelPredictProb[labelIndex])

		sortedLabelIndexList = sorted(labelProbMap, key=labelProbMap.__getitem__, reverse=True)

		maxLabelIndex = sortedLabelIndexList[0]
		subMaxLabelIndex = sortedLabelIndexList[1]

		maxLabelProb = labelProbMap[maxLabelIndex]
		subMaxLabelProb = labelProbMap[subMaxLabelIndex]

		margin = maxLabelProb-subMaxLabelProb

		margin = 0 - margin

		return margin

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
		self.m_unlabeledIDList = unlabeledExList

		# self.m_selectA = self.m_lambda*np.identity(featureDim)
		# self.m_selectAInv = np.linalg.inv(self.m_selectA)

		# self.m_judgeA = self.m_lambda*np.identity(featureDim)
		# self.m_judgeAInv = np.linalg.inv(self.m_judgeA)

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

	def get_judgeClassifier_prob(self, judgeParam, intercept, feature, CB):
		rawProb = np.dot(judgeParam, np.transpose(feature))+intercept
		judgeProbThreshold = 0.8

		cbProb = sigmoid(rawProb-self.m_cbRate*CB)
		# print("cbProb\t", cbProb)
		if cbProb > judgeProbThreshold:
			return True
		else:
			return False

	def get_transfer_flag(self, transferFeatureList, transferFlagList, exId):
		# predLabel = self.m_randomForest.predict(self.m_targetDataFeature[exId].reshape(1, -1))[0]
		predLabel = self.m_transferLabel[exId]

		if len(np.unique(transferFlagList)) > 1:
			self.m_auditor.fit(np.array(transferFeatureList), np.array(transferFlagList))
		else:
			return False, predLabel

		auditorPosProb = self.m_auditor.predict_proba(self.m_targetNameFeature[exId].reshape(1, -1))[0][1]

		transferFlag = False
		if auditorPosProb > 0.8:
			transferFlag = True
		else:
			transferFlag = False

		# transferFlag = self.get_judgeClassifier_prob(self.m_auditor.coef_, self.m_auditor.intercept_, self.m_targetNameFeature[exId].reshape(1, -1), CB)

		if transferFlag:
			return True, predLabel
		else:
			return False, predLabel

	def getAuditorMetric(self, transferFeatureList, transferFlagList, transferFeatureTest, transferLabelTest, targetLabelTest):
		acc = 0.0
		if len(np.unique(transferFlagList)) > 1:
			self.m_auditor.fit(np.array(transferFeatureList), np.array(transferFlagList))

			auditorLabelTest = (transferLabelTest==targetLabelTest)

			predictAuditorLabelTest = self.m_auditor.predict(transferFeatureTest)

			acc = accuracy_score(auditorLabelTest, predictAuditorLabelTest)

		return acc
		
	def addExtraWeakLabels(self, transferFeatureList, transferFlagList, targetNameFeatureTest, transferLabelTest, targetLabelTest):
		extraIDList = self.m_unlabeledIDList
		extraNum = len(extraIDList)

		weakNameFeature = []
		weakLabel = []
		weakIDList = []

		strongNameFeature = self.m_targetNameFeature[self.m_strongLabeledIDList]
		strongLabel = self.m_targetLabel[self.m_strongLabeledIDList]
		# print("extra Num", extraNum)

		correctTransferLabelNum = 0.0
		transferNum = 0

		for extraIndex in range(extraNum):
			extraID = extraIDList[extraIndex]

			## transferFeatureList, transferFlagList are all about strong oracle
			transferFlag, predLabel = self.get_transfer_flag(transferFeatureList, transferFlagList, extraID)

			if transferFlag:
			# 	if predLabel == self.m_targetLabel[extraID]:
			# 		correctTransferLabelNum += 1.0
			# 	transferNum += 1.0
				weakIDList.append(extraID)
				weakLabel.append(predLabel)
			# else:
			# 	extraLabel.append(self.m_targetLabel[extraID])
		# print("extra auditor acc", correctTransferLabelNum*1.0/extraNum, correctTransferLabelNum*1.0/transferNum)
		weakNameFeature = self.m_targetNameFeature[weakIDList]
		weakLabel = np.array(weakLabel)

		print("weak num", len(weakLabel))

		trainNameFeature = np.vstack((strongNameFeature, weakNameFeature))
		trainLabel = np.hstack((strongLabel, weakLabel))

		### clf updated
		trainAcc = self.get_pred_acc(targetNameFeatureTest, targetLabelTest, trainNameFeature, trainLabel)

		print("trainAcc", trainAcc)
		return trainAcc

	def setInitialExList(self, initialExList):
		self.m_initialExList = initialExList

	def pretrainSelectInit(self, train, foldIndex):
		
		initList = self.m_initialExList[foldIndex]
		
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
		totalExtraAccList = []
		# self.get_base_learners()

		correctTransferRatioList = []
		totalTransferNumList = []
		correctTransferLabelNumList = []
		correctUntransferRatioList = []

		totalAuditorPrecisionList = []
		totalAuditorRecallList = []
		totalAuditorAccList = []


		for foldIndex in range(foldNum):
		
			if self.m_multipleClass:
				self.m_clf = LR(multi_class="multinomial", solver='lbfgs', random_state=3)
			else:
				self.m_clf = LR(random_state=3)
			self.m_auditor = LR(random_state=3)

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
			initExList = self.pretrainSelectInit(train, foldIndex)

			targetNameFeatureInit = self.m_targetNameFeature[initExList]
			targetLabelInit = self.m_targetLabel[initExList]

			transferLabelInit = self.m_transferLabel[initExList]

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
			extraAccList = []

			self.m_clf.fit(targetNameFeatureInit, targetLabelInit)

			# targetAuditorLabelInit = (targetLabelInit==transferLabelInit)
			for exId in initExList:
				if self.m_targetLabel[exId] == self.m_transferLabel[exId]:
					transferFlagList.append(1.0)
				else:
					transferFlagList.append(0.0)

				transferFeatureList.append(self.m_targetNameFeature[exId])

			auditorScoreFlag = False
			if len(np.unique(transferFlagList)) > 1:
				self.m_auditor.fit(np.array(transferFeatureList), np.array(transferFlagList))
				auditorScoreFlag = True

			while activeLabelNum < rounds:

				exId = self.select_example(unlabeledExList, auditorScoreFlag) 
				
				exLabel = -1
				
				self.m_strongLabeledIDList.append(exId)
				# self.update_select_confidence_bound(exId)
				# self.update_judge_confidence_bound(exId)
				activeLabelNum += 1.0
				activeLabelFlag = True

				exLabel = self.m_targetLabel[exId]
				
				transferLabel = self.m_transferLabel[exId]
				if transferLabel == exLabel:
					# correctUntransferLabelNum += 1.0
					transferFlagList.append(1.0)
					transferFeatureList.append(self.m_targetNameFeature[exId])
				else:
					# wrongUntransferLabelNum += 1.0
					transferFlagList.append(0.0)
					transferFeatureList.append(self.m_targetNameFeature[exId])

					# auditorPrecision = 0.0
					# if correctTransferLabelNum+wrongTransferLabelNum > 0.0:
					# 	auditorPrecision = correctTransferLabelNum*1.0/(correctTransferLabelNum+wrongTransferLabelNum)

				auditorAcc = self.getAuditorMetric(transferFeatureList, transferFlagList, targetNameFeatureTest, transferLabelTest, targetLabelTest)
				# print("auditorAcc", auditorAcc)
				auditorAccList.append(auditorAcc)

				labeledExList.append(exId)
				unlabeledExList.remove(exId)

				# acc = self.get_pred_acc(targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter)
				# totalAccList[cvIter].append(acc)
				extraAcc = self.addExtraWeakLabels(transferFeatureList, transferFlagList, targetNameFeatureTest, transferLabelTest, targetLabelTest)
				extraAccList.append(extraAcc)
					# humanAccList[cvIter].append(acc)
				queryIter += 1

			# totalAuditorPrecisionList.append(auditorPrecisionList)
			# totalAuditorRecallList.append(auditorRecallList)
			totalAuditorAccList.append(auditorAccList)
			totalExtraAccList.append(extraAccList)
			

			cvIter += 1      
		
		# print("transfer num\t", np.mean(totalTransferNumList), np.sqrt(np.var(totalTransferNumList)))

		# print("extraList", extraAccList, np.mean(extraAccList), np.sqrt(np.var(extraAccList)))
		# print("correct ratio\t", np.mean(correctTransferRatioList), np.sqrt(np.var(correctTransferRatioList)))
		# print("untransfer correct ratio\t", np.mean(correctUntransferRatioList), np.sqrt(np.var(correctUntransferRatioList)))

		# AuditorPrecisionFile = modelVersion+"_auditor_precision.txt"
		# writeFile(totalAuditorPrecisionList, AuditorPrecisionFile)

		# AuditorRecallFile = modelVersion+"_auditor_recall.txt"
		# writeFile(totalAuditorRecallList, AuditorRecallFile)

		AuditorAccFile = modelVersion+"_auditor_acc.txt"
		writeFile(totalAuditorAccList, AuditorAccFile)

		# totalACCFile = modelVersion+"_acc.txt"
		# writeFile(totalAccList, totalACCFile)

		# humanACCFile = modelVersion+"_human_acc.txt"
		# writeFile(humanAccList, humanACCFile)

		extraACCFile = modelVersion+"_extra_acc.txt"
		writeFile(totalExtraAccList, extraACCFile)

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
	dataName = "rice_sensor"

	modelName = "AAWS_extra_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

	modelVersion = modelName+"_"+timeStamp

	"""
	 	processedKitchenElectronics
	"""
	# featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName

	# featureMatrix, labelList = readFeatureLabel(featureLabelFile)
	featureMatrix, labelList = readSensorData()

	transferLabelFile = "../../dataset/sensorType/sdh_soda_rice/transferLabel_sdh--rice.txt"
	auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile)

	featureMatrix = np.array(featureMatrix)
	labelArray = np.array(trueLabelList)
	transferLabelArray = np.array(transferLabelList)

	print("number of types", len(set(labelArray)))
	print('class count of true labels of all ex:\n', ct(transferLabelArray))

	initialExList = [[470, 352, 217],  [203, 280, 54], [267, 16, 190], [130, 8, 318], [290, 96, 418], [252, 447, 55],  [429, 243, 416], [240, 13, 68], [115, 449, 226], [262, 127, 381]]

	fold = 10
	rounds = 150

	multipleClassFlag = True
	al = _ProactiveLearning(fold, rounds, featureMatrix, labelArray, transferLabelArray, "rice_sensor", multipleClassFlag)

	al.setInitialExList(initialExList)

	al.run_CV()