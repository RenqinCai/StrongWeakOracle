"""
proactive learning margin with random initialization, judge classifier to judge whether transfer learning is correct. margin as query strategy
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

"""
electronics, sensor_rice
"""
dataName = "sensor_rice"

modelName = "margin_linearMultipleAuditor_"+dataName
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

	def __init__(self, fold, rounds, featureMatrix, label, transferLabel0, transferLabel1):

		self.m_fold = fold
		self.m_rounds = rounds

		self.m_targetNameFeature = np.array(featureMatrix)
		self.m_targetLabel = np.array(label)

		self.m_transferLabel0 = np.array(transferLabel0)
		self.m_transferLabel1 = np.array(transferLabel1)

		self.m_randomForest = 0
		self.m_tao = 0

		self.m_alpha = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.002 ##0.05

		# self.m_judgeClassifier = 0
		self.m_auditor0 = None
		self.m_auditor1 = None
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

			idScore = self.getMargin(unlabeledId)
		
			unlabeledIdScoreMap[unlabeledId] = idScore

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def getMargin(self, unlabeledId):
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

	def getLUCB(self, unlabeledId):
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
		
		coefDiff = np.abs(np.dot(self.m_clf.coef_, self.m_targetNameFeature[unlabeledId]))

		LCB = coefDiff-2*0.002*selectCB
		LUCB = 1-LCB

		return LUCB

	def get_pred_acc(self, targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter):
		
		self.m_clf.fit(targetNameFeatureIter, targetLabelIter)
		targetLabelPreds = self.m_clf.predict(targetNameFeatureTest)

		acc = accuracy_score(targetLabelTest, targetLabelPreds)
		
		return acc

	def get_base_learners(self):
		self.m_randomForest = RFC(n_estimators=100, criterion='entropy', random_state=3)

		self.m_randomForest.fit(self.m_sourceDataFeature, self.m_sourceLabel)

	def init_confidence_bound(self, featureDim, labeledExList, unlabeledExList):
		self.m_weakLabeledIDList = []
		self.m_strongLabeledIDList = labeledExList

		# self.m_selectA = self.m_lambda*np.identity(featureDim)
		# self.m_selectAInv = np.linalg.inv(self.m_selectA)

		self.m_judgeA = self.m_lambda*np.identity(featureDim)
		self.m_judgeAInv = np.linalg.inv(self.m_judgeA)

	def update_select_confidence_bound(self, exId):
		
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
			return True, cbProb
		else:
			return False, cbProb

	def get_transfer_flag(self, transferFeatureList, transferFlagList0, transferFlagList1, exId, judgeRound):
		# predLabel = self.m_randomForest.predict(self.m_targetDataFeature[exId].reshape(1, -1))[0]

		transferLabel0_exId = self.m_transferLabel0[exId]
		transferLabel1_exId = self.m_transferLabel1[exId]

		predLabel = -1
		if len(np.unique(transferFlagList0)) > 1:
			if len(np.unique(transferFlagList1)) > 1:
				self.m_auditor0.fit(np.array(transferFeatureList), np.array(transferFlagList0))
				self.m_auditor1.fit(np.array(transferFeatureList), np.array(transferFlagList1))
			else:
				return False, -1, predLabel
		else:
			return False, -1, predLabel

		CB = self.get_judge_confidence_bound(exId)

		transferFlag0, LCB0 = self.get_judgeClassifier_prob(self.m_auditor0.coef_, self.m_targetNameFeature[exId].reshape(1, -1), CB, judgeRound)
		transferFlag1, LCB1 = self.get_judgeClassifier_prob(self.m_auditor1.coef_, self.m_targetNameFeature[exId].reshape(1, -1), CB, judgeRound)

		if transferFlag0:
			if transferFlag1:
				if LCB1 > LCB0:
					return True, 1, transferLabel1_exId
				else:
					return True, 0, transferLabel0_exId
			else:
				return True, 0, transferLabel0_exId
		else:
			if transferFlag1:
				return True, 1, transferLabel1_exId

			else:
				return False, -1, predLabel

	def getAuditorMetric(self, transferFeatureList, transferFlagList0, transferFlagList1, transferFeatureTest, transferLabelTest, targetLabelTest):
		acc = 0.0
		# if len(np.unique(transferFlagList0)) > 1:
		# 	if len(np.unique(transferFlagList1)) > 1:
		# 		self.m_auditor0.fit(np.array(transferFeatureList), np.array(transferFlagList0))
		# 		self.m_auditor1.fit(np.array(transferFeatureList), np.array(transferFlagList1))

		# 	auditorLabelTest = (transferLabelTest==targetLabelTest)

		# 	predictAuditorLabelTest = self.m_judgeClassifier.predict(transferFeatureTest)

		# 	acc = accuracy_score(auditorLabelTest, predictAuditorLabelTest)

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
	
		totalAccList = [[] for i in range(10)]
		humanAccList = [[] for i in range(10)]

		correctTransferRatioList = []
		totalTransferNumList = []
		correctUntransferRatioList = []

		totalAuditorPrecisionList = []
		totalAuditorRecallList = []
		totalAuditorAccList = []

		for foldIndex in range(foldNum):
			
			self.m_clf = LR(multi_class="multinomial", solver='lbfgs',random_state=3)
			self.m_auditor0 = LR(random_state=3)
			self.m_auditor1 = LR(random_state=3)

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

			# transferLabelTest = self.m_transferLabel[test]
			transferLabelTest = []

		
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
			transferFlagList0 = []
			transferFlagList1 = []

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
				# self.update_select_confidence_bound(exId)

				# print(idx)
				activeLabelFlag = False
				transferLabelFlag, weakOracleIndex, transferLabel = self.get_transfer_flag(transferFeatureList, transferFlagList0, transferFlagList1, exId, activeLabelNum)

				exLabel = -1
				if transferLabelFlag:
					self.m_weakLabeledIDList.append(exId)
					
					transferLabelNum += 1.0
					activeLabelFlag = False
					
					exLabel = transferLabel
					targetNameFeatureIter = np.vstack((targetNameFeatureIter, self.m_targetNameFeature[exId]))
					targetLabelIter = np.hstack((targetLabelIter, exLabel))
					# targetNameFeatureIter.append(self.m_targetNameFeature[exId])
					# targetLabelIter.append(exLabel)

					if exLabel == self.m_targetLabel[exId]:
						correctTransferLabelNum += 1.0
						print("queryIter\t", queryIter)
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

					weakLabel0 = self.m_transferLabel0[exId]
					weakLabel1 = self.m_transferLabel1[exId]
					
					transferFeatureList.append(self.m_targetNameFeature[exId])

					if weakLabel0 == exLabel:
						correctUntransferLabelNum += 1.0
						transferFlagList0.append(1.0)
					else:
						wrongUntransferLabelNum += 1.0
						transferFlagList0.append(0.0)

					if weakLabel1 == exLabel:
						correctUntransferLabelNum += 1.0
						transferFlagList1.append(1.0)
					else:
						wrongUntransferLabelNum += 1.0
						transferFlagList1.append(0.0)

					auditorAcc = self.getAuditorMetric(transferFeatureList, transferFlagList0, transferFlagList1, targetNameFeatureTest, transferLabelTest, targetLabelTest)
					print("auditorAcc", auditorAcc)

					auditorAccList.append(auditorAcc)

				labeledExList.append(exId)
				unlabeledExList.remove(exId)

				acc = self.get_pred_acc(targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter)
				totalAccList[cvIter].append(acc)
				if activeLabelFlag:
					humanAccList[cvIter].append(acc)
				queryIter += 1

			totalAuditorAccList.append(auditorAccList)

			transferLabelNum = len(self.m_weakLabeledIDList)
			totalTransferNumList.append(transferLabelNum)

			cvIter += 1      
		
		print("transfer num\t", np.mean(totalTransferNumList), np.sqrt(np.var(totalTransferNumList)))

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
	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../../dataset/sensorType/rice_pt_sdh').readlines()]
	tmp = np.genfromtxt('../../dataset/sensorType/rice_hour_sdh', delimiter=',')
	label = tmp[:,-1]

	fn = get_name_features(raw_pt)

	featureMatrix = fn
	labelList = label

	return featureMatrix, labelList

if __name__ == "__main__":

	# raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('./selectedNameFeature4Label_5types.txt').readlines()]

	# raw_pt = [i.strip().split('\t')[:-1] for i in open().readlines()]

	# f = open('../simulatedFeatureLabel_500_100_2.txt')
	# featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName
	# featureMatrix, labelList = readFeatureLabel(featureLabelFile)

	# transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
	# auditorLabelList, transferLabelList = readTransferLabel(transferLabelFile)

	featureMatrix, labelList = readSensorData()

	transferLabelFile0 = "../../dataset/sensorType/sdh_soda_rice/transferLabel_sdh--rice.txt"
	auditorLabelList0, transferLabelList0 = readTransferLabel(transferLabelFile)

	transferLabelFile1 = "../../dataset/sensorType/sdh_soda_rice/transferLabel_soda--rice.txt"
	auditorLabelList1, transferLabelList1 = readTransferLabel(transferLabelFile)

	featureMatrix = np.array(featureMatrix)
	labelArray = np.array(labelList)

	transferLabelArray0 = np.array(transferLabelList0)
	print 'class count of true labels of all ex:\n', ct(transferLabelArray0)

	transferLabelArray1 = np.array(transferLabelList1)
	print 'class count of true labels of all ex:\n', ct(transferLabelArray1)

	fold = 10
	rounds = 150

	al = _ProactiveLearning(fold, rounds, featureMatrix, labelArray, transferLabelArray0, transferLabelArray1)

	al.run_CV()
