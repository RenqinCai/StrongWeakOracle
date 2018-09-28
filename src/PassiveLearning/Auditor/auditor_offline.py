"""
adv active learning offline for auditor by considering the predicted label
"""
from sklearn.decomposition import PCA
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
from sklearn.svm import SVC 
from datetime import datetime

dataName = "electronics"

modelName = "auditor_offline_"+dataName
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp

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

	def __init__(self, fold, rounds, fn, label, category, multipleClass):

		self.m_category = category
		print("category", category)
		self.m_multipleClass = multipleClass
		print("multipleClass", multipleClass)

		self.fold = fold
		self.rounds = rounds

		self.fn = fn
		self.label = label
		# self.transferLabel = tranfersLabel
		# tranfersLabel = tranfersLabel.reshape(-1, 1)
		# self.fn = np.concatenate((fn, tranfersLabel), axis=1)

		self.tao = 0
		self.alpha_ = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.ex_id = dd(list)
		self.m_clf = None

	def PCAFeature(self, reducedDim):
		pca = PCA(n_components=reducedDim, svd_solver="full")
		self.fn = pca.fit_transform(self.fn)

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		# print(len(label_train), sum(label_train), "ratio", sum(label_train)*1.0/len(label_train))
		# print("label_train", label_train)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

	def init_confidence_bound(self, featureDim):
		self.m_A = self.m_lambda*np.identity(featureDim)
		self.m_AInv = np.linalg.inv(self.m_A)

	def update_confidence_bound(self, exId):
		self.m_A += np.outer(self.fn[exId], self.fn[exId])
		self.m_AInv = np.linalg.inv(self.m_A)

	def get_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.fn[exId], self.m_AInv), self.fn[exId]))

		return CB

	def setInitialExList(self, initialExList):
		self.m_initialExList = initialExList

	def pretrainSelectInit(self, train, foldIndex):

		initList = self.m_initialExList[foldIndex]
	
		print("initList", initList)

		return initList

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		print("featureNum", len(self.fn[0]))
		# print("non zero feature num", sum(self.fn[0]))

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
		cvIter = 0
		# random.seed(3)
		totalAccList = [0 for i in range(10)]

		testPosRatioList = []
		trainPosRatioList = []
		# self.PCAFeature(10)

		for foldIndex in range(foldNum):
			
			# self.m_clf = LinearSVC(random_state=3)
			# self.m_clf = SVC(random_state=3)
			# self.m_clf = LR(random_state=3)
			if self.m_multipleClass:
				self.m_clf = LR(multi_class="multinomial", solver='lbfgs',random_state=3,  fit_intercept=False)
			else:
				self.m_clf = LR(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			# trainNum = int(totalInstanceNum*0.2)
			# print("trainNum", trainNum)
			
			fn_test = self.fn[test]
			label_test = self.label[test]

			sampledTrainNum = len(train)
			# sampledTrainNum = 150
			train_sampled = random.sample(train, sampledTrainNum)
			train = train_sampled

			fn_train = self.fn[train]
			label_train = self.label[train]

			testOneNum = np.sum(label_test)
			testNum = len(fn_test)

			trainOneNum = np.sum(label_train)
			trainNum = len(label_train)
			trainPosRatio = trainOneNum*1.0/trainNum
			trainPosRatioList.append(trainPosRatio)

			testPosRatio = testOneNum*1.0/testNum
			testPosRatioList.append(testPosRatio)

			self.m_clf.fit(fn_train, label_train)

			label_preds = self.m_clf.predict(fn_test)
			acc = accuracy_score(label_test, label_preds)

			totalAccList[cvIter] = acc

			cvIter += 1      
		
		print("trainPosRatioList", trainPosRatioList, np.mean(trainPosRatioList), np.sqrt(np.var(trainPosRatioList)))

		print("testPosRatioList", testPosRatioList, np.mean(testPosRatioList), np.sqrt(np.var(testPosRatioList)))

		print("totalAccList", totalAccList, np.mean(totalAccList), np.sqrt(np.var(totalAccList)))

		# totalACCFile = modelVersion+".txt"
		# f = open(totalACCFile, "w")
		# for i in range(10):
		# 	f.write(str(totalAccList[i]))
		# 	# for j in range(totalAlNum):
		# 	# 	f.write(str(totalAccList[i][j])+"\t")
		# 	f.write("\n")
		# f.close()

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
	targetLabelList = []
	auditorLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		# print(float(line[1]))
		auditorLabelList.append(float(line[0]))
		transferLabelList.append(float(line[1]))
		targetLabelList.append(float(line[2]))

	f.close()

	return auditorLabelList, transferLabelList, targetLabelList

def readSensorData():
	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../../dataset/sensorType/sdh_soda_rice/rice_names').readlines()]
	tmp = np.genfromtxt('../../dataset/sensorType/rice_hour_sdh', delimiter=',')
	label = tmp[:,-1]

	fn = get_name_features(raw_pt)

	featureMatrix = fn
	labelList = label

	return featureMatrix, labelList

if __name__ == "__main__":

	dataName = "simulation"

	modelName = "offline_auditor_"+dataName
	timeStamp = datetime.now()
	timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

	modelVersion = modelName+"_"+timeStamp
	fileSrc = dataName

	"""
	 	processedKitchenElectronics
	"""
	if dataName == "electronics":
		
		featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		transferLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/transferLabel_books--electronics.txt"
		auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(labelList)

		transferLabelArray = np.array(transferLabelList)
		auditorLabelArray = np.array(auditorLabelList)

		initialExList = []
		initialExList = [[397, 1942, 200], [1055, 144, 873], [865, 1702, 1769], [1156, 906, 1964], [1562, 1299, 617], [231, 532, 690], [1751, 1247, 1082], [817, 1631, 426], [360, 1950, 1702], [1921, 822, 1528]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, auditorLabelArray, "sentiment_electronics", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()

	"""
	 	sensor type
	"""
	if dataName == "sensor_rice":
		featureMatrix, labelList = readSensorData()

		transferLabelFile0 = "../../dataset/sensorType/sdh_soda_rice/transferLabel_sdh--rice.txt"
		auditorLabelList0, transferLabelList0, trueLabelList = readTransferLabel(transferLabelFile0)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(trueLabelList)
		auditorLabelArray = np.array(auditorLabelList0)

		initialExList = []
		initialExList = [[470, 352, 217],  [203, 280, 54], [267, 16, 190], [3, 362, 268], [328, 307, 194], [77, 413, 380],  [119, 170, 420], [312, 310, 6],  [115, 449, 226], [297, 87, 46]]

		fold = 10
		rounds = 150

		multipleClassFlag = True
		al = active_learning(fold, rounds, featureMatrix, auditorLabelArray, "sensor", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()

	if dataName == "simulation":
		featureLabelFile = "../../../dataset/synthetic/simulatedFeatureLabel_500_20_2.txt"

		featureMatrix, labelList = readFeatureLabel(featureLabelFile)

		transferLabelFile0 = "../../../dataset/synthetic/simulatedTransferLabel_500_20_2.txt"
		auditorLabelList0, transferLabelList0, trueLabelList = readTransferLabel(transferLabelFile0)

		featureMatrix = np.array(featureMatrix)
		labelArray = np.array(trueLabelList)
		auditorLabelArray = np.array(auditorLabelList0)

		initialExList = []
		initialExList = [[42, 438, 9],  [246, 365, 299], [145, 77, 45], [353, 369, 299], [483, 337, 27], [489, 468, 122],  [360, 44, 412], [263, 284, 453], [449, 3, 261], [244, 200, 47]]

		fold = 10
		rounds = 150

		multipleClassFlag = False
		al = active_learning(fold, rounds, featureMatrix, auditorLabelArray, "synthetic", multipleClassFlag)

		al.setInitialExList(initialExList)

		al.run_CV()