"""
train auditor offline with varioud feautre dim via PCA
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

modelName = "PCA_auditor_offline_"+dataName
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

	def __init__(self, fold, rounds, fn, tranfersLabel, label, targetLabel):

		self.fold = fold
		self.rounds = rounds

		self.fn = fn
		self.label = label
		self.targetLabel = targetLabel

		self.transferLabel = tranfersLabel
		# tranfersLabel = tranfersLabel.reshape(-1, 1)
		# self.fn = np.concatenate((fn, tranfersLabel), axis=1)

		self.tao = 0
		self.alpha_ = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.ex_id = dd(list)

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

		posRatioList = []

		# PCADim = 20
		# self.PCAFeature(PCADim)
		# print("PCADim", PCADim)
		foldNum = 1
		for foldIndex in range(foldNum):
			
			# self.m_clf = LinearSVC(random_state=3)
			self.m_clf = SVC(random_state=3)
			# self.m_clf = LR(random_state=3)
			# self.m_clf = RFC(random_state=3)


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

			fn_train = self.fn[train]
			label_train = self.label[train]

			testOneNum = np.sum(label_test)
			testNum = len(fn_test)

			posRatio = testOneNum*1.0/testNum
			posRatioList.append(posRatio)

			self.m_clf.fit(fn_train, label_train)

			label_preds = self.m_clf.predict(fn_test)
			# print("label_preds", label_preds)
			acc = accuracy_score(label_test, label_preds)

			debugExl = "auditor.xlsx"
			savePredLabel(debugExl, self.transferLabel[train], self.targetLabel[train], label_test, label_preds)

			totalAccList[cvIter] = acc

			cvIter += 1      
		
		print("posRatioList", posRatioList, np.mean(posRatioList), np.sqrt(np.var(posRatioList)))

		print("totalAccList", totalAccList, np.mean(totalAccList), np.sqrt(np.var(totalAccList)))

		totalACCFile = modelVersion+".txt"
		f = open(totalACCFile, "w")
		for i in range(10):
			f.write(str(totalAccList[i]))
			# for j in range(totalAlNum):
			# 	f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

import xlsxwriter
def savePredLabel(labelFile, predictLabelList, trueLabelList, auditorLabelList, predAuditorLabelList):
	workbook = xlsxwriter.Workbook(labelFile)
	worksheet = workbook.add_worksheet()

	labelNum = len(predictLabelList)

	worksheet.write(row=1, column=1).value = "predictLabel"
	worksheet.write(row=1, column=2).value = "trueLabel"
	worksheet.write(row=1, column=3).value = "auditorLabel"
	worksheet.write(row=1, column=4).value = "predAuditorLabel"

	for labelIndex in range(labelNum):
		colIndex = 1
		worksheet.write(row=labelIndex+2, column=colIndex).value = predictLabelList[labelIndex]
		colIndex += 1
		worksheet.write(row=labelIndex+2, column=colIndex).value = trueLabelList[labelIndex]
		colIndex += 1
		worksheet.write(row=labelIndex+2, column=colIndex).value = auditorLabelList[labelIndex]
		colIndex += 1
		worksheet.write(row=labelIndex+2, column=colIndex).value = predAuditorLabelList[labelIndex]

	workbook.close()
	

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

if __name__ == "__main__":

	###processedKitchenElectronics
	featureLabelFile = "../../dataset/processed_acl/processedBooksElectronics/"+dataName

	featureMatrix, labelList = readFeatureLabel(featureLabelFile)
	featureMatrix = np.array(featureMatrix)
	labelArray = np.array(labelList)

	###processedKitchenElectronics transferLabel_electronics--kitchen.txt
	transferLabelFile = "../../dataset/processed_acl/processedBooksElectronics/transferLabel_books--electronics.txt"
	auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)
	transferLabelArray = np.array(transferLabelList)
	
	auditorLabelArray = np.array(auditorLabelList)
	targetLabelArray = np.array(targetLabelList)
	# print(auditorLabel)
	# exit()
	# label = np.array([float(i.strip()) for i in open('targetAuditorLabel.txt').readlines()])

	# tmp = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
	# label = tmp[:,-1]
	print('class count of true labels of all ex:\n', ct(labelArray))
	print("count of auditor", ct(auditorLabelArray))
	# exit()
	# mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

	# fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100
	al = active_learning(fold, rounds, featureMatrix, transferLabelArray, auditorLabelArray, targetLabelArray)

	al.run_CV()
