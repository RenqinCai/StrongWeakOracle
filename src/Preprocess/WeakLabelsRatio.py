"""
ratio of weak labels
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

modelName = "weakLabelRatio_"+sourceDataName+"_"+targetDataName
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp

random.seed(10)
np.random.seed(10)

		
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



def getNoiseRatio(trueLabelList, transferLabelList):
	labelNum = len(trueLabelList)

	exIndexArray = np.array([i for i in range(labelNum)])

	posNoiseRatioTrainList = []
	posNoiseRatioTestList = []
	negNoiseRatioTrainList = []
	negNoiseRatioTestList = []

	foldNum = 10
	for foldIndex in range(foldNum):
		exIndexArrayTrain, exIndexArrayTest = train_test_split(exIndexArray, test_size=0.1)

		### true label is positive
		posNoiseLabelNum = 0.0

		### true label is negative
		negNoiseLabelNum = 0.0

		posLabelNum = 0.0
		negLabelNum = 0.0

		for exIndex in exIndexArrayTrain:
			transferLabel = transferLabelList[exIndex]
			trueLabel = trueLabelList[exIndex]

			if trueLabel == 1.0:
				if transferLabel != trueLabel:
					posNoiseLabelNum += 1.0

				posLabelNum += 1.0

			else:
				if transferLabel != trueLabel:
					negNoiseLabelNum += 1.0

				negLabelNum += 1.0

		posNoiseRatio = posNoiseLabelNum/posLabelNum
		negNoiseRatio = negNoiseLabelNum/negLabelNum

		posNoiseRatioTrainList.append(posNoiseRatio)
		negNoiseRatioTrainList.append(negNoiseRatio)
		# print("train pos noise ratio", posNoiseRatio)
		# print("train neg noise ratio", negNoiseRatio)

		### true label is positive
		posNoiseLabelNum = 0.0
		### true label is negative
		negNoiseLabelNum = 0.0

		posLabelNum = 0.0
		negLabelNum = 0.0

		for exIndex in exIndexArrayTest:
			transferLabel = transferLabelList[exIndex]
			trueLabel = trueLabelList[exIndex]

			if trueLabel == 1.0:
				if transferLabel != trueLabel:
					posNoiseLabelNum += 1.0

				posLabelNum += 1.0

			else:
				if transferLabel != trueLabel:
					negNoiseLabelNum += 1.0

				negLabelNum += 1.0

		posNoiseRatio = posNoiseLabelNum/posLabelNum
		negNoiseRatio = negNoiseLabelNum/negLabelNum
		posNoiseRatioTestList.append(posNoiseRatio)
		negNoiseRatioTestList.append(negNoiseRatio)
		# print("test pos noise ratio", posNoiseRatio)
		# print("test neg noise ratio", negNoiseRatio)

	print("mean train pos noise ratio", posNoiseRatioTrainList,  np.mean(posNoiseRatioTrainList), np.var(posNoiseRatioTrainList))
	
	print("mean train pos noise ratio", posNoiseRatioTestList, np.mean(posNoiseRatioTestList), np.var(posNoiseRatioTestList))
	
	print("mean train neg noise ratio", negNoiseRatioTrainList, np.mean(negNoiseRatioTrainList), np.var(negNoiseRatioTrainList))

	print("mean train neg noise ratio", negNoiseRatioTestList, np.mean(negNoiseRatioTestList), np.var(negNoiseRatioTestList))

if __name__ == "__main__":

	dataName = "electronics"
	featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

	featureMatrix, labelList = readFeatureLabel(featureLabelFile)

	transferLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/transferLabel_books--electronics.txt"
	auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile)

	getNoiseRatio(trueLabelList, transferLabelList)
