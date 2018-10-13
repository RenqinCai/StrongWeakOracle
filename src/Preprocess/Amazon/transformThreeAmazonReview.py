"""
use amazon reviews Books, Kitchen ---> Electronics
"""


import numpy as np
import os
from sklearn.feature_selection import SelectKBest, chi2

class _Vocab:
	def __init__(self):
		self.m_vocSize = 0
		###wordStr: wordID
		self.m_word2IDMap = {}
		###wordStr: wordDF
		self.m_wordDFMap = {}

		self.m_ID2wordMap = {}

	def addWordDF(self, wordStr):
		if wordStr not in self.m_wordDFMap:
			self.m_wordDFMap.setdefault(wordStr, 0.0)

		self.m_wordDFMap[wordStr] += 1.0

class _Doc:
	def __init__(self):
		###True:pos; False:neg
		self.m_posNeg = True
		self.m_label = ""
		##wordStr: wordTF
		self.m_wordMap = {}
		self.m_wordList = []

	def addWordTF(self, wordStr, wordTF):
		if wordStr not in self.m_wordMap.keys():
			self.m_wordMap.setdefault(wordStr, wordTF)
		else:
			print("error existing word")

def readReviewFile(fileName, vocabObj, docList, label, sentiment):
	f = open(fileName)

	for rawLine in f:
		splittedLine = rawLine.strip().split(" ")
		lineLen = len(splittedLine)
		# print(splittedLine, lineLen)
		docObj = _Doc()
		docList.append(docObj)

		docObj.m_label = label
		docObj.m_posNeg = sentiment

		for unitIndex in range(lineLen-1):
			wordUnit = splittedLine[unitIndex]
			wordUnitSplitted = wordUnit.split(":")

			wordStr = wordUnitSplitted[0]
			wordTF = float(wordUnitSplitted[1])

			vocabObj.addWordDF(wordStr)

			docObj.addWordTF(wordStr, wordTF)

def filterWord42Vocab(vocabObj, vocabObj1, vocabObj2, vocabObj3):

	preWordStrList1 = list(vocabObj1.m_wordDFMap.keys())
	preWordStrList2 = list(vocabObj2.m_wordDFMap.keys())
	preWordStrList3 = list(vocabObj3.m_wordDFMap.keys())

	thresholdDocNumMax = 100

	wordStrList1 = []
	for wordStr in preWordStrList1:
		wordDF = vocabObj1.m_wordDFMap[wordStr]
		if wordDF > 3.0:
			if wordDF < thresholdDocNumMax:
				wordStrList1.append(wordStr)

	wordStrList2 = []
	for wordStr in preWordStrList2:
		wordDF = vocabObj2.m_wordDFMap[wordStr]
		if wordDF > 3.0:
			if wordDF < thresholdDocNumMax:
				wordStrList2.append(wordStr)

	wordStrList3 = []
	for wordStr in preWordStrList3:
		wordDF = vocabObj3.m_wordDFMap[wordStr]
		if wordDF > 3.0:
			if wordDF < thresholdDocNumMax:
				wordStrList3.append(wordStr)

	commonWordStrList = set(wordStrList1).intersection(set(wordStrList2))
	commonWordStrList = commonWordStrList.intersection(set(wordStrList3))
	commonWordStrList = list(commonWordStrList)

	print("commonWordStrList", len(commonWordStrList))

	for wordStr in commonWordStrList:
		wordDF =  vocabObj1.m_wordDFMap[wordStr]+vocabObj2.m_wordDFMap[wordStr]+vocabObj3.m_wordDFMap[wordStr]
		vocabObj.m_wordDFMap[wordStr] = wordDF 

		wordID = len(vocabObj.m_word2IDMap)
		vocabObj.m_word2IDMap.setdefault(wordStr, wordID)
		vocabObj.m_ID2wordMap.setdefault(wordID, wordStr)

	vocabObj.m_vocSize = len(vocabObj.m_word2IDMap)
	print("voc size", vocabObj.m_vocSize)

	return commonWordStrList

def saveVocab(saveDir, vocabObj, label):
	vocabFileName = os.path.join(saveDir, label+"vocab")
	f = open(vocabFileName, "w")
	for wordStr in vocabObj.m_word2IDMap:
		f.write(wordStr+":"+str(vocabObj.m_word2IDMap[wordStr]))
		f.write("\n")

	f.close()

def saveReview(saveDir, vocabObj, docList, label):
	print("***************label", label)
	reviewFileName = os.path.join(saveDir, label)
	f = open(reviewFileName, "w")
	docNum = len(docList)
	print("docNum", docNum)

	newDocList = []

	for docObj in docList:
		docObj.m_wordList = [0.0 for i in range(vocabObj.m_vocSize)]
		for wordStr in docObj.m_wordMap:
			wordTF = docObj.m_wordMap[wordStr]
			
			if wordStr in vocabObj.m_word2IDMap:
				wordIndex = vocabObj.m_word2IDMap[wordStr]
				if wordTF == 0.0:
					print("error")
					docObj.m_wordList[wordIndex] = 0.0
				else:
					docObj.m_wordList[wordIndex] = wordTF
					# docObj.m_wordList[wordIndex] = (1.0+np.log(wordTF))
		# if sum(docObj.m_wordList) == 0.0:
		# 	continue
	# 	newDocList.append(docObj)

	# newDocNum = len(newDocList)
	# print("removing empty doc after preprocessing", newDocNum)
	# wordNum = vocabObj.m_vocSize

	# for docObj in newDocList:
	# 	for wordIndex in range(wordNum):
	# 		wordStr = vocabObj.m_ID2wordMap[wordIndex]
	# 		DF = vocabObj.m_wordDFMap[wordStr]
	# 		DF = np.log(newDocNum*1.0/DF)

	# 		wordTF = docObj.m_wordList[wordIndex]
	# 		wordTFIDF = DF*wordTF
	# 		docObj.m_wordList[wordIndex] = wordTFIDF

	
	for docObj in newDocList:
		# if docObj.m_label != label:
		# 	continue
		for wordIndex in range(wordNum):

			f.write(str(docObj.m_wordList[wordIndex])+"\t")

		if docObj.m_posNeg == True:
			f.write(str(1.0))
		else:
			f.write(str(0.0))
		f.write("\n")

	f.close()



# vocabObj = _Vocab()
# commonWordStrList = filterWord42Vocab(vocabObj, vocabObj1, vocabObj2)

# outputFile = "../../../dataset/processed_acl/processedBooksElectronics"

# saveReview(outputFile, vocabObj, docList1, "kitchen")
# saveReview(outputFile, vocabObj, docList2, "electronics")
# saveVocab(outputFile, vocabObj, "kitchen_electronics")
# filterWordByDF(vocabObj, docList)

# saveReview(outputFile, vocabObj, docList, "kitchen_electronics")

# inputFile_pos_ele = "../../../dataset/processed_acl/electronics/positive.review"
# inputFile_neg_ele = "../../../dataset/processed_acl/electronics/negative.review"

# # inputFile_pos = "../processed_acl/dvd/positive.review"
# # inputFile_neg = "../processed_acl/dvd/negative.review"

docList1 = []
vocabObj1 = _Vocab()

inputFile_pos_books = "../../../dataset/processed_acl/books/positive.review"
inputFile_neg_books = "../../../dataset/processed_acl/books/negative.review"

readReviewFile(inputFile_pos_books, vocabObj1, docList1, "books", True)
readReviewFile(inputFile_neg_books, vocabObj1, docList1, "books", False)

docList2 = []
vocabObj2 = _Vocab()

inputFile_pos_ele = "../../../dataset/processed_acl/electronics/positive.review"
inputFile_neg_ele = "../../../dataset/processed_acl/electronics/negative.review"

readReviewFile(inputFile_pos_ele, vocabObj2, docList2, "electronics", True)
readReviewFile(inputFile_neg_ele, vocabObj2, docList2, "electronics", False)

docList3 = []
vocabObj3 = _Vocab()

inputFile_pos_ele = "../../../dataset/processed_acl/kitchen/positive.review"
inputFile_neg_ele = "../../../dataset/processed_acl/kitchen/negative.review"

readReviewFile(inputFile_pos_ele, vocabObj3, docList3, "kitchen", True)
readReviewFile(inputFile_neg_ele, vocabObj3, docList3, "kitchen", False)

vocabObj = _Vocab()
commonWordStrList = filterWord42Vocab(vocabObj, vocabObj1, vocabObj2, vocabObj3)

outputFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics"

saveReview(outputFile, vocabObj, docList1, "books")
saveReview(outputFile, vocabObj, docList2, "electronics")
saveReview(outputFile, vocabObj, docList3, "kitchen")
saveVocab(outputFile, vocabObj, "books_kitchen_electronics")

# outputFile = "../../../dataset/processed_acl/processedKitchenElectronics"


# filterByChiSquare(vocabObj, docList)
# readReviewFile(inputFile_pos_ele, vocabObj, docList, "electronics", True)
# readReviewFile(inputFile_neg_ele, vocabObj, docList, "electronics", False)

# readReviewFile(inputFile_pos, vocabObj, docList, "dvd", True)
# readReviewFile(inputFile_neg, vocabObj, docList, "dvd", False)

# readReviewFile(inputFile_pos, vocabObj, docList, "books", True)
# readReviewFile(inputFile_neg, vocabObj, docList, "books", False)

# filterWordByDF(vocabObj, docList)

# saveReview(outputFile, vocabObj, docList, "kitchen_electronics")


