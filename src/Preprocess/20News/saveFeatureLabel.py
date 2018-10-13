from sklearn.datasets import fetch_20newsgroups
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['rec.sport.baseball', 'rec.sport.hockey', 'talk.politics.misc', 'talk.religion.misc']

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=categories)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.95, min_df=5, max_features=2000, stop_words='english', analyzer='word')
vectors = vectorizer.fit_transform(newsgroups.data)
featureNameList = vectorizer.get_feature_names()
featureNum = len(featureNameList)

featureFile = "baseball_hockey_politicsMisc_religionMisc_vocab"
f = open(featureFile, "w")

featureNameMap = {}
for featureIndex in range(featureNum):
    featureName = featureNameList[featureIndex]
    featureNameMap[featureIndex] = featureName
    f.write(featureName+":"+str(featureIndex)+"\n")
f.close()

newsNum = vectors.shape[0]
labelList = newsgroups.target

baseBallFileName = "baseball"
baseBallF = open(baseBallFileName, "w")
baseBallFeatureLabelMatrix = []

for newsIndex in range(newsNum):
    if labelList[newsIndex] == 0:
        baseBallVec = [0.0 for i in range(featureNum)]
        for featureIndex in range(featureNum):
            featureVal = vectors[newsIndex, featureIndex]
            if featureVal > 0.0:
                baseBallVec[featureIndex] = featureVal
                baseBallF.write(str(featureVal)+"\t")
            else:
            	baseBallF.write(str(0.0)+"\t")
        
        baseBallF.write("\n")     
        baseBallFeatureLabelMatrix.append(baseBallVec)

baseBallF.close()


hockeyFileName = "hockey"
hockeyF = open(hockeyFileName, "w")
hockeyFeatureLabelMatrix = []

for newsIndex in range(newsNum):
    if labelList[newsIndex] == 1:
        hockeyVec = [0.0 for i in range(featureNum)]
        for featureIndex in range(featureNum):
            featureVal = vectors[newsIndex, featureIndex]
            if featureVal > 0.0:
                hockeyVec[featureIndex] = featureVal
                hockeyF.write(str(featureVal)+"\t")
            else:
            	hockeyF.write(str(0.0)+"\t")
        hockeyF.write("\n")
                
        hockeyFeatureLabelMatrix.append(hockeyVec)

hockeyF.close()


politicsMiscFileName = "politicsMisc"
politicsMiscF = open(politicsMiscFileName, "w")
politicsMiscFeatureLabelMatrix = []

for newsIndex in range(newsNum):
    if labelList[newsIndex] == 2:
        politicsMiscVec = [0.0 for i in range(featureNum)]
        for featureIndex in range(featureNum):
            featureVal = vectors[newsIndex, featureIndex]
            if featureVal > 0.0:
                politicsMiscVec[featureIndex] = featureVal
                politicsMiscF.write(str(featureVal)+"\t")
            else:
            	politicsMiscF.write(str(0.0)+"\t")
        
        politicsMiscF.write("\n")       
        politicsMiscFeatureLabelMatrix.append(politicsMiscVec)

politicsMiscF.close()


religionMiscFileName = "religionMisc"
religionMiscF = open(religionMiscFileName, "w")
religionMiscFeatureLabelMatrix = []

for newsIndex in range(newsNum):
    if labelList[newsIndex] == 3:
    	religionMiscVec = [0.0 for i in range(featureNum)]
        for featureIndex in range(featureNum):
            featureVal = vectors[newsIndex, featureIndex]
            if featureVal > 0.0:
                religionMiscVec[featureIndex] = featureVal
                religionMiscF.write(str(featureVal)+"\t")
            else:
            	religionMiscF.write(str(0.0)+"\t")
        
        religionMiscF.write("\n")          
        religionMiscFeatureLabelMatrix.append(religionMiscVec)

religionMiscF.close()

