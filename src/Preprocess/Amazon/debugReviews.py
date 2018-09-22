vocabFileName = "../../../dataset/processed_acl/processedBooksKitchenElectronics/books_kitchen_electronicsvocab"

f = open(vocabFileName)

### featureIndex: featureStr
featureID2StrMap = {}

for rawLine in f:
	rawLine = rawLine.strip().split(":")
	featureStr = rawLine[0]
	featureID = int(rawLine[1])

	featureID2StrMap.setdefault(featureID, featureStr)

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

dataName = "electronics"
featureLabelFile = "../../../dataset/processed_acl/processedBooksKitchenElectronics/"+dataName

featureMatrix, labelList = readFeatureLabel(featureLabelFile)

reviewIDList = [1197, 1025, 1877, 1591]
# reviewIDList = [1358, 1873, 107, 697]

# adjFeatureList = [1, 3126, 31, 71, 73, 76, 93, 103, 105, 110, 939, 154, 162, 173, 3067, 191, 204, 2496, 242, 263, 264, 268, 297, 308, 311, 327, 2005, 340, 1638, 371, 372, 383, 389, 393, 1849, 399, 409, 415, 430, 444, 457, 460, 466, 469, 478, 514, 537, 543, 547, 556, 561, 567, 604, 630, 646, 647, 1897, 711, 1291, 778, 809, 855, 870, 1940, 906, 912, 950, 952, 959, 960, 973, 986, 988, 995, 1017, 1033, 3236, 1048, 1052, 1054, 1090, 1126, 1136, 1151, 1185, 1218, 1246, 557, 1270, 1275, 1297, 1299, 1300, 1308, 1312, 1314, 1329, 1331, 527, 1357, 1380, 1394, 3296, 1438, 1459, 3370, 1471, 1475, 1500, 1545, 1546, 3214, 3255, 2594, 1581, 1943, 1589, 2072, 1608, 1610, 1618, 1660, 1678, 3457, 1741, 1747, 1766, 1786, 3365, 1812, 270, 1821, 1823, 1830, 1832, 1833, 1853, 1866, 1938, 1891, 1898, 1904, 1918, 1977, 1980, 2778, 1995, 1656, 2033, 1003, 1436, 1452, 2093, 2103, 2105, 1825, 2122, 2134, 2137, 2161, 3247, 2178, 2182, 2200, 2206, 2207, 2210, 2222, 2235, 2239, 2242, 3479, 2252, 2271, 2279, 2284, 2301, 3448, 1336, 2331, 2342, 2382, 2404, 2412, 1720, 2434, 2436, 2443, 2849, 2456, 2470, 301, 2489, 2497, 2503, 2514, 497, 2533, 2548, 2573, 2576, 2578, 3443, 3678, 1238, 2685, 2697, 1076, 2733, 2760, 2766, 2768, 2794, 2798, 2809, 1026, 2844, 2862, 2589, 2871, 2883, 2802, 3325, 19, 2938, 2940, 2955, 2957, 2975, 2988, 3002, 3018, 3042, 3568, 3052, 3192, 158, 175, 3088, 3097, 3105, 3405, 3117, 3120, 3148, 3180, 3194, 842, 866, 801, 3239, 3251, 3253, 3269, 1382, 3318, 3341, 3355, 3356, 3357, 1803, 3367, 223, 1042, 3401, 2057, 3408, 3417, 3426, 2274, 2329, 3454, 3466, 3473, 3537, 3541, 3564, 3581, 3667, 3603, 3613, 3672]

for reviewID in reviewIDList:
	featureList = featureMatrix[reviewID]
	print("=============review %d========"%reviewID, "*********sentiment %d ******"%labelList[reviewID])
	for featureIndex in range(len(featureList)):
		# if featureIndex not in adjFeatureList:
		# 	continue
		featureVal = featureList[featureIndex]
		if featureVal > 0:
			print(featureID2StrMap[featureIndex], ":", featureVal)
