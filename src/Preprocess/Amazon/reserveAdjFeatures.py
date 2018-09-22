import nltk

fileName = "../../../dataset/processed_acl/processedBooksKitchenElectronics/books_kitchen_electronicsvocab"

f = open(fileName)

reservedFeatureList = []
reservedFeaturIDList = []

adjTags = ["JJ", "JJR", "JJS"]

for rawLine in f:
	rawLine = rawLine.strip().split(":")

	featureStr = rawLine[0]
	featureID = int(rawLine[1])
	# rawLine = rawLine.strip().split(":")[0]

	if "_" in featureStr:
		# print(rawLine)
		splittedLine = featureStr.strip().split("_")
		for wordStr in splittedLine:
			if not wordStr:
				continue
			posTag = nltk.pos_tag([wordStr])[0][1]
			if posTag in adjTags:
				print(wordStr, "posTag", posTag, featureStr)
				reservedFeatureList.append(featureStr)
				reservedFeaturIDList.append(featureID)
				break

	else:
		# print(rawLine)
		for wordStr in featureStr:
			# print(wordStr)
			posTag = nltk.pos_tag([wordStr])[0][1]
			# print(posTag)
			if posTag in adjTags:
				reservedFeatureList.append(wordStr)
				reservedFeaturIDList.append(featureID)


print("reservedFeatureList", reservedFeatureList)
print("reservedFeaturIDList", reservedFeaturIDList)