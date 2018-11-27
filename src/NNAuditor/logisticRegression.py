"""
add glove word embedding 
"""
import numpy as np
import random
import os
import json

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.linear_model import LogisticRegression as LR
from utils import sentence2word_normalized
import torch

max_wordLength = 100

class _Doc:
	def __init__(self):
		self.m_sents = []
		self.m_words = []
		self.m_wordsNum = 0
		### 1: pos; 0: neg
		self.m_sentiment = 0

		self.m_sent_num = 0
		self.m_ID = ""

	def addSent(self, words_sent):
		self.m_sents.append(words_sent)
		self.m_sent_num += 1

	def setWords(self, words):
		self.m_words = words
		self.m_wordsNum = len(self.m_words)

	def setDocID(self, doc_ID):
		self.m_ID = doc_ID

	def setSentiment(self, rating):
		if rating > 3.0:
			self.m_sentiment = True
		
		if rating < 3.0:
			self.m_sentiment = False

class _Corpus:
	def __init__(self):
		self.m_docs = []
		self.m_doc_num = 0

		self.m_word2id_map = {}
		self.m_id2word_map = {}

		### word embedding
		self.m_word2embed = []
		self.m_voc_size = 0
		self.m_embed_size = 0

		self.m_train_ratio = 0.7
		self.m_val_ratio = 0.1
		self.m_test_ratio = 0.2

	def add_doc(self, docObj):
		self.m_docs.append(docObj)
		self.m_doc_num += 1

	def load_embeddings(self, glove_embedding, embeddingFileName):
		f = open(embeddingFileName)
		# glove_embedding = dict()

		for rawLine in f:
			splitted_line = rawLine.split()
			word = splitted_line[0]
			word_vec = np.asarray(splitted_line[1:], dtype='float32')

			glove_embedding[word] = word_vec
			self.m_embed_size = len(word_vec)

		f.close()

	def init_preTrain_embeddings(self, wordEmbedding):
		self.m_voc_size = len(self.m_word2id_map)
		self.m_word2embed = np.zeros((self.m_voc_size, self.m_embed_size))

		for word in self.m_word2id_map:
			word_id = self.m_word2id_map[word]
			
			word_vec = []
			if word in wordEmbedding:
				word_vec = wordEmbedding[word]
			else:
				word_vec = wordEmbedding['unk']
			self.m_word2embed[word_id] = word_vec

		self.m_word2embed = torch.from_numpy(self.m_word2embed).float()

	def parse_docs(self, dataFileName, embeddingFileName):
		##
		glove_embedding = dict()

		# embeddingFileName = os.path.join(dirName, embeddingFileName)
		print("load embedding ...")
		self.load_embeddings(glove_embedding, embeddingFileName)
		print("... end load embedding")


		print("parse docs ...")
		self.parse_doc(dataFileName, glove_embedding)
		# for fileName in os.listdir(dirName):
		# 	if fileName.endswith(".json"):
				# self.parse_doc(dirName, fileName)

		# print("init word embedding ...")
		# self.init_preTrain_embeddings(glove_embedding)
		# print("... end word embedding")

	def parse_doc(self, dataFileName, glove_embedding):

		# jsonFile = os.path.join(dirName, dataFileName)
		jsonFile = dataFileName
		f = open(jsonFile)

		for rawLine in f:
			review_json = json.loads(rawLine)
			review_text = review_json["content"]
			review_ID = review_json["reviewID"]
			review_rating = review_json["rating"]

			sents = sent_tokenize(review_text)
			sents_num = len(sents)
			for sent_index in range(sents_num):
				sent = sents[sent_index]
				words = sentence2word_normalized(sent)

				if len(words) == 0:
					continue

				word_num = len(words)
				words_IDs = []
				for word_index in range(word_num):
					word = words[word_index]
					word_id = -1

					if word not in glove_embedding:
						continue

					if word not in self.m_word2id_map:
						word_id = len(self.m_word2id_map)
						self.m_word2id_map[word] = word_id
		f.close()
		print("load word 2 id map ...")
		self.m_word2id_map['unk'] = len(self.m_word2id_map)


		f = open(jsonFile)
		for rawLine in f:
			review_json = json.loads(rawLine)
			review_text = review_json["content"]
			review_ID = review_json["reviewID"]
			review_rating = review_json["rating"]

			docObj = _Doc()
			docObj.setDocID(review_ID)

			sents = sent_tokenize(review_text)
			sents_num = len(sents)
			words_IDs = []
			for sent_index in range(sents_num):
				sent = sents[sent_index]
				words = sentence2word_normalized(sent)

				if len(words) == 0:
					continue

				word_num = len(words)
				
				for word_index in range(word_num):
					word = words[word_index]
					word_id = -1

					if word not in self.m_word2id_map:
						word_id = self.m_word2id_map['unk']
					else:
						word_id = self.m_word2id_map[word]

					words_IDs.append(word_id)

			words_IDs = words_IDs[:max_wordLength]

			if len(words_IDs) < 80:
				continue

			if len(words_IDs) > 150:
				continue

			docObj.setWords(words_IDs)

			if docObj.m_wordsNum == 0:
				continue
			docObj.setSentiment(review_rating)
			self.add_doc(docObj)
			if self.m_doc_num %1000 == 0:
				print("load doc num", self.m_doc_num)
		f.close()

	def splitTrainValidateTest(self):
		for i in range(10):
			random.shuffle(self.m_docs)
		doc_num = len(self.m_docs)
		train_num = int(doc_num*self.m_train_ratio)
		test_num = int(doc_num*self.m_test_ratio)
		val_num = doc_num - train_num - test_num

		train_data = self.m_docs[:train_num]
		val_data = self.m_docs[train_num: train_num+val_num] 
		test_data = self.m_docs[train_num+val_num:]
		print("train_num{:3d}, val_num{:3d}, test_num{:3d}".format(train_num, val_num, test_num))
		return train_data, val_data, test_data