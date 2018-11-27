# BCEWithLogitsLoss

import numpy as np
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model
import pickle


"""
initialize arg parser
"""

parser = argparse.ArgumentParser(description='pytorch NN auditor')
parser.add_argument('--data', type=str, default='./data/', help='location of the data')
parser.add_argument('--model', type=str, default='GRU', help='type of recurrent net (GRU)')
parser.add_argument('--emsize', type=int, default=300, help='size of word embeddings')
parser.add_argument('--wordhid', type=int, default=200, help='number of word hidden units per layer')
parser.add_argument('--senthid', type=int, default=200, help='number of sent hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--save_dir', type=str, default='./checkpoint/', help='dir to save the final model')
parser.add_argument('--onnx_export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('-max_norm',type=float,default=1.0)


args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	if not args.cuda:
		print('warning; you have a cuda device, so you should probably run with --cuda')

device=torch.device("cuda")
batch_size = args.batch_size

def gpuData(input):
	return input.to(device)

"""
load data
"""
print("-"*89)

# data_dirName = "../../dataset/Julian/smallReview_electronics.json"
data_fileName = "../../dataset/Julian/smallReview_electronics.json"
embedding_fileName = "../../../../HightlightPrediction/Src/InferSent/fastText/crawl-300d-2M.vec"

corpusObj = data._Corpus()
corpusObj.parse_docs(data_fileName, embedding_fileName)

"""
split data: train, validate, test
"""
train_data, val_data, test_data = corpusObj.splitTrainValidateTest()
# train_data = gpuData(train_data)
# val_data = gpuData(val_data)
# test_data = gpuData(test_data)

"""
build model
"""
auditor_NN = model.auditorNN(args)
auditor_NN = gpuData(auditor_NN)

criterion = nn.BCEWithLogitsLoss()
print("-"*89)

"""
train model
"""

def batchify_data(source):
	batch_x = []
	batch_y = []

	max_wordsNum_sent = 0
	for docObj in source:
		sents_doc = docObj.m_sents
		sentNum_doc = len(sents_doc)
		for sentIndex_doc in range(sentNum_doc):
			words_sent = sents_doc[sentIndex_doc]
			wordsNum = len(words_sent)

			if wordsNum > max_wordsNum_sent:
				max_wordsNum_sent = wordsNum

	sentNumList_batch = []
	for docObj in source:
		sents_doc = docObj.m_sents
		sentNum_doc = len(sents_doc)

		sentNumList_batch.append(sentNum_doc)

		batch_sents_doc = []

		sentiment_doc = docObj.m_sentiment
		batch_y.append(sentiment_doc)

		for sentIndex_doc in range(sentNum_doc):
			words_sent = sents_doc[sentIndex_doc]
			words_num = len(words_sent)

			batch_words_sent = []
			for word_index in range(words_num):
				word_id = words_sent[word_index]

				word_embed = corpusObj.m_word2embed[word_id]
				batch_words_sent.append(word_embed.unsqueeze(0))

			if words_num < max_wordsNum_sent:
				pad_wordsNum = max_wordsNum_sent - words_num
				batch_words_sent.append(torch.zeros([pad_wordsNum, args.emsize]))

			batch_words_sent = torch.cat(batch_words_sent)
			batch_sents_doc.append(batch_words_sent.unsqueeze(0))

		batch_sents_doc = torch.cat(batch_sents_doc)

		batch_x.append(batch_sents_doc)

	batch_x = torch.cat(batch_x)
	batch_y = torch.from_numpy(np.array(batch_y)*1.0).float()

	return batch_x.to(device), batch_y.to(device), sentNumList_batch

def train():
	auditor_NN.train()
	for batch_index, docIndex_batch in enumerate(np.arange(0, len(train_data), batch_size)):
		batch_train_data = train_data[docIndex_batch: docIndex_batch+batch_size]
		batch_x, batch_y, sentNumList_batch = batchify_data(batch_train_data)

		pred_logits_batch = auditor_NN(batch_x, sentNumList_batch)

		optimizer.zero_grad()
		loss = criterion(pred_logits_batch.squeeze(), batch_y)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(auditor_NN.parameters(), args.max_norm)
		optimizer.step()

		if batch_index %100 == 0:
			print(" training | batch_index{:3d} | loss{:5.2f} |".format(batch_index, loss))


"""
evaluate model
"""
def evaluate(evaluate_data):
	auditor_NN.eval()
	total_loss = 0.0
	batch_num = 0.0
	for batch_index, docIndex_batch in enumerate(np.arange(0, len(evaluate_data), batch_size)): 
		batch_evaluate_data = evaluate_data[docIndex_batch: docIndex_batch+batch_size]

		batch_x, batch_y, sentNumList_batch = batchify_data(batch_evaluate_data)
		pred_logits_batch = auditor_NN(batch_x, sentNumList_batch)

		loss = criterion(pred_logits_batch.squeeze(), batch_y)
		total_loss += loss
		batch_num += 1.0

	# print("avg loss", total_loss/batch_num)
	avg_loss = total_loss/batch_num
	auditor_NN.train()
	return avg_loss

def getAcc(evaluate_data):
	auditor_NN.eval()

	total_TP = 0.0
	total_FP = 0.0
	total_TN = 0.0
	total_FN = 0.0

	for batch_index, docIndex_batch in enumerate(np.arange(0, len(evaluate_data), batch_size)): 
		batch_evaluate_data = evaluate_data[docIndex_batch: docIndex_batch+batch_size]

		batch_x, batch_y, sentNumList_batch = batchify_data(batch_evaluate_data)
		pred_logits_batch = auditor_NN(batch_x, sentNumList_batch)
		pred_probs_batch = torch.sigmoid(pred_logits_batch)

		pred_labels_batch = pred_probs_batch > 0.5
		acc_flag_batch = (batch_y == pred_labels_batch*2.0-1)
		TP = np.sum(acc_flag_batch)

		Pos = np.sum(pred_labels_batch)
		FP = Pos - TP

		acc_flag_batch = (1.0-2*batch_y == 1.0-pred_labels_batch)
		TN = np.sum(acc_flag_batch)

		Neg = np.sum(1.0-pred_labels_batch)
		FN = Neg - TN

		total_TP += TP
		total_FP += FP

		total_TN += TN
		total_FN += FN

	precision = total_TP*1.0 / (total_TP+total_FP)
	recall = total_TP*1.0/ (total_TP+total_FN)
	acc = (total_TP+total_TN)*1.0/(total_TP+total_FP+total_TN+total_FN)

	print("precision{:5.2f}, recall{:5.2f}, acc{:5.2f}".format(precision, recall, acc))



"""
main function
"""

def save(self):
	checkpoint = {'model':self.state_dict(), 'args': self.args}
	best_path = '%s%s_seed_%d.pt' % (args.save_dir, model_name, args.seed)
	torch.save(checkpoint,best_path)

# lr = args.lr
best_val_loss = None

try:

	optimizer = torch.optim.Adam(auditor_NN.parameters(), lr=args.lr)

	for epoch_index in range(args.epochs):
		print('*'*89)
		epoch_start_time = time.time()
		train()
		val_loss = evaluate(val_data)
		print('| epoch{:3d} | time{:5.2f}s | valid loss{:5.2f}|'.format(epoch_index, (time.time() - epoch_start_time), val_loss))

		if not best_val_loss or val_loss < best_val_loss:
			best_val_loss = val_loss
			best_path = auditor_NN.save()
			# with open(args.save, "wb") as f:
			# 	torch.save(auditor_NN, f)
	
	getAcc(test_data)

except KeyboardInterrupt:
	print('-'*89)
	print('exit from training early ctrl-c')


