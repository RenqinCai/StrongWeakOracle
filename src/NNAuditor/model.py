import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device=torch.device("cuda")

class BasicModule(torch.nn.Module):
	def __init__(self, args):
		super(BasicModule, self).__init__()
		self.m_args = args
		self.m_model_name = str(type(self))

	def maxPool(self, x):
		batch_size = x.size(0)
		seq_len = x.size(1)
		hidden_size = x.size(2)

		x = x.transpose(1, 2)
		x = F.max_pool1d(x, seq_len)

		x = x.transpose(1, 2).squeeze(1)

		return x

	def padDoc(self, x, pad_lens):
		pad_x = []
		sentStartIndex_doc = 0
		sentEndIndex_doc = 0

		hidden_size = x.size(1)

		max_pad_len = np.max(pad_lens)
		# print("max_pad_len", max_pad_len)
		avg_pad_len = int(np.mean(pad_lens))

		for pad_len in pad_lens:
			sentEndIndex_doc += pad_len
			pad_x_sample = []
			pad_x_sample.append(x[sentStartIndex_doc:sentEndIndex_doc])

			if pad_len < avg_pad_len:
				left_pad_len = avg_pad_len-pad_len
				pad_tensor = torch.zeros([left_pad_len, hidden_size]).to(device)
				pad_x_sample.append(pad_tensor)

			if pad_len > avg_pad_len:
				pad_x_sample = []
				# print("pad_len > avg_pad_len", pad_len, avg_pad_len)
				pad_x_sample.append(x[sentStartIndex_doc:sentStartIndex_doc+avg_pad_len])
				
			pad_x_sample = torch.cat(pad_x_sample).unsqueeze(0)
			pad_x.append(pad_x_sample)
			sentStartIndex_doc += pad_len
			# print("pad_x_sample", pad_x_sample.size())
		pad_x = torch.cat(pad_x)

		return pad_x.to(device)

	def save(self):
		checkpoint = {'model':self.state_dict(), 'args': self.m_args}
		best_path = '%s%s_seed_%d.pt' % (self.m_args.save_dir,self.m_model_name,self.m_args.seed)
		torch.save(checkpoint, best_path)

		return best_path

class auditorNN(BasicModule):
	def __init__(self, args, embed=None):
		super(auditorNN, self).__init__(args)
		self.m_model_name = "auditorNN"
		self.m_args = args

		embed_dim = self.m_args.emsize

		word_hidden_dim = self.m_args.wordhid
		sent_hidden_dim = self.m_args.senthid

		self.word_RNN = nn.GRU(input_size=embed_dim, hidden_size=word_hidden_dim, batch_first=True, bidirectional=True)

		self.sent_RNN = nn.GRU(input_size=word_hidden_dim*2, hidden_size=sent_hidden_dim, batch_first=True, bidirectional=True)

		self.linear_output = nn.Linear(sent_hidden_dim*2, 1)

	def forward(self, words, sentNum_doc):
		words_sent, _ = self.word_RNN(words)

		sents = self.maxPool(words_sent)

		sents = self.padDoc(sents, sentNum_doc)

		sents_doc, _ = self.sent_RNN(sents)

		docs = self.maxPool(sents_doc)

		pred_logits = self.linear_output(docs)

		return pred_logits
