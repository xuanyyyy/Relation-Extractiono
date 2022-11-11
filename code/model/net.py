"""各模型和损失函数的具体实现"""

import torch

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self, data_loader, params):
		super(CNN, self).__init__()
		#导入数据集对应的词向量
		embedding_vectors = data_loader.get_loaded_embedding_vectors()
		#文本信息和位置信息的嵌入层
		self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)
		self.pos1_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
		self.pos2_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

		self.max_len = params.max_len
		#dropout层
		self.dropout = nn.Dropout(params.dropout_ratio)

		feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
		#通过cnn对句子级别的特征进行编码
		self.covns = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=feature_dim,
									out_channels=params.filter_num,
									kernel_size=k),nn.Tanh(),nn.MaxPool1d(kernel_size=self.max_len-k+1)) for k in params.filters])

		filter_dim = params.filter_num * len(params.filters)
		labels_num = len(data_loader.label2idx)
		#输出层
		self.linear = nn.Linear(filter_dim, labels_num)

		self.loss = nn.CrossEntropyLoss()

		if params.gpu >= 0:
			self.cuda(device=params.gpu)

	def forward(self, x):
		batch_sents = x['sents']
		batch_pos1s = x['pos1s']
		batch_pos2s = x['pos2s']
		word_embs = self.word_embedding(batch_sents)
		pos1_embs = self.pos1_embedding(batch_pos1s)
		pos2_embs = self.pos2_embedding(batch_pos2s)

		input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2)  # batch_size x seq_len x feature_dim
		input_feature = input_feature.permute(0,2,1) #(batch_size,feature_dim,seq_len)
		input_feature = self.dropout(input_feature)

		out = [conv(input_feature) for conv in self.covns] #(batch_size,filter_num,1)
		"""
			对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1 (维度1是max_pool的结果)
			每个Window_size 产生filter_num个feature,然后把这些feature拼接起来
		"""
		out = torch.cat(out,dim=1)
		out = self.dropout(out)
		out = out.view(-1,out.size(1)) #(batch_size, (filter_num*window_num))
		x = self.dropout(out)
		x = self.linear(x)
		return x
