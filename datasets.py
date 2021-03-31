# *****************************************
# 
# Customised dataset class for 
# SemEval 2021 Task 7: Hahackathon
# 
# ****************************************

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, DebertaTokenizer

import sys

class Hahackathon(torch.utils.data.Dataset):
	'''
	Hahackathon dataset

	filename: train/val/test file to be read

	basenet : bert/ernie/roberta/deberta

	is_test : if the input file does not have groundtruth labels
	        : (for evaluation on the leaderboard)
	'''

	def __init__(self, filename, basenet= 'bert', max_length= 128, stop_words= False, is_test= False):
		super(Hahackathon, self).__init__()

		self.is_test = is_test

		if stop_words:
			self.nlp  = English()

		self.data    = self.read_file(filename, stop_words)

		if basenet == 'bert':
			print("Tokenizer: bert-base-uncased\n")
			self.token = BertTokenizer.from_pretrained('bert-base-uncased')
		elif basenet == 'ernie':
			print("Tokenizer: ernie-2.0-en\n")
			self.token = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
		elif basenet == 'roberta':
			print("Tokenizer: roberta-base\n")
			self.token = RobertaTokenizer.from_pretrained('roberta-base')
		elif basenet == 'deberta':
			print("Tokenizer: microsoft/deberta-base\n")
			self.token = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
		
		self.max_length = max_length
		self.segment_id = torch.tensor([1] * self.max_length).view(1, -1)

		
	def read_file(self, filename, stop_words):
		df = pd.read_csv(filename)

		# removing stop-words
		# https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/
		if stop_words:
			for i in range(len(df)):
				text = df.iloc[i]['text']
				text = text.split(' ')
				filtered_text =[] 

				for word in text:
					lexeme = self.nlp.vocab[word]
					if lexeme.is_stop == False:
						filtered_text.append(word) 
				# print("original", text)
				text = ' '.join(filtered_text)
				# print("after", filtered_text)
				# print("after", text)
				df.loc[i, 'text'] = text

		# replace all NaN with 0
		# will be used dring loss function computation
		df = df.fillna(0)

		if not self.is_test:
			df['humor_controversy'] = df['humor_controversy'].astype('int')

		# print(df.shape)
		print("Sampled input from the file: {}".format(filename))
		print(df.head())

		return df

	
	def get_tokenized_text(self, text):		
		# marked_text = "[CLS] " + text + " [SEP]"
		encoded = self.token(text= text,  					# the sentence to be encoded
							 add_special_tokens= True,  	# add [CLS] and [SEP]
							 max_length= self.max_length,  	# maximum length of a sentence
							 padding= 'max_length',  		# add [PAD]s
							 return_attention_mask= True,  	# generate the attention mask
							 return_tensors = 'pt',  		# return PyTorch tensors
							 truncation= True
							) 

		input_id = encoded['input_ids']
		mask_id  = encoded['attention_mask']

		return input_id, mask_id

		
	def __len__(self):
		return len(self.data)
	

	def __getitem__(self, idx):
		
		text  = self.data.iloc[idx]['text']

		label = []

		if not self.is_test:
			label.append(self.data.iloc[idx]['is_humor'])
			label.append(self.data.iloc[idx]['humor_controversy'])
			label.append(self.data.iloc[idx]['humor_rating'])
			label.append(self.data.iloc[idx]['offense_rating'])

		else:
			label.append(self.data.iloc[idx]['id'])

		label = torch.tensor(label)

		input_id, mask_id  = self.get_tokenized_text(text)
		
		return [input_id, mask_id, self.segment_id], label