# ******************************************************************
# 
# Contains different multitask architectures used for training the model
# 
# temporal   : Embeddings followed by a LSTM layer
# 
# ******************************************************************

# Dropout is placed after (fc output + ReLU)
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

# No activation after LSTM layer
# https://towardsdatascience.com/reading-between-the-layers-lstm-network-7956ad192e58

# Using dropout in the LSTM layer

import sys
import torch

from transformers import BertModel, AutoModel, RobertaModel, DebertaForSequenceClassification, DebertaModel

dropout_prob= 0.2

class BERT_based_classifier(torch.nn.Module):
	'''
		Base class inherited by different architectures used for multitask learning
	'''
	
	def __init__(self, basenet= 'bert', bert_freeze= False, temporal= False, n_outputs= 2, penultimate= False, fc_dim= 256, lstm_dim= 256, one_layer= False, no_classfier= False):      
		
		super(BERT_based_classifier, self).__init__()

		# Load pre-trained model (weights)
		self.basenet = basenet
		if basenet == 'bert':
			print("Base architecture: bert-base-uncased")
			self.encoder = BertModel.from_pretrained('bert-base-uncased',
													  output_hidden_states = False, # Whether the model returns all hidden-states.
													)
		elif basenet == 'ernie':
			print("Base architecture: ernie-2.0-en")
			self.encoder = AutoModel.from_pretrained("nghuyong/ernie-2.0-en",
													  output_hidden_states = False
													)
		elif basenet == 'roberta':
			print("Base architecture: roberta-base")
			self.encoder = RobertaModel.from_pretrained('roberta-base',
														 output_hidden_states = False
													   )
		elif basenet == 'deberta':
			print("Base architecture: microsoft/deberta-base")
			self.encoder = DebertaModel.from_pretrained('microsoft/deberta-base',
														 output_hidden_states = False
														)

		if bert_freeze:
			print("Freezing BERT!")
			# freeze bert so that is is not finetuned
			for name, param in self.encoder.named_parameters():                
				if param.requires_grad is not None:
					param.requires_grad = False

		self.temporal = temporal
		if self.temporal:
			self.lstm       = torch.nn.LSTM(input_size= 768, hidden_size= lstm_dim, dropout= dropout_prob)
			self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features= fc_dim, out_features= n_outputs))

		else:
			if not one_layer:
				self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features= 768, out_features= fc_dim),
													  torch.nn.ReLU(),
													  torch.nn.Dropout(p= dropout_prob),
													  torch.nn.Linear(in_features= fc_dim, out_features= n_outputs))
			else:
				self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features= 768, out_features= n_outputs))

		
	def forward(self, input_id, mask_id, token_type_id):
		
		# if output_hidden_states == True:
		# feat.keys() = ['last_hidden_state', 'hidden_states']
		# if output_hidden_states == False:
		# feat.keys() = ['last_hidden_state', 'pooler_output']

		if self.temporal:
			if self.basenet == 'roberta':
				feat = self.encoder(input_ids= input_id, attention_mask= mask_id)['last_hidden_state']
			else:
				feat = self.encoder(input_ids= input_id, attention_mask= mask_id, token_type_ids= token_type_id)['last_hidden_state']

			feat    = feat.squeeze().permute(1, 0, 2)
			_, temp = self.lstm(feat)
			feat    = temp[0].squeeze()

		else:
			if self.basenet == 'roberta':
				feat = self.encoder(input_ids= input_id, attention_mask= mask_id)['pooler_output']
			else:
				feat = self.encoder(input_ids= input_id, attention_mask= mask_id, token_type_ids= token_type_id)['pooler_output']

		return feat


class multitask_fc(BERT_based_classifier):
	'''
	LLM embeddings followed by one/two FC layer with MCE/CE loss
	
	if one_layer:
		basenet -> FC -> MSE/CE
	else:
		basenet -> FC   -> ReLU -> Dropout -> FC -> MSE + CE
	'''

	def __init__(self, temporal= False, n_outputs= 6, fc_dim= 256, one_layer= False):      
		
		super(multitask_fc, self).__init__(temporal= temporal, n_outputs= n_outputs, fc_dim= fc_dim, one_layer= one_layer)
		
	def forward(self, input_id, mask_id, token_type_id, go_input_id= None, go_mask_id= None):
		
		feat   = super(multitask_fc, self).forward(input_id, mask_id, token_type_id)		#torch.Size([64, 256])		
		output = self.classifier(feat)
		return output


class multitask_lstm_fc(BERT_based_classifier):
	'''
	  Mutitask model with two separate branches for classification and regression task

	  temporal= True
	  basenet -> LSTM -> FC   -> CE
		  |
		   ----> FC   -> ReLU -> Dropout -> FC -> MSE
	'''

	def __init__(self, basenet= 'bert', temporal= True, bert_freeze= False, fc_dim= 256, in_features= 768):      

		super(multitask_lstm_fc, self).__init__(basenet= basenet, temporal= temporal, bert_freeze= False, n_outputs= 4, fc_dim= fc_dim)
		self.classifier2_1 = torch.nn.Sequential(torch.nn.Linear(in_features= 768, out_features= 256),
												 torch.nn.ReLU(),
												 torch.nn.Dropout(p= dropout_prob))

		self.classifier2_2 = torch.nn.Sequential(torch.nn.Linear(in_features= fc_dim, out_features= 2))
		
	def forward(self, input_id, mask_id, token_type_id, go_input_id= None, go_mask_id= None):
		
		feat1   = super(multitask_lstm_fc, self).forward(input_id, mask_id, token_type_id)		#torch.Size([64, 256])

		if self.basenet == 'roberta':
			feat2 = self.classifier2_1(self.encoder(input_ids= input_id, attention_mask= mask_id)['pooler_output'])
		else:
			feat2 = self.classifier2_1(self.encoder(input_ids= input_id, attention_mask= mask_id, token_type_ids= token_type_id)['pooler_output'])

		output1 = self.classifier (feat1)		
		output2 = self.classifier2_2(feat2)
									
		# torch.Size([128, 4]) torch.Size([128, 2]) torch.Size([128, 6])
		output  = torch.cat((output1, output2), dim= 1)

		return output