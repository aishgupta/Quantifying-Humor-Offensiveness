# ******************************************************************
# 
# Contains train and test modules with multitask loss function
# 
# Use "get_result_on_evaldata" to generate predictions on the 
# evaluation data for challenge leaderboard
# 
# ******************************************************************

import torch
import torch.nn.functional as F
import sys

from utils import *

from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score, accuracy_score
 
mse_criterion = torch.nn.MSELoss()
ce_criterion  = torch.nn.CrossEntropyLoss()


def concat_output(output, device):
	temp = torch.tensor([]).to(device)
	
	if output.shape[1] == 2:
		temp = torch.argmax(output[:, 0:2], dim=1).view(-1,1)
		return temp

	temp = torch.argmax(output[:, 0:2], dim=1).view(-1,1)
	temp = torch.cat((temp, torch.argmax(output[:, 2:4], dim= 1).view(-1,1)), dim= 1)
	temp = torch.cat((temp, output[:, 4].view(-1,1)), dim= 1)
	temp = torch.cat((temp, output[:, 5].view(-1,1)), dim= 1)
	return temp


def loss_func(output, target, device):

	humor_id = (target[:, 0]==1)
	target   = target.float()

	loss1 = ce_criterion(output[:, 0:2], target[:, 0].long())	                	# is_humor binary classification
	loss2 = ce_criterion(output[humor_id][:, 2:4], target[humor_id][:, 1].long())	# humor_controversy binary classification
	loss3 = mse_criterion(output[humor_id][:, 4], target[humor_id][:, 2])			# humor rating regression loss
	loss4 = mse_criterion(output[:, 5], target[:, 3])								# offensive rating regression loss
	loss  = loss1+loss2+loss3+loss4

	# this block of generating temp is verified throughly
	temp = concat_output(output, device)
	temp = temp.detach()

	return loss, temp


def get_result_on_evaldata(model, dataloader, filename, device):
	model.eval()
	outputs  = torch.tensor([]).to(device)
	text_ids = torch.tensor([]).to(device)

	with torch.no_grad():
		for batch_id, batch_data in enumerate(dataloader):
			data       = batch_data[0]
			text_id    = batch_data[1].to(device)

			token_id   = data[0].to(device).squeeze()
			mask_id    = data[1].to(device).squeeze()
			segment_id = data[2].to(device).squeeze()

			output = model(token_id, mask_id, segment_id)

			temp     = concat_output(output, device)
			outputs  = torch.cat((outputs, temp), 0)
			text_ids = torch.cat((text_ids, text_id), 0)
	
	print(outputs.shape, text_ids.shape)
	outputs  = outputs.cpu().numpy()
	text_ids = text_ids.cpu().numpy()

	results = np.hstack((text_ids, outputs))
	df      = pd.DataFrame(data    = results,  
	                       columns = ['id', 'is_humor', 'humor_controversy', 'humor_rating', 'offense_rating']) 

	df['id']                = (df['id']).astype('int')
	df['is_humor']          = (df['is_humor']).astype('int')
	df['humor_controversy'] = (df['humor_controversy']).astype('int')
	df = df[['id', 'is_humor', 'humor_rating', 'humor_controversy', 'offense_rating']]
	
	df.to_csv(filename, index = False)
	print("Result file is saved to {}".format(filename))


def test(model, dataloader, device):
	model.eval()
	test_loss = AverageMeter()
	outputs   = torch.tensor([]).to(device)
	targets   = torch.tensor([]).to(device)

	with torch.no_grad():
		for batch_id, batch_data in enumerate(dataloader):
			data       = batch_data[0]
			target     = batch_data[1].to(device)

			token_id   = data[0].to(device).squeeze()
			mask_id    = data[1].to(device).squeeze()
			segment_id = data[2].to(device).squeeze()

			output     = model(token_id, mask_id, segment_id)

			loss, temp = loss_func(output, target, device= device)
			outputs    = torch.cat((outputs, temp)  , 0)
			targets    = torch.cat((targets, target), 0)

			test_loss.update(loss.item(), target.shape[0])

	outputs = outputs.cpu().numpy()
	targets = targets.cpu().numpy()

	humor_fscore          = f1_score          (targets[:, 0], outputs[:, 0])
	controversy_fscore    = f1_score          (targets[:, 1], outputs[:, 1])
	humor_acc             = accuracy_score    (targets[:, 0], outputs[:, 0])
	controversy_acc       = accuracy_score    (targets[:, 1], outputs[:, 1])
	humor_rating_rmse     = mean_squared_error(targets[:, 2], outputs[:, 2], squared= False)
	offensive_rating_rmse = mean_squared_error(targets[:, 3], outputs[:, 3], squared= False)

	return test_loss.avg, [humor_fscore, humor_acc, controversy_fscore, controversy_acc, humor_rating_rmse, offensive_rating_rmse]


def train(model, dataloader, optimizer, device):
	model.train()
	train_loss = AverageMeter()

	for batch_id, batch_data in enumerate(dataloader):
		data       = batch_data[0]
		target     = batch_data[1].to(device)

		input_id   = data[0].to(device).squeeze()
		mask_id    = data[1].to(device).squeeze()
		segment_id = data[2].to(device).squeeze()

		output = model(input_id, mask_id, segment_id)

		loss, _ = loss_func(output, target, device= device)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss.update(loss.item(), target.shape[0])

	return train_loss.avg