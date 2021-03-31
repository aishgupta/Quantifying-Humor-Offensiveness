# ******************************************************************
# Author: Aishwarya
# 
# Main file to train, validate and evaluate the model for
# SemEval 2021 Task 7
# 
# Sample run: python main.py
# 
# ******************************************************************

import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score, accuracy_score

from utils import *
from datasets import *
from architectures import *
from train_test import *

#===============================================================================
# Argparse
#===============================================================================

parser = argparse.ArgumentParser(description = 'Humor Analysis')

# Model and dataset selection parameters
parser.add_argument('--basenet',           type = str,            default = 'bert',       help = 'basenet')

# Learning hyperparameters
parser.add_argument('--batch_size',        type = int,            default = 32,           help = 'training batch-size (default: 128)')
parser.add_argument('--epochs',            type = int,            default = 10,           help = 'number of epochs (default: 30)')
parser.add_argument('--lr_max',            type = float,          default = 0.00002,      help = 'learning rate (default: 0.01)')

# Initialization
parser.add_argument('--resume_model',      type = str,            default = None,         help = 'resume model training')
parser.add_argument('--seed',              type = int,            default = 0,            help = 'random seed (default: 1)')

# Model saving and testing parameters
parser.add_argument('--run_iter',          type = str,            default = '100',       help = 'maintains the number of the experiment')
parser.add_argument('--test_model',      action = 'store_true',   default = False,        help = 'test model')
parser.add_argument('--test_batch_size',   type = int,            default = 128,          help = 'testing batch-size')

# Whether to use GPU or not
parser.add_argument('--no_cuda',         action = 'store_true',   default = False,        help = 'disables CUDA training')


args = parser.parse_args()
warnings.filterwarnings("ignore")
print_header("Arguments")
print_args(args)

#===============================================================================
# Device Selction (GPU/CPU) and Seed Initialization
#===============================================================================
use_cuda = not args.no_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
kwargs   = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(args.seed)
if(use_cuda):
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.benchmark = True
np.random.seed(args.seed)

print("\nDevice: " + str(device) +"; Seed: "+str(args.seed))


#===============================================================================
# Dataset and Model Selection
#===============================================================================
save_dir = os.path.join('model_weights', 'run_{}'.format(args.run_iter))
create_folder(save_dir)

trainfile = '../dataset/hahackathon/train_dev_test.csv'
valfile   = '../dataset/hahackathon/val10.csv'
testfile  = '../dataset/hahackathon/public_test.csv'

traindata    = torch.utils.data.DataLoader(dataset= Hahackathon(trainfile, basenet= args.basenet, )              , batch_size= args.batch_size     , shuffle= True , **kwargs)
valdata      = torch.utils.data.DataLoader(dataset= Hahackathon(valfile  , basenet= args.basenet, )              , batch_size= args.test_batch_size, shuffle= False, **kwargs)
testdata     = torch.utils.data.DataLoader(dataset= Hahackathon(testfile , basenet= args.basenet, is_test= True) , batch_size= args.test_batch_size, shuffle= False, **kwargs)

# print("Size of the dataset\nTrain: {}, Validation: {}, Test: {}".format(len(traindata.dataset), len(valdata.dataset)))#, len(testdata.dataset)))
print("Size of the dataset\nTrain: {}, Validation: {}\n".format(len(traindata.dataset), len(valdata.dataset)))

print("Using multitask model with two separate heads for classification and regression")
model = multitask_lstm_fc(basenet= args.basenet)
# model = multitask_fc(basenet= args.basenet)

if torch.cuda.device_count() > 1:
	print("Using {} GPUs! \n".format(torch.cuda.device_count()))
	model = torch.nn.DataParallel(model)
model = model.to(device)

if args.resume_model is not None:
	model, start_epoch = load_weights(model, args.resume_model)
else:
	start_epoch = 1


#===============================================================================
# Training/Testing starts
#===============================================================================
writer = SummaryWriter(log_dir= save_dir)

if not args.test_model:

	print_header("Training starts")
	
	optimizer = torch.optim.Adam(model.parameters(), lr= args.lr_max)
	
	for ep in range(start_epoch, start_epoch+args.epochs):
		
		train_loss         = train(model, traindata, optimizer, device)
		test_loss, metrics = test(model, valdata, device)

		writer.add_scalar(               'Loss/train', train_loss, ep)
		writer.add_scalar(                 'Loss/val', test_loss , ep)
		writer.add_scalar(          'is_humor_f1/val', metrics[0], ep)
		writer.add_scalar(         'is_humor_acc/val', metrics[1], ep)
		writer.add_scalar( 'humor-controversy_f1/val', metrics[2], ep)
		writer.add_scalar('humor-controversy_acc/val', metrics[3], ep)
		writer.add_scalar(         'humor_rating/val', metrics[4], ep)
		writer.add_scalar(       'offense_rating/val', metrics[5], ep)

		save_path = os.path.join(save_dir, "epoch_{}".format(ep) + ".pth")
		save_model(model, optimizer, opt= None, epoch= ep, save_file= save_path)
		model.to(device)

		print("Ep: {}\t Lr: {:.6f}\t Train Loss: {:.4f}\t Val Loss: {:.4f}\t Val: Humor: {:.4f}/{:.4f}, Controversy: {:.4f}/{:.4f}, Humor-rating: {:.4f}, Offensive-rating: {:.4f} ".format(ep, get_lr(optimizer), train_loss, test_loss, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

else:
	ep = start_epoch

print_header("Testing on the Validation Dataset")
test_loss, metrics = test(model, valdata, device)
print("Val Loss: {:.4f}\t Val: Humor: {:.4f}/{:.4f}, Controversy: {:.4f}/{:.4f}, Humor-rating: {:.4f}, Offensive-rating: {:.4f} ".format(test_loss, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

print_header("Generating prediction file for post-evaluation phase")
filename   = "results_publictest.csv"
get_result_on_evaldata(model, testdata, filename, device= device)