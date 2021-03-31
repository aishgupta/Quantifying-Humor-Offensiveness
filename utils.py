# TODO: Change the name of emotion_module as it is conflicting
# with weight loading

import os
import sys
import argparse
import warnings

import pandas as pd
import numpy as np
import matplotlib.image as npimg
import matplotlib.pyplot as plt
# import cv2

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable


def load_weights(model, resume_model= None):
	'''
		Load saved model state dictionary
		Input:
			resume_model : path of the saved checkpoint to be loaded
		Output:
			model        : weight loaded model
			start_epoch  : starting epoch of the training 
	'''
	print("=> Loading model weights from {}".format(resume_model))
	ckpt = torch.load(resume_model, map_location=torch.device('cpu'))

	if 'state_dict' in ckpt:
		state_dict = ckpt['state_dict']
	else:
		state_dict = ckpt['model']
	
	# state_dict = ckpt
	
	# state_dict_pretrained = {k.replace('module.', ""): v for k,v in state_dict.items()}
	state_dict_pretrained = state_dict
	# print(model)
	# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
	model_dict = model.state_dict()

	state_dict = {k: v for k, v in state_dict_pretrained.items() if (k in model_dict) and (v.shape == model_dict[k].shape)}

	# overwrite entries in the existing state dict
	model_dict.update(state_dict) 

	# load the new state dict
	model.load_state_dict(model_dict)

	print("Length of model-dict  : {}".format(len(model_dict)))	
	print("Length of loaded dict : {}".format(len(state_dict)))

	if len(model_dict)!=len(state_dict):
		not_in_state_dict = {k: v.shape for k, v in model_dict.items() if k not in state_dict}
		print("Layers which are in model-dict but not in loaded state-dict:")
		print(not_in_state_dict)

		not_in_model_dict = {k: v.shape for k, v in state_dict_pretrained.items() if (k not in state_dict)}
		print("Layers which are in loaded state-dict but not in model-dict:")
		print(not_in_model_dict)
	
	start_epoch = ckpt['epoch'] + 1
	# start_epoch = 1
	return model, start_epoch


def n_trainable(model):
	# check whether requires_grad is True?
	
	trainable_parameters = 0
	for i,j in enumerate(model.parameters(), 1):
		trainable_parameters += j.requires_grad

	if(trainable_parameters == i):
		print("All the layers of the model are getting trained")
	else:
		print("Some layers are frozen.")
		print("Number of Layer parameters :{} \t Trainable layer parameters:{}".format(i, trainable_parameters))

	# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Count of trainable-parameters: ", pytorch_total_params)


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param.shape) == 1 or name in skip_list: # bias/bn term or selected layer
			# print("Not applying weight decay to", name)
			no_decay.append(param)
		else:
			decay.append(param)
	return [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay   , 'weight_decay': weight_decay}]


def save_model(model, optimizer, opt, epoch, save_file):
	if isinstance(model, torch.nn.DataParallel):
		state = {
			'opt': opt,
			'model': model.cpu().module.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch,
		}
	else:
		state = {
			'opt': opt,
			'model': model.cpu().state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch,
		}
	torch.save(state, save_file)


def get_lr(optimizer):
	"""
		Returns the learning rate for printing
	"""
	for param_group in optimizer.param_groups:
		return param_group['lr']


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def print_args(args, logger=None):
	"""
		Print the argument parser
	"""
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<30} : {}'.format(k, v))
		else:
			print('{:<30} : {}'.format(k, v))


def print_header(str):
	"""
		Print any string with specific format
	"""
	print("\n")
	print("="*80)
	print(" "*30, str)
	print("="*80)


def create_folder(output_folder_path):
	"""
		Tries to create a folder if folder is not present
	"""
	if os.path.exists(output_folder_path):
		print("Directory exists {}".format(output_folder_path))
		return 0
	else:
		print("Creating directory {}".format(output_folder_path))
		os.makedirs(output_folder_path)
		return 1