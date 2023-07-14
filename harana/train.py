# Training and evaluation of the model

# My import
from .tools import *
from .pipeline import Pipeline

# Regular import
import os
import sys
import torch
from torch.utils.data import DataLoader
import argparse
torch.manual_seed(0)

from harana.datasets.preprocessing import AllDatasets

###############################
#     Program Entry Point     #
###############################
	


def main(args_in):
	parser = argparse.ArgumentParser("Training")

	parser.add_argument('--dataset', help="name of the dataset, ('BPSFH')", default="BPSFH", type=str)
	
	parser.add_argument('--batch_size', help="number of samples in each batch", default = 256, type=int)
	parser.add_argument('--sample_size', help="number of frames in each batch", default = 96, type=int)
	parser.add_argument('--segment_max_len', help="number of samples in each batch", default = 24, type=int)
	parser.add_argument('--harmony_type', help="type of the harmony representation, (RQ, KDQ, KRQ, K)", default = "RQ", type=str)


	# Extract the information from the input arguments
	args = parser.parse_args(args_in)
	dataset_name = args.dataset

	batch_size = args.batch_size
	sample_size = args.sample_size
	segment_max_len = args.segment_max_len
	harmony_type = args.harmony_type


	#############
	#   Setup   #
	#############

	# Get the train-validation split of the dataset
	# Use the first available split as the validation dataset
	train_splits = AllDatasets.available_splits()
	validation_splits = [train_splits.pop(0)]

	print(f"Preparing training dataset...")
	# Generate the training and validation datasets


	train_dataset = AllDatasets(base_dir=None,
						  splits=train_splits,
						  sample_size=sample_size,
						  harmony_type=harmony_type,
						  reset_data=False,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  beat_as_unit=True,
						  validation=False,
						  seed=0)

	print(f"Preparing validation dataset...")
	validation_dataset = AllDatasets(base_dir=None,
						  splits=validation_splits,
						  sample_size=sample_size,
						  harmony_type=harmony_type,
						  reset_data=False,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  beat_as_unit=True,
						  validation=True,
						  seed=0)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)



	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



	'''
	note2rqpcsemicrf_lr = 0.001
	note2rqpcsemicrf_max_epoch = 500
	note2rqpcsemicrf_eval_period = 10
	note2rqpcsemicrf_pipeline = Pipeline(
					note_encoder_type='PC',
					harmony_encoder_type='PC',
					decoder_type='SemiCRF',
					harmony_type=harmony_type,
					batch_size=batch_size,
					sample_size=sample_size,
    				device=device,
    				lr=note2rqpcsemicrf_lr,
    				max_epoch=note2rqpcsemicrf_max_epoch,
    				eval_period=note2rqpcsemicrf_eval_period,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None,
    				pipeline_type=tools.PIPELINE_TYPE_TEST,
    				)
	
	note2rqpcsemicrf_pipeline.test(validation_loader)
	'''

	'''
	note2rqsoftmax_lr = 0.0001
	note2rqsoftmax_max_epoch = 1000
	note2rqsoftmax_eval_period = 10
	note2rqsoftmax_pipeline = Pipeline(
					note_encoder_type='CRNN',
					decoder_type='SoftMax',
					harmony_type=harmony_type,
					batch_size=batch_size,
					sample_size=sample_size,
    				device=device,
    				lr=note2rqsoftmax_lr,
    				max_epoch=note2rqsoftmax_max_epoch,
    				eval_period=note2rqsoftmax_eval_period,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None,
    				pipeline_type=tools.PIPELINE_TYPE_TRAIN,
    				)

	note2rqsoftmax_pipeline.train(train_loader, validation_loader)
	'''



	'''
	note2rqsoftmax_lr = 0.0001
	note2rqsoftmax_max_epoch = 1000
	note2rqsoftmax_eval_period = 10
	note2rqsoftmax_pipeline = Pipeline(
					note_encoder_type='CRNN',
					decoder_type='NADE',
					harmony_type=harmony_type,
					batch_size=batch_size,
					sample_size=sample_size,
    				device=device,
    				lr=note2rqsoftmax_lr,
    				max_epoch=note2rqsoftmax_max_epoch,
    				eval_period=note2rqsoftmax_eval_period,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None,
    				pipeline_type=tools.PIPELINE_TYPE_TRAIN,
    				)

	note2rqsoftmax_pipeline.train(train_loader, validation_loader)
	'''
	'''
	note2rqsemicrf_lr = 0.0001
	note2rqsemicrf_max_epoch = 1000
	note2rqsemicrf_eval_period = 10
	note2rqsemicrf_pipeline = Pipeline(
					note_encoder_type='PC',
					decoder_type='SemiCRF',
					harmony_type=harmony_type,
					batch_size=batch_size,
					sample_size=sample_size,
    				device=device,
    				lr=note2rqsemicrf_lr,
    				max_epoch=note2rqsemicrf_max_epoch,
    				eval_period=note2rqsemicrf_eval_period,
    				pretrain=False,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None,
    				pipeline_type=tools.PIPELINE_TYPE_TEST,
    				)

	note2rqsemicrf_pipeline.test(validation_loader)
	'''


	note2rqsemicrf_lr = 0.0001
	note2rqsemicrf_max_epoch = 10000
	note2rqsemicrf_eval_period = 10
	note2rqsemicrf_pipeline = Pipeline(
					note_encoder_type='CRNN',
					decoder_type='SemiCRF',
					harmony_type=harmony_type,
					batch_size=batch_size,
					sample_size=sample_size,
    				device=device,
    				lr=note2rqsemicrf_lr,
    				max_epoch=note2rqsemicrf_max_epoch,
    				eval_period=note2rqsemicrf_eval_period,
    				pretrain=False,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None,
    				pipeline_type=tools.PIPELINE_TYPE_TRAIN,
    				)
	
	note2rqsemicrf_pipeline.train(train_loader, validation_loader)



if __name__ == "__main__":
	main(sys.argv[1:])





