# Training and evaluation of the model

# My import
from .datasets.BPSFH_Qiaoyu import BPSFH
from .pipeline import Note2HarmonySoftmaxTrainer, Harmony2HarmonySoftmaxTrainer, Note2HarmonySemiCRFTrainer

# Regular import
import sys
import torch
from torch.utils.data import DataLoader
import argparse
torch.manual_seed(0)

###############################
#     Program Entry Point     #
###############################
	


def main(args_in):
	parser = argparse.ArgumentParser("Training")

	parser.add_argument('--dataset', help="name of the dataset, ('BPSFH')", default="BPSFH", type=str)
	
	parser.add_argument('--batch_size', help="number of samples in each batch", default = 8, type=int)
	parser.add_argument('--sample_size', help="number of samples in each batch", default = 48, type=int)
	parser.add_argument('--segment_max_len', help="number of samples in each batch", default = 16, type=int)
	parser.add_argument('--harmony_type', help="type of the harmony representation, (CHORD, RN)", default = "RN", type=str)


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
	train_splits = BPSFH.available_splits()
	validation_splits = [train_splits.pop(0)]

	print(f"Preparing training dataset...")
	# Generate the training and validation datasets
	train_dataset = BPSFH(base_dir=None,
						  splits=train_splits,
						  sample_size=sample_size,
						  harmony_type=harmony_type,
						  reset_data=True,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  seed=0)

	print(f"Preparing validation dataset...")
	validation_dataset = BPSFH(base_dir=None,
						  splits=validation_splits,
						  sample_size=sample_size,
						  harmony_type=harmony_type,
						  reset_data=False,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  seed=0)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

	# Initialize the device
	device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

	'''
	note2harmonysoftmax_lr = 0.001
	note2harmonysoftmax_max_epoch = 1000
	note2harmonysoftmax_eval_period = 50
	note2harmonysoftmax_trainer = Note2HarmonySoftmaxTrainer(
					batch_size=batch_size,
					sample_size=sample_size,
					harmony_type=harmony_type,
    				device=device,
    				lr=note2harmonysoftmax_lr,
    				max_epoch=note2harmonysoftmax_max_epoch,
    				eval_period=note2harmonysoftmax_eval_period,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None
    				)
	
	note2harmonysoftmax_trainer.train(train_loader, validation_loader)
	'''
	'''
	harmony2harmonysoftmax_lr = 0.01
	harmony2harmonysoftmax_max_epoch = 100
	harmony2harmonysoftmax_eval_period = 10
	harmony2harmonysoftmax_trainer = Harmony2HarmonySoftmaxTrainer(
					batch_size=batch_size,
					sample_size=sample_size,
					harmony_type=harmony_type,
    				device=device,
    				lr=harmony2harmonysoftmax_lr,
    				max_epoch=harmony2harmonysoftmax_max_epoch,
    				eval_period=harmony2harmonysoftmax_eval_period,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				save_loc=None
    				)
	
	harmony2harmonysoftmax_trainer.train(train_loader, validation_loader)
	'''


	note2harmonysemicrf_lr = 0.001
	note2harmonysemicrf_max_epoch = 1000
	note2harmonysemicrf_eval_period = 10
	note2harmonysemicrf_trainer = Note2HarmonySemiCRFTrainer(
					batch_size=batch_size,
					sample_size=sample_size,
					harmony_type=harmony_type,
    				device=device,
    				lr=note2harmonysemicrf_lr,
    				max_epoch=note2harmonysemicrf_max_epoch,
    				eval_period=note2harmonysemicrf_eval_period,
    				store_model=True,
    				save_model=True,
    				reset_model=True,
    				pretrain=True,
    				save_loc=None
    				)

	
	note2harmonysemicrf_trainer.train(train_loader, validation_loader)





if __name__ == "__main__":
	main(sys.argv[1:])





