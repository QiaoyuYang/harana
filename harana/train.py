# scripts to train the model
from .datasets.BPSFH_processor import BPSFHDataset

import sys
import torch
import torch.optim as optim
import argparse


def main(args_in):
	parser = argparse.ArgumentParser("Training")

	parser.add_argument('--dataset', help="name of the dataset, ('BPSFH')", default = "BPSFH", type=str)
	parser.add_argument('--frame_type', help="type of the representation to encode the basic time unit, ('inter_onset', 'fixed_size')", default = "fixed_size", type=str)
	parser.add_argument('--fpqn', help="number of frames per quarter note", default = 12, type=int)
	
	args = parser.parse_args(args_in)

	# the root directory of all the datasets
	dataset_root_dir = 'harana/datasets'
	
	dataset_name = args.dataset
	frame_type = args.frame_type
	fpqn = args.fpqn

	# generate a customized pytorch Dataset with the specified dataset name
	dataset = BPSFHDataset(dataset_root_dir, frame_type, fpqn)




if __name__ == "__main__":
	main(sys.argv[1:])





