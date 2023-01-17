# Training and evaluation of the model

# My import
from .datasets.BPSFH_Qiaoyu import BPSFH
from .models.model_assembler import ModelComplete
from . import tools
from . import evaluation

# Regular import
import psutil

import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
torch.manual_seed(0)

###############################
#     Program Entry Point     #
###############################

def main(args_in):
	parser = argparse.ArgumentParser("Training")

	parser.add_argument('--dataset', help="name of the dataset, ('BPSFH')", default="BPSFH", type=str)
	
	parser.add_argument('--batch_size', help="number of samples in each batch", default = 8, type=int)
	parser.add_argument('--segment_max_len', help="max number of frames in a segment", default = 8, type=int)

	parser.add_argument('--decode_type', help="type of the decoder, (softmax, nade, semi_crf)", default = "softmax", type=str)
	parser.add_argument('--harmony_type', help="type of the harmony representation, (CHORD, RN)", default = "RN", type=str)


	# Extract the information from the input arguments
	args = parser.parse_args(args_in)
	dataset_name = args.dataset

	batch_size = args.batch_size
	segment_max_len = args.segment_max_len
	
	decode_type = args.decode_type
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
						  reset_data=True,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  seed=0)

	print(f"Preparing validation dataset...")
	validation_dataset = BPSFH(base_dir=None,
						  splits=validation_splits,
						  reset_data=False,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  seed=0)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

	# Initialize the device
	device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

	# Initialize the model and the optimizer
	model = ModelComplete(batch_size, segment_max_len, harmony_type, decode_type, device)
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.0001)

	evaluator = evaluation.Evaluator(decode_type, harmony_type, batch_size)



	# The number of total parameters
	#total_params = sum(param.numel() for param in model.parameters())


	# Initialize a tensor board
	tb = SummaryWriter()

	# The first epoch is not in the eval mode
	eval_epoch = False

	# The training process consists of 100 epochs
	for epoch in range(1,1001):
		print(f"\n\n\nepoch: {epoch}")



		evaluator.initialize()

		# Run evaluation every 5 epochs
		if epoch%10 == 0:
			eval_epoch = True
		

		################
		#   Training   #
		################

		# Keep track of the number of batches used in the current epoch 
		num_train_batch = 0
		for batch_idx, train_batch in enumerate(train_loader):
			print(f"Memory used: {psutil.Process().memory_info().rss / (1024 * 1024)}MB")
			
			num_train_batch += 1
			

			# Forward pass
			optimizer.zero_grad()

			print(f"\nTraining... epoch: {epoch}, batch: {batch_idx}")
			train_loss = model.get_loss(train_batch)
			print(f"Training loss: {train_loss}")

			# Backward propogation
			print("Back propagation")
			train_loss.backward()


			# Parameter update
			print("Updating weights")
			optimizer.step()


			# Run evaluation on the training dataset
			if eval_epoch:

				# Disable the gradient during evaluation
				with torch.no_grad():
					print("Evaluating on the training batch")

					decoded_result_train = model.decoder.decode()
					evaluator.update_train(train_batch, train_loss, decoded_result_train)


		##################
		#   Validation   #
		##################
		
		# Run evaluation on the validation dataset
		if eval_epoch:
			validation_loss_all = []
			validation_acc_root_all = []
			validation_acc_quality_all = []
			validation_acc_overall_all = []
			
			num_validation_batch = 0
			for batch_idx, validation_batch in enumerate(validation_loader):
				num_validation_batch += 1
				
				with torch.no_grad():
					print(f"\nValidation... epoch: {epoch}, batch: {batch_idx}")
					print("Evaluating on the validation batch")
					
					validation_loss = model.get_loss(validation_batch)
					print(f"Validation loss: {validation_loss}")

					decoded_result_validation = model.decoder.decode()
					evaluator.update_validation(validation_batch, validation_loss, decoded_result_validation)
			

			####################################
			#   Report Evaluation statistics   #
			####################################
			evaluator.summarize_epoch(epoch, num_train_batch, num_validation_batch)
			eval_epoch = False




if __name__ == "__main__":
	main(sys.argv[1:])





