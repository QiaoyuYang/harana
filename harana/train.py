# Training and evaluation of the model

# My import
from .datasets.BPSFH_processor import BPSFHDataset
from .datasets.BPSFH import BPSFH
from .models.model_assembler import ModelComplete
from .utils import chord, eval_utils

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
	
	parser.add_argument('--batch_size', help="number of samples in each batch", default=16, type=int)
	parser.add_argument('--sample_size', help="number of frames in each sample", default=8, type=int)
	parser.add_argument('--segment_max_len', help="max number of frames in a segment", default=8, type=int)
	
	parser.add_argument('--frame_type', help="type of the representation to encode the basic time unit, ('inter_onset', 'fixed_size')", default="fixed_size", type=str)
	parser.add_argument('--tpqn', help="number of ticks per quarter note", default=24, type=int)
	parser.add_argument('--fpqn', help="number of frames per quarter note if fixed-sized frames are used", default=4, type=int)

	parser.add_argument('--num_label', help="number of chord labels", default=120, type=int)

	parser.add_argument('--note_transform_type', help="type of the note transform, (none, cnn, dense_gru)", default="none", type=str)
	parser.add_argument('--chord_transform_type', help="type of the chord transform, (weight_vector, fc1, fc2)", default="fc1", type=str)
	parser.add_argument('--decode_type', help="type of the decoder, (softmax, nade, semi_crf)", default="semi_crf", type=str)
	parser.add_argument('--label_type', help="type of the chord label, (root_quality, key_rn)", default="root_quality", type=str)

	parser.add_argument('--embedding_size', help="dimension size of the embedding", default="13", type=int)

	# Extract the information from the input arguments
	args = parser.parse_args(args_in)
	dataset_name = args.dataset
	num_label = args.num_label
	batch_size = args.batch_size
	sample_size = args.sample_size
	segment_max_len = args.segment_max_len
	frame_type = args.frame_type
	tpqn = args.tpqn
	fpqn = args.fpqn
	note_transform_type = args.note_transform_type
	chord_transform_type = args.chord_transform_type
	decode_type = args.decode_type
	label_type = args.label_type
	embedding_size = args.embedding_size


	#############
	#   Setup   #
	#############
	# The root directory of all the datasets
	dataset_root_dir = 'harana/datasets'


	# Generate a customized pytorch Dataset with the specified dataset name
	#dataset_bpsfh = BPSFHDataset(dataset_root_dir, sample_size, frame_type, tpqn, fpqn)
	dataset_bpsfh = BPSFH(base_dir=None,
						  tracks=None,
						  ticks_per_quarter=tpqn,
						  frames_per_quarter=fpqn,
						  frames_per_sample=sample_size,
						  reset_data=False,
						  store_data=True,
						  save_data=True,
						  save_loc=None,
						  seed=0)

	# The proportion of the dataset that is used to train
	train_proportion = 0.9
	train_size = round(len(dataset_bpsfh) * train_proportion)

	# The rest is used for validation
	validation_size = len(dataset_bpsfh) - train_size
	
	# Split the dataset and create corresponding DataLoaders
	train_dataset, validation_dataset = random_split(dataset_bpsfh, [train_size, validation_size])
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

	while True:
		for batch in train_loader:
			print()

	# Initialize the device
	device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

	# Initialize the model and the optimizer
	model = ModelComplete(batch_size, sample_size, segment_max_len, num_label, note_transform_type, chord_transform_type, decode_type, label_type, embedding_size, device)
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.005)

	evaluator = eval_utils.Evaluator(decode_type, label_type)



	# The number of total parameters
	#total_params = sum(param.numel() for param in model.parameters())


	# Test code for the forward pass
	#train_batch = next(iter(train_loader))
	#train_loss = model.get_loss(train_batch)
	#decoded_result_train = model.decoder.decode()
	#train_acc_root, train_acc_quality, train_acc_overall = eval_utils.ComputeEval(train_batch, decoded_result_train, decode_type, label_type, batch_size, sample_size)
	#decoded_segments_train = model.decoder.semicrf.decode()
	#print(decoded_segments_train[3])


	# Initialize a tensor board
	tb = SummaryWriter()

	# The first epoch is not in the eval mode
	eval_epoch = False

	# The training process consists of 100 epochs
	for epoch in range(1,201):
		print(f"\n\n\nepoch: {epoch}")

		evaluator.initialize()

		# Run evaluation every 5 epochs
		if epoch%5 == 1:
			eval_epoch = True
		

		################
		#   Training   #
		################

		# Keep track of the number of batches used in the current epoch 
		num_train_batch = 0
		for batch_idx, train_batch in enumerate(train_loader):
			print(f"Memory used: {psutil.Process().memory_info().rss / (1024 * 1024)}MB")

			# A batch is only used if it is a complete batch with the designated size
			if train_batch['pc_dist_seq'].shape[0] != batch_size:
				continue
			else:
				num_train_batch += 1
			

			# Forward pass
			optimizer.zero_grad()

			print(f"\nTraining... epoch: {epoch}, batch: {batch_idx}")
			train_loss = model.get_loss(train_batch)


			print(f"Training loss: {train_loss}")
			#train_loss_all.append(train_loss.item())

			#print(torch.cuda.memory_summary(device=None, abbreviated=False))

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
					print(model.decoder.semicrf.segment_score[0])
					print(model.decoder.semicrf.transitions)
					print(decoded_result_train[0])
					evaluator.update_train(train_batch, train_loss, decoded_result_train, decode_type, label_type, batch_size, sample_size)


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
				if validation_batch['pc_dist_seq'].shape[0] != batch_size:
					continue
				else:
					num_validation_batch += 1
				
				with torch.no_grad():
					print(f"\nValidation... epoch: {epoch}, batch: {batch_idx}")
					print("Evaluating on the validation batch")
					
					validation_loss = model.get_loss(validation_batch)
					decoded_result_validation = model.decoder.decode()
					evaluator.update_validation(validation_batch, validation_loss, decoded_result_validation, decode_type, label_type, batch_size, sample_size)
			

			####################################
			#   Report Evaluation statistics   #
			####################################
			evaluator.summarize_epoch(epoch, num_train_batch, num_validation_batch)
			eval_epoch = False


if __name__ == "__main__":
	main(sys.argv[1:])





