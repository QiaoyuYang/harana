# Training and evaluation of the model

# My import
from .datasets.BPSFH_processor import BPSFHDataset
from .models.model_assembler import ModelComplete
from .utils import chord

# Regular import
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
torch.manual_seed(0)


#############################
#   Evaluation statistics   #
#############################

# Wrapper for computing the evaluation statistics
def ComputeEval(chord_seq_gt, decoded_segments, batch_size, sample_size):
    
    print('Computing accuracy ...')

    # Convert each predicted segment sequence to a frame-level chord sequence
    chords_seq_pred = segments2chord_seq(decoded_segments, batch_size, sample_size)
    
    # Get the average accuracy across the batch
    root_acc, quality_acc, overall_acc = get_accuracy(chord_seq_gt, chords_seq_pred, batch_size, sample_size)
    
    # And return them
    return root_acc, quality_acc, overall_acc


# Convert each predicted segment sequence to a frame-level chord sequence
def segments2chord_seq(segments_pred, batch_size, sample_size):


	frame_idx = 0

	# The list of chord sequences of all samples in the batch
	chord_seq = []

	# The chord label sequence of a single sample
	chord_seq_sample = []
	
	# Loop through samples in the batch
	for sample_idx in range(batch_size):

		# Loop through predicted segments in the sample
		for segment in segments_pred[sample_idx]:

			# Extract the label index, and boundary frames of the chord in the segment
			label_idx, segment_start_frame, segment_end_frame = segment

			# Append the chord label index of each frame in the segment to the chord sequence of the sample
			for i in range(segment_start_frame, segment_end_frame + 1):
				chord_seq_sample.append(label_idx)
		
		# Append the complete chord sequence of the sample to the list of all chord sequences in the batch
		chord_seq.append(chord_seq_sample)

	# Return the result as a tensor
	return torch.tensor(chord_seq)


# Get the average accuracy across the batch
def get_accuracy(chord_seq_gt, chord_seq_pred, batch_size, sample_size):

	# The list to store each type of accuracy
	root_acc_all = []
	quality_acc_all = []
	overall_acc_all = []
	
	# Loop through samples
	for sample_idx in range(batch_size):

		# The variables to count number of correct frames in the sample
		root_correct = 0
		quality_correct = 0
		overall_correct = 0

		# Loop through frames
		for frame_idx in range(sample_size):

			# Get the root, quality and bass of the current gt chord
			symbol_gt = chord.index2symbol(int(chord_seq_gt[sample_idx, frame_idx].item()))
			root_pc_gt, quality_gt, bass_pc_gt = chord.parse_symbol(symbol_gt)

			# Get the root, quality and bass of the current predicted chord
			symbol_pred = chord.index2symbol(int(chord_seq_pred[sample_idx, frame_idx].item()))
			root_pc_pred, quality_pred, bass_pc_pred = chord.parse_symbol(symbol_pred)

			# Check if the predicted chord is correct
			if root_pc_gt == root_pc_pred:
				root_correct += 1
			if quality_gt == quality_pred:
				quality_correct += 1
			if root_pc_gt == root_pc_pred and quality_gt == quality_pred:
				overall_correct += 1

		# Calculte the accuracy of each type
		root_acc_sample = root_correct / sample_size
		quality_acc_sample = quality_correct / sample_size
		overall_acc_sample = overall_correct / sample_size

		# Append them to the list of accuracies of all samples in the batch
		root_acc_all.append(root_acc_sample)
		quality_acc_all.append(quality_acc_sample)
		overall_acc_all.append(overall_acc_sample)
	
	# Return the average accuracy across the batch
	return sum(root_acc_all) / batch_size, sum(quality_acc_all) / batch_size, sum(overall_acc_all) / batch_size




###############################
#     Program Entry Point     #
###############################

def main(args_in):
	parser = argparse.ArgumentParser("Training")

	parser.add_argument('--dataset', help="name of the dataset, ('BPSFH')", default = "BPSFH", type=str)
	
	parser.add_argument('--batch_size', help="number of samples in each batch", default = 64, type=int)
	parser.add_argument('--sample_size', help="number of frames in each sample", default = 48, type=int)
	parser.add_argument('--segment_max_len', help="max number of frames in a segment", default = 8, type=int)
	
	parser.add_argument('--frame_type', help="type of the representation to encode the basic time unit, ('inter_onset', 'fixed_size')", default = "fixed_size", type=str)
	parser.add_argument('--tpqn', help="number of ticks per quarter note", default = 24, type=int)
	parser.add_argument('--fpqn', help="number of frames per quarter note if fixed-sized frames are used", default = 4, type=int)

	parser.add_argument('--num_label', help="number of chord labels", default = 144, type=int)

	parser.add_argument('--note_transform_type', help="type of the note transform, (none, cnn, dense_gru)", default = "none", type=str)
	parser.add_argument('--chord_transform_type', help="type of the chord transform, (none, weight_vector, fc1, fc2)", default = "none", type=str)
	parser.add_argument('--decode_type', help="type of the decoder, (softmax, nade, semi_crf)", default = "semi_crf", type=str)

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


	#############
	#   Setup   #
	#############
	# The root directory of all the datasets
	dataset_root_dir = 'harana/datasets'

	# Generate a customized pytorch Dataset with the specified dataset name
	dataset_bpsfh = BPSFHDataset(dataset_root_dir, sample_size, frame_type, tpqn, fpqn)

	# The proportion of the dataset that is used to train
	train_proportion = 0.9
	train_size = round(len(dataset_bpsfh) * train_proportion)

	# The rest is used for validation
	validation_size = len(dataset_bpsfh) - train_size
	
	# Split the dataset and create corresponding DataLoaders
	train_dataset, validation_dataset = random_split(dataset_bpsfh, [train_size, validation_size])
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

	# Initialize the device
	device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

	# Initialize the model and the optimizer
	model = ModelComplete(batch_size, sample_size, segment_max_len, num_label, note_transform_type, chord_transform_type, decode_type, device)
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.01)

	# The number of total parameters
	#total_params = sum(param.numel() for param in model.parameters())

	'''
	# Test code for the forward pass
	batch = next(iter(train_loader))
	train_loss = model(batch)
	decoded_segments_train = model.semicrf.decode()
	print(decoded_segments_train[0])
	'''



	# Initialize a tensor board
	tb = SummaryWriter()

	# The first epoch is not in the eval mode
	eval_epoch = False

	# The training process consists of 100 epochs
	for epoch in range(1,101):
		print(f"\n\n\nepoch: {epoch}")

		# Run evaluation every 5 epochs
		if epoch%5 == 1:
			eval_epoch = True
		

		################
		#   Training   #
		################

		# Initialize the variables that store evaluation results
		train_loss_all = []
		train_acc_root_all = []
		train_acc_quality_all = []
		train_acc_overall_all = []
		
		# Keep track of the number of batches used in the current epoch 
		num_train_batch = 0
		for batch_idx, train_batch in enumerate(train_loader):

			# A batch is only used if it is a complete batch with the designated size
			if train_batch['pc_dist_seq'].shape[0] != batch_size:
				continue
			else:
				num_train_batch += 1
			

			# Forward pass
			print(f"\nTraining... epoch: {epoch}, batch: {batch_idx}")
			train_loss = model(train_batch)


			print(f"Training loss: {train_loss}")
			train_loss_all.append(train_loss.item())

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

					decoded_segments_train = model.decoder.decode()
					train_acc_root, train_acc_quality, train_acc_overall = ComputeEval(train_batch['chord_seq'], decoded_segments_train, batch_size, sample_size)
					print("Training accuracy")
					print(f"\troot: {train_acc_root}")
					print(f"\tquality: {train_acc_quality}")
					print(f"\toverall: {train_acc_overall}")
					
					train_acc_root_all.append(train_acc_root)
					train_acc_quality_all.append(train_acc_quality)
					train_acc_overall_all.append(train_acc_overall)


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
					
					validation_loss = model(validation_batch)
					decoded_segments_validation = model.decoder.decode()
					validation_acc_root, validation_acc_quality, validation_acc_overall = ComputeEval(validation_batch['chord_seq'], decoded_segments_validation, batch_size, sample_size)
					
					print(f"Validation loss: {validation_loss}")
					print("Training accuracy")
					print(f"root: {validation_acc_root}")
					print(f"quality: {validation_acc_quality}")
					print(f"overall: {validation_acc_overall}")
					
					validation_loss_all.append(validation_loss.item())
					validation_acc_root_all.append(validation_acc_root)
					validation_acc_quality_all.append(validation_acc_quality)
					validation_acc_overall_all.append(validation_acc_overall)
			

			####################################
			#   Report Evaluation statistics   #
			####################################
			
			# Print the result of evaluation and update the tensor board
			train_loss_epoch = sum(train_loss_all) / num_train_batch
			train_acc_root_epoch = sum(train_acc_root_all) / num_train_batch
			train_acc_quality_epoch = sum(train_acc_quality_all) / num_train_batch
			train_acc_overall_epoch = sum(train_acc_overall_all) / num_train_batch
			print(f"\nEvaluation summary of training on epoch {epoch}")
			print(f"loss: {train_loss_epoch}")
			print(f"root accuracy: {train_root_acc_epoch}")
			print(f"quality accuracy: {train_acc_quality_epoch}")
			print(f"overall accuracy: {train_acc_overall_epoch}")
			tb.add_scalar("Training Loss", train_loss_epoch, epoch)
			tb.add_scalar("Training Accuracy - Root", train_acc_root_epoch, epoch)
			tb.add_scalar("Training Accuracy - Quality", train_acc_quality_epoch, epoch)
			tb.add_scalar("Training Accuracy - Overall", train_acc_overall_epoch, epoch)

			# Print the result of evaluation and update the tensor board
			validation_loss_epoch = sum(validation_loss_all) / num_validation_batch
			validation_acc_root_epoch = sum(validation_acc_root_all) / num_validation_batch
			validation_acc_quality_epoch = sum(validation_acc_quality_all) / num_validation_batch
			validation_acc_overall_epoch = sum(validation_acc_overall_all) / num_validation_batch
			print(f"\nEvaluation summary of training on epoch {epoch}")
			print(f"loss: {validation_loss_epoch}")
			print(f"root accuracy: {validation_root_acc_epoch}")
			print(f"quality accuracy: {validation_acc_quality_epoch}")
			print(f"overall accuracy: {validation_acc_overall_epoch}")
			tb.add_scalar("Training Loss", validation_loss_epoch, epoch)
			tb.add_scalar("Training Accuracy - Root", validation_acc_root_epoch, epoch)
			tb.add_scalar("Training Accuracy - Quality", validation_acc_quality_epoch, epoch)
			tb.add_scalar("Training Accuracy - Overall", validation_acc_overall_epoch, epoch)
			eval_epoch = False

if __name__ == "__main__":
	main(sys.argv[1:])





