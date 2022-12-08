# Training and evaluation of the model

# My import
from ..models.model_assembler import ModelComplete
from . import chord


# Regular import
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
torch.manual_seed(0)


class Evaluator:

	def __init__(self, decode_type, label_type):
		self.decode_type = decode_type
		self.label_type = label_type
		self.tb = SummaryWriter()
		self.initialize()

	def initialize(self):
		self.train_loss_all = []
		self.train_acc_overall_all = []
		
		self.validation_loss_all = []
		self.validation_acc_overall_all = []
		
		if self.label_type == "root_quality":
			self.train_acc_root_all = []
			self.train_acc_quality_all = []

			self.validation_acc_root_all = []
			self.validation_acc_quality_all = []
		
		if self.label_type == "key_rn":
			self.train_acc_key_all = []
			self.train_acc_rn_all = []

			self.validation_acc_key_all = []
			self.validation_acc_rn_all = []

	# Wrapper for computing the evaluation statistics
	def ComputeEvalSegment(self, chord_seq_gt, decoded_segments, label_type, batch_size, sample_size):
		print('Computing accuracy ...')

		if label_type == "root_quality":
			# Convert each predicted segment sequence to a frame-level chord sequence
			chords_seq_pred = segments2chord_seq(decoded_segments, batch_size, sample_size)

			# Get the average accuracy across the batch
			root_acc, quality_acc, overall_acc = get_accuracy_chord(chord_seq_gt, chords_seq_pred, batch_size, sample_size)

			# And return them
			return root_acc, quality_acc, overall_acc

	def update_train(self, train_batch, train_loss, decode_result, decode_type, label_type, batch_size, sample_size):
		
		self.train_loss_all.append(train_loss)
		
		if label_type == "root_quality":
			train_acc_root, train_acc_quality, train_acc_overall = self.ComputeAcc(train_batch, decode_result, decode_type, label_type, batch_size, sample_size)
			
			print("Training accuracy")
			print(f"\troot: {train_acc_root}")
			print(f"\tquality: {train_acc_quality}")
			print(f"\toverall: {train_acc_overall}")

			self.train_acc_root_all.append(train_acc_root)
			self.train_acc_quality_all.append(train_acc_quality)
			self.train_acc_overall_all.append(train_acc_overall)
		
		if label_type == "key_rn":
			train_acc_key, train_acc_rn, train_acc_overall = self.ComputeAcc(train_batch, decode_result, decode_type, label_type, batch_size, sample_size)
			
			print("Training accuracy")
			print(f"\tkey: {train_acc_key}")
			print(f"\trn: {train_acc_rn}")
			print(f"\toverall: {train_acc_overall}")

			self.train_acc_key_all.append(train_acc_key)
			self.train_acc_rn_all.append(train_acc_rn)
			self.train_acc_overall_all.append(train_acc_overall)

	def update_validation(self, validation_batch, validation_loss, decode_result, decode_type, label_type, batch_size, sample_size):
		
		self.validation_loss_all.append(validation_loss)
		
		if label_type == "root_quality":
			validation_acc_root, validation_acc_quality, validation_acc_overall = self.ComputeAcc(validation_batch, decode_result, decode_type, label_type, batch_size, sample_size)
			
			print("Validation accuracy")
			print(f"\troot: {validation_acc_root}")
			print(f"\tquality: {validation_acc_quality}")
			print(f"\toverall: {validation_acc_overall}")

			self.validation_acc_root_all.append(validation_acc_root)
			self.validation_acc_quality_all.append(validation_acc_quality)
			self.validation_acc_overall_all.append(validation_acc_overall)
		
		if label_type == "key_rn":
			validation_acc_key, validation_acc_rn, validation_acc_overall = self.ComputeAcc(validation_batch, decode_result, decode_type, label_type, batch_size, sample_size)
			
			print("Validation accuracy")
			print(f"\tkey: {validation_acc_key}")
			print(f"\trn: {validation_acc_rn}")
			print(f"\toverall: {validation_acc_overall}")

			self.validation_acc_key_all.append(validation_acc_key)
			self.validation_acc_rn_all.append(validation_acc_rn)
			self.validation_acc_overall_all.append(validation_acc_overall)
	
	# Wrapper for computing the evaluation statistics
	def ComputeAcc(self, batch, decode_result, decode_type, label_type, batch_size, sample_size):
		print('Computing accuracy ...')

		if label_type == "root_quality":
			root_seq_gt = batch["root_seq"]
			quality_seq_gt = batch["quality_seq"]
			if decode_type == "softmax":
				root_seq_pred, quality_seq_pred = decode_result
				root_acc, quality_acc, overall_acc = get_accuracy_root_quality(root_seq_gt, quality_seq_gt, root_seq_pred, quality_seq_pred, batch_size, sample_size)

			if decode_type == "semi_crf":
				# Convert each predicted segment sequence to a frame-level chord sequence
				chord_seq_pred = segments2chord_seq(decode_result, batch_size, sample_size)

				# Get the average accuracy across the batch
				root_acc, quality_acc, overall_acc = get_accuracy_chord(root_seq_gt, quality_seq_gt, chord_seq_pred, batch_size, sample_size)

			# And return them
			return root_acc, quality_acc, overall_acc

		if label_type == "key_rn":
			if decode_type == "softmax":
				key_seq_gt = batch["key_seq"]
				rn_seq_gt = batch["rn_seq"]
				key_seq_pred, rn_seq_pred = decode_result
				key_acc, rn_acc, overall_acc = get_accuracy_key_rn(key_seq_gt, rn_seq_gt, key_seq_pred, rn_seq_pred, batch_size, sample_size)

			return key_acc, rn_acc, overall_acc

	def summarize_epoch(self, epoch, num_train_batch, num_validation_batch):

		if self.label_type == "root_quality":
			# Print the result of evaluation and update the tensor board
			train_loss_epoch = sum(self.train_loss_all) / num_train_batch
			train_acc_root_epoch = sum(self.train_acc_root_all) / num_train_batch
			train_acc_quality_epoch = sum(self.train_acc_quality_all) / num_train_batch
			train_acc_overall_epoch = sum(self.train_acc_overall_all) / num_train_batch
			print(f"\nEvaluation summary of training on epoch {epoch}")
			print(f"loss: {train_loss_epoch}")
			print(f"root accuracy: {train_acc_root_epoch}")
			print(f"quality accuracy: {train_acc_quality_epoch}")
			print(f"overall accuracy: {train_acc_overall_epoch}")
			self.tb.add_scalar("Training Loss", train_loss_epoch, epoch)
			self.tb.add_scalar("Training Accuracy - Root", train_acc_root_epoch, epoch)
			self.tb.add_scalar("Training Accuracy - Quality", train_acc_quality_epoch, epoch)
			self.tb.add_scalar("Training Accuracy - Overall", train_acc_overall_epoch, epoch)

			# Print the result of evaluation and update the tensor board
			validation_loss_epoch = sum(self.validation_loss_all) / num_validation_batch
			validation_acc_root_epoch = sum(self.validation_acc_root_all) / num_validation_batch
			validation_acc_quality_epoch = sum(self.validation_acc_quality_all) / num_validation_batch
			validation_acc_overall_epoch = sum(self.validation_acc_overall_all) / num_validation_batch
			print(f"\nEvaluation summary of validation on epoch {epoch}")
			print(f"loss: {validation_loss_epoch}")
			print(f"root accuracy: {validation_acc_root_epoch}")
			print(f"quality accuracy: {validation_acc_quality_epoch}")
			print(f"overall accuracy: {validation_acc_overall_epoch}")
			self.tb.add_scalar("Validation Loss", validation_loss_epoch, epoch)
			self.tb.add_scalar("Validation Accuracy - Root", validation_acc_root_epoch, epoch)
			self.tb.add_scalar("Validation Accuracy - Quality", validation_acc_quality_epoch, epoch)
			self.tb.add_scalar("Validation Accuracy - Overall", validation_acc_overall_epoch, epoch)
		
		if self.label_type == "key_rn":
			# Print the result of evaluation and update the tensor board
			train_loss_epoch = sum(self.train_loss_all) / num_train_batch
			train_acc_key_epoch = sum(self.train_acc_key_all) / num_train_batch
			train_acc_rn_epoch = sum(self.train_acc_rn_all) / num_train_batch
			train_acc_overall_epoch = sum(self.train_acc_overall_all) / num_train_batch
			print(f"\nEvaluation summary of training on epoch {epoch}")
			print(f"loss: {train_loss_epoch}")
			print(f"key accuracy: {train_acc_key_epoch}")
			print(f"rn accuracy: {train_acc_rn_epoch}")
			print(f"overall accuracy: {train_acc_overall_epoch}")
			self.tb.add_scalar("Training Loss", train_loss_epoch, epoch)
			self.tb.add_scalar("Training Accuracy - Key", train_acc_key_epoch, epoch)
			self.tb.add_scalar("Training Accuracy - Roman Numeral", train_acc_rn_epoch, epoch)
			self.tb.add_scalar("Training Accuracy - Overall", train_acc_overall_epoch, epoch)

			# Print the result of evaluation and update the tensor board
			validation_loss_epoch = sum(self.validation_loss_all) / num_validation_batch
			validation_acc_key_epoch = sum(self.validation_acc_key_all) / num_validation_batch
			validation_acc_rn_epoch = sum(self.validation_acc_rn_all) / num_validation_batch
			validation_acc_overall_epoch = sum(self.validation_acc_overall_all) / num_validation_batch
			print(f"\nEvaluation summary of validation on epoch {epoch}")
			print(f"loss: {validation_loss_epoch}")
			print(f"key accuracy: {validation_acc_key_epoch}")
			print(f"rn accuracy: {validation_acc_rn_epoch}")
			print(f"overall accuracy: {validation_acc_overall_epoch}")
			self.tb.add_scalar("Validation Loss", validation_loss_epoch, epoch)
			self.tb.add_scalar("Validation Accuracy - Key", validation_acc_key_epoch, epoch)
			self.tb.add_scalar("Validation Accuracy - Roman Numeral", validation_acc_rn_epoch, epoch)
			self.tb.add_scalar("Validation Accuracy - Overall", validation_acc_overall_epoch, epoch)



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

def get_accuracy_chord(root_seq_gt, quality_seq_gt, chord_seq_pred, batch_size, sample_size):
		
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

			root_cur_gt = root_seq_gt[sample_idx, frame_idx]
			quality_cur_gt = quality_seq_gt[sample_idx, frame_idx]

			# Get the root, quality and bass of the current predicted chord
			chord_cur_pred = chord_seq_pred[sample_idx, frame_idx]
			root_cur_pred, quality_cur_pred = chord.parse_chord_index(chord_cur_pred)

			# Check if the predicted chord is correct
			if root_cur_gt == root_cur_pred:
				root_correct += 1
			if quality_cur_gt == quality_cur_pred:
				quality_correct += 1
			if root_cur_gt == root_cur_pred and quality_cur_gt == quality_cur_pred:
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
	
# Get the average accuracy across the batch based on the root_quality label type
def get_accuracy_root_quality(root_seq_gt, quality_seq_gt, root_seq_pred, quality_seq_pred, batch_size, sample_size):

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

			root_cur_gt = root_seq_gt[sample_idx, frame_idx]
			quality_cur_pred = quality_seq_pred[sample_idx, frame_idx]

			root_cur_gt = root_seq_gt[sample_idx, frame_idx]
			quality_cur_pred = quality_seq_pred[sample_idx, frame_idx]

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

# Get the average accuracy across the batch based on the root_quality label type
def get_accuracy_root_quality(root_seq_gt, quality_seq_gt, root_seq_pred, quality_seq_pred, batch_size, sample_size):

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

			root_cur_gt = root_seq_gt[sample_idx, frame_idx]
			quality_cur_gt = quality_seq_gt[sample_idx, frame_idx]

			root_cur_pred = root_seq_pred[sample_idx, frame_idx]
			quality_cur_pred = quality_seq_pred[sample_idx, frame_idx]

			# Check if the predicted chord is correct
			if root_cur_gt == root_cur_pred:
				root_correct += 1
			if quality_cur_gt == quality_cur_pred:
				quality_correct += 1
			if root_cur_gt == root_cur_pred and quality_cur_gt == quality_cur_pred:
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


# Get the average accuracy across the batch based on the key_rn label type
def get_accuracy_key_rn(key_seq_gt, rn_seq_gt, key_seq_pred, rn_seq_pred, batch_size, sample_size):

	# The list to store each type of accuracy
	key_acc_all = []
	rn_acc_all = []
	overall_acc_all = []
		
	# Loop through samples
	for sample_idx in range(batch_size):

		# The variables to count number of correct frames in the sample
		key_correct = 0
		rn_correct = 0
		overall_correct = 0

		# Loop through frames
		for frame_idx in range(sample_size):

			key_cur_gt = key_seq_gt[sample_idx, frame_idx]
			rn_cur_gt = rn_seq_gt[sample_idx, frame_idx]

			key_cur_pred = key_seq_pred[sample_idx, frame_idx]
			rn_cur_pred = rn_seq_pred[sample_idx, frame_idx]

			# Check if the predicted chord is correct
			if key_cur_gt == key_cur_pred:
				key_correct += 1
			if rn_cur_gt == rn_cur_pred:
				rn_correct += 1
			if key_cur_gt == key_cur_pred and rn_cur_gt == rn_cur_pred:
				overall_correct += 1

		# Calculte the accuracy of each type
		key_acc_sample = key_correct / sample_size
		rn_acc_sample = rn_correct / sample_size
		overall_acc_sample = overall_correct / sample_size

		# Append them to the list of accuracies of all samples in the batch
		key_acc_all.append(key_acc_sample)
		rn_acc_all.append(rn_acc_sample)
		overall_acc_all.append(overall_acc_sample)
		
	# Return the average accuracy across the batch
	return sum(key_acc_all) / batch_size, sum(rn_acc_all) / batch_size, sum(overall_acc_all) / batch_size