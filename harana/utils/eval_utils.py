# Training and evaluation of the model

# My import
from ..models import model_assembler
from . import chord


# Regular import
import torch
from torch.utils.tensorboard import SummaryWriter

label_type2categories = {
	"root_quality" : ["root", "quality", "overall"],
	"key_rn" : ["key", "rn"],
	"all" : ["key_tonic", "key_mode", "key", "pri_deg", "sec_deg", "degree", "root", "quality", "inversion", "chord", "rn"]
}

class Evaluator:

	def __init__(self, decode_type, label_type, batch_size, sample_size):
		self.decode_type = decode_type
		self.label_type = label_type
		self.categories = label_type2categories[self.label_type]
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.tb = SummaryWriter()
		self.initialize()

	def initialize(self):
		self.train_loss_all = []		
		self.validation_loss_all = []
		
		self.train_accuracy_all = {}
		self.validation_accuracy_all = {}

		for category in self.categories:
			self.train_accuracy_all[category] = []
			self.validation_accuracy_all[category] = []

	def update_train(self, train_batch, train_loss, decode_result):
		
		self.train_loss_all.append(train_loss)
		
		train_accuracy = self.ComputeAcc(train_batch, decode_result)
		
		print("Training accuracy")
		for category in self.categories:
			print(f"\t{category}: {train_accuracy[category]}")
			self.train_accuracy_all[category].append(train_accuracy[category])

	def update_validation(self, validation_batch, validation_loss, decode_result):
		
		self.validation_loss_all.append(validation_loss)

		validation_accuracy = self.ComputeAcc(validation_batch, decode_result)
		print("Training accuracy")
		for category in self.categories:
			print(f"\t{category}: {validation_accuracy[category]}")
			self.validation_accuracy_all[category].append(validation_accuracy[category])

	
	# Wrapper for computing the evaluation statistics
	def ComputeAcc(self, batch, decode_result):
		print('Computing accuracy ...')

		if self.decode_type == "semi_crf":
			# Convert each predicted segment sequence to a frame-level chord sequence
			root_pred, quality_pred = self.segments2chord_seq(decode_result)
			decode_result = {"root" : root_pred, "quality" : quality_pred}
		
		return self.get_accuracy_all(batch, decode_result)

	# Convert each predicted segment sequence to a frame-level chord sequence
	def segments2frame_seq(self, segments_pred):
		frame_idx = 0

		# The list of chord sequences of all samples in the batch
		root_frame_seq = []
		quality_frame_seq = []

		# The chord label sequence of a single sample
		root_frame_seq_sample = []
		quality_frame_seq_sample = []

		# Loop through samples in the batch
		for sample_idx in range(self.batch_size):
			
			# Loop through predicted segments in the sample
			for segment in segments_pred[sample_idx]:
				
				# Extract the label index, and boundary frames of the chord in the segment
				label_index, segment_start_frame, segment_end_frame = segment
				root_pc, quality_index = chord.pase_chord_index(label_index)

				# Append the chord label index of each frame in the segment to the chord sequence of the sample
				for i in range(segment_start_frame, segment_end_frame + 1):
					root_frame_seq_sample.append(root_pc)
					quality_frame_seq_sample.append(quality_index)
			
			# Append the complete chord sequence of the sample to the list of all chord sequences in the batch
			root_frame_seq.append(root_frame_seq_sample)
			quality_frame_seq.append(quality_frame_seq_sample)
		
		# Return the result as a tensor
		return torch.tensor(root_frame_seq), torch.tensor(quality_frame_seq)

	# Get the average accuracy across the batch
	def get_accuracy_all(self, batch, decode_result):

		# The list to store each type of accuracy
		accuracy_all = {}
		accuracy_sample = {}
		num_correct_sample = {}

		for category in self.categories:
			accuracy_all[category] = []

		# Loop through samples
		for sample_idx in range(self.batch_size):
			# The variables to count number of correct frames in the sample
			for category in self.categories:
				num_correct_sample[category] = 0

			# Loop through frames
			for frame_idx in range(self.sample_size):
				for decode_category in model_assembler.label_type2categories[self.label_type]:
					class_cur_gt = batch[decode_category][sample_idx, frame_idx]
					class_cur_pred = decode_result[decode_category][sample_idx, frame_idx]
					if class_cur_gt == class_cur_pred:
						num_correct_sample[decode_category] += 1

				root_cur_gt = batch["root"][sample_idx, frame_idx]
				quality_cur_gt = batch["quality"][sample_idx, frame_idx]		
				
				key_cur_gt = batch["key"][sample_idx, frame_idx]
				rn_cur_gt = batch["rn"][sample_idx, frame_idx]
				
				key_tonic_cur_gt = batch["key_tonic"][sample_idx, frame_idx]
				key_mode_cur_gt = batch["key_mode"][sample_idx, frame_idx]

				pri_deg_cur_gt = batch["pri_deg"][sample_idx, frame_idx]
				sec_deg_cur_gt = batch["sec_deg"][sample_idx, frame_idx]

				inversion_cur_gt = batch["inversion"][sample_idx, frame_idx]

				if self.label_type == "root_quality":
					root_cur_pred = decode_result["root"][sample_idx, frame_idx]
					quality_cur_pred = decode_result["quality"][sample_idx, frame_idx]
					if root_cur_gt == root_cur_pred and quality_cur_gt == quality_cur_pred:
						num_correct_sample["overall"] += 1

				if self.label_type == "key_rn":
					key_cur_pred = decode_result["key"][sample_idx, frame_idx]
					rn_cur_pred = decode_result["rn"][sample_idx, frame_idx]
					if key_cur_gt == key_cur_pred and rn_cur_gt == rn_cur_pred:
						num_correct_sample["overall"] += 1

				if self.label_type == "all":
					key_tonic_cur_pred = decode_result["key_tonic"][sample_idx, frame_idx]
					key_mode_cur_pred = decode_result["key_mode"][sample_idx, frame_idx]
					pri_deg_cur_pred = decode_result["pri_deg"][sample_idx, frame_idx]
					sec_deg_cur_pred = decode_result["sec_deg"][sample_idx, frame_idx]
					root_cur_pred = decode_result["root"][sample_idx, frame_idx]
					quality_cur_pred = decode_result["quality"][sample_idx, frame_idx]
					inversion_cur_pred = decode_result["inversion"][sample_idx, frame_idx]

					if key_tonic_cur_gt == key_tonic_cur_pred:
						num_correct_sample["key"] += 1
					if pri_deg_cur_gt == pri_deg_cur_pred and sec_deg_cur_gt == sec_deg_cur_pred:
						num_correct_sample["degree"] += 1
					if root_cur_gt == root_cur_pred and quality_cur_gt == quality_cur_pred and inversion_cur_gt == inversion_cur_pred:
						num_correct_sample["chord"] += 1
					if key_tonic_cur_gt == key_tonic_cur_pred and key_mode_cur_gt == key_mode_cur_pred and pri_deg_cur_gt == pri_deg_cur_pred \
								and sec_deg_cur_gt == sec_deg_cur_pred and quality_cur_gt == quality_cur_pred and inversion_cur_gt == inversion_cur_pred:
						num_correct_sample["rn"] += 1

			# Calculte the accuracy of each type
			for category in self.categories:
				accuracy_all[category].append(num_correct_sample[category]/self.sample_size)
		
		for category in self.categories:
			accuracy_all[category] = sum(accuracy_all[category])/self.batch_size

		# Return the average accuracy across the batch
		return accuracy_all

	def summarize_epoch(self, epoch, num_train_batch, num_validation_batch):

		train_loss_epoch = sum(self.train_loss_all) / num_train_batch
		validation_loss_epoch = sum(self.validation_loss_all) / num_validation_batch

		print(f"\nEvaluation summary of epoch {epoch}")
		
		print(f"Loss")
		print(f"\tTraining loss: {train_loss_epoch}")
		print(f"\tValidation loss: {validation_loss_epoch}")
		self.tb.add_scalar("Training Loss", train_loss_epoch, epoch)
		self.tb.add_scalar("Validation Loss", validation_loss_epoch, epoch)
		
		for category in self.categories:
			train_accuracy_epoch_cur = sum(self.train_accuracy_all[category]) / num_train_batch
			validation_accuracy_epoch_cur = sum(self.validation_accuracy_all[category]) / num_validation_batch
			print(f"\n{category}")
			print(f"\tTraining {category} accuracy: {train_accuracy_epoch_cur}")
			print(f"\tValidation {category} accuracy: {validation_accuracy_epoch_cur}")
			self.tb.add_scalar(f"Training accuracy - {category}", train_accuracy_epoch_cur, epoch)
			self.tb.add_scalar(f"Validation accuracy - {category}", validation_accuracy_epoch_cur, epoch)






