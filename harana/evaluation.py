# Training and evaluation of the model

# My import
from . import tools


# Regular import
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Evaluator:

	def __init__(self, decode_type, harmony_type, batch_size):
		self.decode_type = decode_type
		self.harmony_type = harmony_type
		self.harmony_components = tools.HARMONY_COMPONENTS[harmony_type]
		self.batch_size = batch_size
		self.sample_size = tools.FRAMES_PER_SAMPLE
		self.tb = SummaryWriter()
		self.initialize()

	# Initialize the variables that hold all values of a metric in an epoch
	def initialize(self):
		self.train_loss_all = list()	
		self.validation_loss_all = list()
		
		self.train_accuracy_all = dict()
		self.validation_accuracy_all = dict()

		for component in self.harmony_components:
			self.train_accuracy_all[component] = list()
			self.validation_accuracy_all[component] = list()

	# Update the variables related to training
	def update_train(self, train_batch, train_loss, decode_result):
		
		self.train_loss_all.append(train_loss)
		
		train_accuracy = self.ComputeAcc(train_batch, decode_result)
		
		print("Training accuracy")
		for component in self.harmony_components:
			print(f"\t{component}: {train_accuracy[component]}")
			self.train_accuracy_all[component].append(train_accuracy[component])

	# Update the variables related to validation
	def update_validation(self, validation_batch, validation_loss, decode_result):
		
		self.validation_loss_all.append(validation_loss)

		validation_accuracy = self.ComputeAcc(validation_batch, decode_result)
		
		print("Validation accuracy")
		for component in self.harmony_components:
			print(f"\t{component}: {validation_accuracy[component]}")
			self.validation_accuracy_all[component].append(validation_accuracy[component])

	# Compute the frame level accuracy for each component
	def ComputeAcc(self, batch, decode_result):

		# Extract the gt component indexes
		if self.harmony_type == "CHORD":
			harmony_component_gt = batch[tools.KEY_CHORD_COMPONENT_GT]
		elif self.harmony_type == "RN":
			harmony_component_gt = batch[tools.KEY_RN_COMPONENT_GT]

		acc = dict()
		if self.decode_type == 'semi_crf':
			# The output of semi-crf decoder is a set of segments
			# Need to divide them into frames first
			decode_result_temp = np.empty_like(harmony_component_gt)
			for sample_index, segments_sample in enumerate(decode_result):
				for i, [harmony_index, start_frame, end_frame] in enumerate(segments_sample):
					harmony_component_indexes = tools.Harmony.parse_harmony_index(harmony_index, self.harmony_type)
					for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
						decode_result_temp[sample_index, i, start_frame : end_frame + 1] = harmony_component_indexes[i]
			decode_result = decode_result_temp

			# Then compute the accuracy
			for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
				sample_acc = (np.equal(harmony_component_gt[:, i, :], decode_result[:, i, :])).sum(dim=1) / tools.FRAMES_PER_SAMPLE
				acc[component] = sample_acc.sum() / self.batch_size
		
		elif self.decode_type == 'softmax':
			for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
				# Compute the accuracy of each sample
				sample_acc = (harmony_component_gt[:, i, :] == decode_result[component]).sum(dim=1) / tools.FRAMES_PER_SAMPLE
				# Then average across samples in the batch
				acc[component] = sample_acc.sum() / self.batch_size

		return acc

	# Sumarize the metrics in an epoch and write to tensorboard
	def summarize_epoch(self, epoch, num_train_batch, num_validation_batch):

		train_loss_epoch = sum(self.train_loss_all) / num_train_batch
		validation_loss_epoch = sum(self.validation_loss_all) / num_validation_batch

		print(f"\nEvaluation summary of epoch {epoch}")
		
		print(f"Loss")
		print(f"\tTraining loss: {train_loss_epoch}")
		print(f"\tValidation loss: {validation_loss_epoch}")
		self.tb.add_scalar("Training Loss", train_loss_epoch, epoch)
		self.tb.add_scalar("Validation Loss", validation_loss_epoch, epoch)
		
		for component in self.harmony_components:
			train_accuracy_epoch_cur = sum(self.train_accuracy_all[component]) / num_train_batch
			validation_accuracy_epoch_cur = sum(self.validation_accuracy_all[component]) / num_validation_batch
			print(f"\n{component}")
			print(f"\tTraining {component} accuracy: {train_accuracy_epoch_cur}")
			print(f"\tValidation {component} accuracy: {validation_accuracy_epoch_cur}")
			self.tb.add_scalar(f"Training accuracy - {component}", train_accuracy_epoch_cur, epoch)
			self.tb.add_scalar(f"Validation accuracy - {component}", validation_accuracy_epoch_cur, epoch)






