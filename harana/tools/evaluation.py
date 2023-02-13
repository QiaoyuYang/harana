# Training and evaluation of the model

# My import
from .. import tools


# Regular import
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Evaluator:

	def __init__(self, harmony_type, batch_size, sample_size):
		self.harmony_type = harmony_type
		self.harmony_components = tools.HARMONY_COMPONENTS[harmony_type]
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.tb = SummaryWriter()
		self.initialize()
		self.initialize_best()
	
	# Initialize the variables that hold all values of a metric in an epoch
	def initialize(self):
		self.num_batch = dict()
		self.loss_all = dict()
		self.acc_all = dict()
		self.sq_all = dict()
		for stage in tools.STAGES:
			self.num_batch[stage] = 0
			self.loss_all[stage] = list()	
			self.acc_all[stage] = dict()
			self.sq_all[stage] = dict()
			for component in self.harmony_components:
				self.acc_all[stage][component] = list()
				self.sq_all[stage][component] = list()

	def initialize_best(self):
		self.lowest_val_loss = pow(10, 5)
		self.best_val_acc = dict()
		for component in self.harmony_components:
			self.best_val_acc[component] = 0

	def get_gt(self, batch):

		frame_gt = batch[tools.KEY_HARMONY_COMPONENT_GT]

		return frame_gt
	
	def update_metrics(self, loss_batch, frame_gt, decode_result, stage, acc=True, seg=False):

		frame_decode, segment_decode = decode_result
		
		self.loss_all[stage].append(loss_batch)
		
		if acc:
			acc_batch = self.compute_acc(frame_gt, frame_decode)
			print(f"{stage} accuracy")
			for component in self.harmony_components:
				print(f"\t{component}: {acc_batch[component]}")
				self.acc_all[stage][component].append(acc_batch[component])
		if seg:
			sq_batch = self.compute_sq(segment_gt, segment_decode)
			print(f"{stage} SQ")
			for component in self.harmony_components:
				print(f"\t{component}: {sq_batch[component]}")
				self.sq_all[stage][component].append(seq_batch[component])

	# Compute the frame level accuracy for each component
	def compute_acc(self, frame_gt, frame_decode):

		acc = dict()
		for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
			# Compute the accuracy of each sample
			sample_acc = (frame_gt[:, i, :] == frame_decode[component]).sum(dim=1) / frame_gt.shape[-1]
			# Then average across samples in the batch
			acc[component] = sample_acc.sum() / self.batch_size

		return acc

	def compute_sq(self, batch, decode_result):
		pass

	# Sumarize the metrics in an epoch and write to tensorboard
	def summarize_epoch(self, epoch):
		print(f"\nEvaluation summary of epoch {epoch}")
		for stage in tools.STAGES:
			loss_epoch = sum(self.loss_all[stage]) / self.num_batch[stage]

			if stage == 'Validation' and loss_epoch < self.lowest_val_loss:
				self.lowest_val_loss = loss_epoch
				best_model = True
			else:
				best_model = False

			print(stage)
			print(f"\t{stage} loss: {loss_epoch}")
			self.tb.add_scalar(f"{stage} Loss", loss_epoch, epoch)
			print(f"\t{stage} Accuracy")
			for component in self.harmony_components:
				acc_epoch_cur = sum(self.acc_all[stage][component]) / self.num_batch[stage]
				print(f"\t{component}: {acc_epoch_cur}")
				self.tb.add_scalar(f"{stage} accuracy - {component}", acc_epoch_cur, epoch)

				if best_model:
					self.best_val_acc[component] = acc_epoch_cur

		print(f'\nlowest validation loss: {self.lowest_val_loss}')
		print(f'best validation accuracy')
		for component in self.harmony_components:
			print(f"\t{component}: {self.best_val_acc[component]}")

		return best_model






