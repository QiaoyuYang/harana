# Training and evaluation of the model

# My import
from .. import tools


# Regular import
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Evaluator:

	def __init__(self, harmony_type, batch_size, sample_size, pipeline_type):
		self.harmony_type = harmony_type
		self.harmony_components = tools.HARMONY_COMPONENTS[harmony_type]
		self.batch_size = batch_size
		self.harmony_sample_size = round(sample_size / tools.HARMONY_UNIT_SPAN)
		self.pipeline_type = pipeline_type
		self.tb = SummaryWriter()
		self.initialize_best()
	
	# Initialize the variables that hold all values of a metric in an epoch
	def initialize(self):
		
		self.num_batch = dict()
		self.loss_all = dict()
		
		self.component_acc_all = dict()
		self.majmin_acc_all = dict()
		self.overall_acc_all = dict()
		
		self.under_seg_all = dict()
		self.over_seg_all = dict()
		self.sq_all = dict()

		for stage in tools.STAGES[self.pipeline_type]:
			self.num_batch[stage] = 0
			self.loss_all[stage] = list()	
			
			self.component_acc_all[stage] = dict()
			for component in self.harmony_components:
				self.component_acc_all[stage][component] = list()
			
			self.overall_acc_all[stage] = list()
			self.majmin_acc_all[stage] = list()
			
			self.under_seg_all[stage] = list()
			self.over_seg_all[stage] = list()
			self.sq_all[stage] = list()

	def initialize_best(self):
		self.lowest_val_loss = pow(10, 5)
		
		self.best_component_val_acc = dict()
		for component in self.harmony_components:
			self.best_component_val_acc[component] = 0
		self.best_majmin_val_acc = 0
		self.best_overall_val_acc = 0
		
		self.best_val_under_seg = 0
		self.best_val_over_seg = 0
		self.best_val_sq = 0


	def get_gt(self, batch):

		frame_gt = batch[tools.KEY_HARMONY_COMPONENT_GT]

		return frame_gt
	
	def update_metrics(self, loss_batch, decode_result, gt, stage):
		
		frame_gt, interval_gt, _ = gt
		frame_decode, interval_decode, _ = decode_result
		'''
		for i in range(self.batch_size):
			print(f"\nSample {i}")
			print(f"ground truth interval: {interval_gt[i]}")
			print(f"decoded interval: {interval_decode[i]}")
		'''
		
		self.batch_size = frame_gt.shape[0]
		
		self.loss_all[stage].append(loss_batch)
		
		component_acc_batch, overall_acc_batch, majmin_acc_batch = self.compute_component_acc(frame_gt, frame_decode)
		print(f"{stage} accuracy")
		for component in self.harmony_components:
			print(f"\t{component}: {component_acc_batch[component]}")
			self.component_acc_all[stage][component].append(component_acc_batch[component])

		print(f"\tmajmin: {majmin_acc_batch}")
		self.majmin_acc_all[stage].append(majmin_acc_batch)
		print(f"\toverall: {overall_acc_batch}")
		self.overall_acc_all[stage].append(overall_acc_batch)

		
		under_seg_batch, over_seg_batch, sq_batch = self.compute_sq(interval_gt, interval_decode)
		print(f"{stage} segmentation quality")
		print(f"\tunder Segmentation: {under_seg_batch}")
		self.under_seg_all[stage].append(under_seg_batch)
		print(f"\tover Segmentation: {over_seg_batch}")
		self.over_seg_all[stage].append(over_seg_batch)
		print(f"\toverall: {sq_batch}")
		self.sq_all[stage].append(sq_batch)

	# Compute the frame level accuracy for each component
	def compute_component_acc(self, frame_gt, frame_decode):

		component_acc = dict()
		correct_frames = torch.zeros(self.batch_size, self.harmony_sample_size, len(self.harmony_components))
		for i, component in enumerate(self.harmony_components):
			correct_frames[..., i] = (frame_gt[:, i, :] == frame_decode[:, i, :])
			# Compute the accuracy of each sample
			sample_component_acc = (correct_frames[..., i]).sum(dim=1) / self.harmony_sample_size
			# Then average across samples in the batch
			component_acc[component] = sample_component_acc.sum() / self.batch_size

		correct_majmin_frames = (np.vectorize(tools.QUALITY2MAJMIN.get)(frame_gt[:, 1, :].cpu()) == np.vectorize(tools.QUALITY2MAJMIN.get)(frame_decode[:, 1, :].cpu()))
		sample_majmin_acc = (correct_majmin_frames).sum(axis=1) / self.harmony_sample_size
		majmin_acc = sample_majmin_acc.sum() / self.batch_size
		
		sample_overall_acc = correct_frames.prod(dim=-1).sum(dim=1) / self.harmony_sample_size
		overall_acc = sample_overall_acc.sum() / self.batch_size
		
		return component_acc, overall_acc, majmin_acc

	# Adapted from https://github.com/craffel/mir_eval/blob/master/mir_eval/chord.py
	def directional_hamming_distance(self, ref_intervals, est_intervals):
		
		est_boundaries = torch.unique(est_intervals.flatten())
		duration_mismatch = 0
		for onset_cur, offset_cur in ref_intervals:
			duration_cur = offset_cur + 1 - onset_cur
			est_boundaries_within_interval_cur = est_boundaries[(est_boundaries >= onset_cur) & (est_boundaries < offset_cur + 1)]
			candidate_matching_boundaries = torch.hstack([onset_cur, est_boundaries_within_interval_cur, offset_cur + 1])
			duration_mismatch += duration_cur - torch.diff(candidate_matching_boundaries).max()

		return duration_mismatch / (ref_intervals[-1, 1] + 1 - ref_intervals[0, 0])

	def compute_under_seg(self, ref_intervals, est_intervals):

		return 1 - self.directional_hamming_distance(est_intervals, ref_intervals)

	def compute_over_seg(self, ref_intervals, est_intervals):

		return 1 - self.directional_hamming_distance(ref_intervals, est_intervals)

	def compute_sq(self, interval_gt, interval_decode):

		under_seg = 0
		over_seg = 0
		sq = 0
		for sample_index in range(self.batch_size):
			ref_intervals = interval_gt[sample_index]
			est_intervals = interval_decode[sample_index]
			under_seg_sample = self.compute_under_seg(ref_intervals, est_intervals)
			over_seg_sample = self.compute_over_seg(ref_intervals, est_intervals)
			under_seg += under_seg_sample
			over_seg += over_seg_sample
			sq += min(under_seg_sample, over_seg_sample)
		
		under_seg /= self.batch_size
		over_seg /= self.batch_size
		sq /= self.batch_size

		return under_seg, over_seg, sq


	# Sumarize the metrics in an epoch and write to tensorboard
	def summarize_epoch(self, epoch):
		print(f"\nEvaluation summary of epoch {epoch}")
		for stage in tools.STAGES[self.pipeline_type]:
			loss_epoch = sum(self.loss_all[stage]) / self.num_batch[stage]

			if stage != "Training":
				if stage == "Test" or loss_epoch < self.lowest_val_loss:
					best_model = True
				else:
					best_model = False

			print(stage)
			print(f"\t{stage} loss: {loss_epoch}")
			if self.pipeline_type is tools.PIPELINE_TYPE_TRAIN:
				self.tb.add_scalar(f"{stage} Loss", loss_epoch, epoch)
			if stage != "Training" and best_model:
				self.lowest_val_loss = loss_epoch
			
			print(f"\t{stage} Accuracy")
			for component in self.harmony_components:
				component_acc_epoch = sum(self.component_acc_all[stage][component]) / self.num_batch[stage]
				print(f"\t\t{component}: {component_acc_epoch}")
				if self.pipeline_type is tools.PIPELINE_TYPE_TRAIN:
					self.tb.add_scalar(f"{stage} accuracy - {component}", component_acc_epoch, epoch)

				if stage != "Training" and best_model:
					self.best_component_val_acc[component] = component_acc_epoch
			
			majmin_acc_epoch_cur = sum(self.majmin_acc_all[stage]) / self.num_batch[stage]
			overall_acc_epoch_cur = sum(self.overall_acc_all[stage]) / self.num_batch[stage]
			print(f"\t\toverall: {overall_acc_epoch_cur}")
			if self.pipeline_type is tools.PIPELINE_TYPE_TRAIN:
				self.tb.add_scalar(f"{stage} accuracy - majmin", majmin_acc_epoch_cur, epoch)
				self.tb.add_scalar(f"{stage} accuracy - overall", overall_acc_epoch_cur, epoch)
			if stage != "Training" and best_model:
				self.best_majmin_val_acc = majmin_acc_epoch_cur
				self.best_overall_val_acc = overall_acc_epoch_cur

			print(f"\t{stage} Segmentation Quality")
			under_seg_epoch = sum(self.under_seg_all[stage]) / self.num_batch[stage]
			over_seg_epoch = sum(self.over_seg_all[stage]) / self.num_batch[stage]
			sq_epoch = sum(self.sq_all[stage]) / self.num_batch[stage]
			print(f"\t\tunder segmentation: {under_seg_epoch}")
			print(f"\t\tover segmentation: {over_seg_epoch}")
			print(f"\t\toverall: {sq_epoch}")
			if self.pipeline_type is tools.PIPELINE_TYPE_TRAIN:
				self.tb.add_scalar(f"{stage} sq - under segmentation", under_seg_epoch, epoch)
				self.tb.add_scalar(f"{stage} sq - over segmentation", over_seg_epoch, epoch)	
				self.tb.add_scalar(f"{stage} sq - overall", sq_epoch, epoch)	
			if stage != "Training" and best_model:
				self.best_val_under_seg = under_seg_epoch
				self.best_val_over_seg = over_seg_epoch
				self.best_val_sq = sq_epoch

		

		print(f'\nlowest validation loss: {self.lowest_val_loss}')
		
		print(f'best validation accuracy')
		for component in self.harmony_components:
			print(f"\t{component}: {self.best_component_val_acc[component]}")

		print(f"\tmajmin: {self.best_majmin_val_acc}")
		print(f"\toverall: {self.best_overall_val_acc}")
		
		print(f'best segmentation quality')
		print(f"\tunder segmentation: {self.best_val_under_seg}")
		print(f"\tover segmentation: {self.best_val_over_seg}")
		print(f"\toverall: {self.best_val_sq}")

		return best_model






