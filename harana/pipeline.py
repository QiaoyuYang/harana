from .models.model_assembler import Note2HarmonySoftMax, Note2HarmonySemiCRF, Note2HarmonyNADE, Note2HarmonyRuleSemiCRF
from .models.decoder import SemiCRFDecoder
from . import tools

import os
import shutil
import psutil
from abc import abstractmethod
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Pipeline:

	def __init__(self, note_encoder_type='CRNN', decoder_type='SemiCRF', harmony_type=tools.DEFAULT_HARMONY_TYPE,
					batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, max_seg_len=tools.DEFAULT_MAX_SEG_LEN, device=tools.DEFAULT_DEVICE,
					lr=tools.DEFAULT_LR, max_epoch=tools.DEFAULT_MAX_EPOCH, eval_period=tools.DEFAULT_EVAL_PERIOD, pretrain=False,
					store_model=True, save_model=True, reset_model=True, save_loc=None, pipeline_type=tools.PIPELINE_TYPE_TRAIN):

		self.note_encoder_type = note_encoder_type
		self.decoder_type = decoder_type
		self.harmony_type = harmony_type

		self.batch_size = batch_size
		self.sample_size = sample_size
		self.max_seg_len = max_seg_len
		self.device = device
		
		self.lr = lr
		self.max_epoch = max_epoch
		self.eval_period = eval_period
		self.pretrain = pretrain

		# Set the storing and saving parameters
		self.store_model = store_model
		self.save_model = save_model
		self.save_loc = tools.DEFAULT_CHECKPOINT_DIR if save_loc is None else save_loc

		if os.path.exists(self.get_model_path()) and reset_model:
			# Remove any saved model
			shutil.rmtree(self.get_model_path())

		if self.save_model:
			# Make sure the directory for saving/loading model exists
			os.makedirs(self.get_model_path(), exist_ok=True)

		if self.store_model:
			self.model = self.load()

		self.evaluator = tools.Evaluator(harmony_type, batch_size, sample_size, pipeline_type)
	
	def load(self):
		
		if self.decoder_type == "SoftMax":
			model = Note2HarmonySoftMax(self.note_encoder_type, self.harmony_type, self.batch_size, self.sample_size, self.device)
		if self.decoder_type == "NADE":
			model = Note2HarmonyNADE(self.note_encoder_type, self.harmony_type, self.batch_size, self.sample_size, self.device)
		elif self.decoder_type == "SemiCRF":
			if self.note_encoder_type == "PC":
				model = Note2HarmonyRuleSemiCRF(self.harmony_type, self.batch_size, self.sample_size, self.max_seg_len, self.device)
			elif self.note_encoder_type == "CRNN":
				model = Note2HarmonySemiCRF(self.harmony_type, self.batch_size, self.sample_size, self.max_seg_len, self.device)

		if isinstance(model.decoder, SemiCRFDecoder):
			train_gt_path = os.path.join(tools.DEFAULT_GROUND_TRUTH_DIR, 'train')
			validation_gt_path = os.path.join(tools.DEFAULT_GROUND_TRUTH_DIR, 'validation')
			train_transition_path = os.path.join(train_gt_path, f'trans.{tools.NPY_EXT}')
			validation_transition_path = os.path.join(validation_gt_path, f'trans.{tools.NPY_EXT}')
			model.decoder.semicrf.train_transitions = torch.tensor(np.load(train_transition_path, allow_pickle=True))
			model.decoder.semicrf.train_transitions = model.decoder.semicrf.train_transitions.to(self.device)
			model.decoder.semicrf.validation_transitions = torch.tensor(np.load(validation_transition_path, allow_pickle=True))
			model.decoder.semicrf.validation_transitions = model.decoder.semicrf.validation_transitions.to(self.device)

		model_path = self.get_model_path(with_filename=True)
		
		if self.save_model and os.path.exists(model_path):
			model.load_state_dict(torch.load(model_path), strict=False)

		if self.pretrain:
			pretrained_model_name = "_".join((self.harmony_type, "CRNN", "SoftMax"))
			pretrained_model_path = os.path.join(self.save_loc, pretrained_model_name)
			pretrained_model_filename = os.listdir(pretrained_model_path)[-1]
			unmatched_parameters = model.load_state_dict(torch.load(os.path.join(pretrained_model_path, pretrained_model_filename)), strict=False)

		return model

	def train(self, train_loader, validation_loader):
		
		self.model.to(self.device)
		optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=0.0001)

		# The first epoch is not in the eval mode
		eval_epoch = False

		# The training process consists of 100 epochs
		for epoch in range(1, self.max_epoch + 1):
			print(f"\n\n\nepoch: {epoch}")

			# Run evaluation regularly
			if epoch%self.eval_period == 0:
				eval_epoch = True

			if eval_epoch:
				self.evaluator.initialize()

			stage = 'Training'
			# Keep track of the number of batches used in the current epoch 
			for batch_idx, train_batch in enumerate(train_loader):
				print(f"Memory used: {psutil.Process().memory_info().rss / (1024 * 1024)}MB")

				# Forward pass
				optimizer.zero_grad()
				print(f"\nTraining... epoch: {epoch}, batch: {batch_idx}")

				train_loss = self.model.get_loss(train_batch, stage)
				print(f"Training loss: {train_loss}")

				# Backward propogation
				print("Back propagation")
				train_loss.backward()

				# Parameter update
				print("Updating weights")
				optimizer.step()

				# Run evaluation on the training dataset
				if eval_epoch:
					self.evaluator.num_batch[stage] += 1
					
					# Disable the gradient during evaluation
					with torch.no_grad():
						
						print("Evaluating on the training batch")
						decode_result_train = self.model.decode()
						
						gt_train = self.model.decoder.frame_postprocess(self.evaluator.get_gt(train_batch))
						
						self.evaluator.update_metrics(train_loss, decode_result_train, gt_train, stage)

			stage = 'Validation'
			
			# Run evaluation on the validation dataset
			if eval_epoch:
				
				for batch_idx, validation_batch in enumerate(validation_loader):
					
					self.evaluator.num_batch[stage] += 1
					
					with torch.no_grad():
						print(f"\nValidation... epoch: {epoch}, batch: {batch_idx}")
						print("Evaluating on the validation batch")

						validation_loss = self.model.get_loss(validation_batch, stage)
						print(f"Validation loss: {validation_loss}")

						decode_result_validation = self.model.decode()
						
						gt_validation = self.model.decoder.frame_postprocess(self.evaluator.get_gt(validation_batch))
						
						self.evaluator.update_metrics(validation_loss, decode_result_validation, gt_validation, stage)

				###################################
				#   Report Evaluation statistics   #
				####################################
				
				best_model = self.evaluator.summarize_epoch(epoch)
				if best_model:
					torch.save(self.model.state_dict(), self.get_model_path(with_filename=True))
				
				eval_epoch = False

	def test(self, test_loader):
		self.model.to(self.device)

		# Initialize a tensor board
		tb = SummaryWriter()
		
		self.evaluator.initialize()
		
		stage = 'Test'
		
		for batch_idx, test_batch in enumerate(test_loader):
			
			self.evaluator.num_batch[stage] += 1
			
			with torch.no_grad():
				
				print(f"\nTesting... batch: {batch_idx}")
				print("Evaluating on the Test batch")

				test_loss = self.model.get_loss(test_batch, stage)
				print(f"Validation loss: {test_loss}")

				decode_result_test = self.model.decode()

				gt_test = self.model.decoder.frame_postprocess(self.evaluator.get_gt(test_batch))
				
				self.evaluator.update_metrics(test_loss, decode_result_test, gt_test, stage)

		self.evaluator.summarize_epoch(0)
	
	def get_model_path(self, with_filename=False):
		
		if with_filename:
			# Get the path to the model directory
			path = os.path.join(self.save_loc, f'{self.model_name()}', f'checkpoint_{datetime.now().strftime("%y%m%d_%H%M%S")}.{tools.PT_EXT}')
		else:
			path = os.path.join(self.save_loc, f'{self.model_name()}')

		return path
	
	def model_name(self):
		
		if self.decoder_type == "SoftMax":
			model_name = "_".join((self.harmony_type, self.note_encoder_type, self.decoder_type))
		elif self.decoder_type == "NADE":
			model_name = "_".join((self.harmony_type, self.note_encoder_type, self.decoder_type))
		elif self.decoder_type == "SemiCRF":
			model_name = "_".join((self.harmony_type, self.note_encoder_type, self.decoder_type))
		return model_name