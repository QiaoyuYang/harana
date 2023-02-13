from .models.model_assembler import Note2HarmonySoftmax, Harmony2HarmonySoftmax, Note2HarmonySemiCRF
from . import tools

import os
import shutil
import psutil
from abc import abstractmethod
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class TrainingPipeline:

	def __init__(self, batch_size, sample_size, harmony_type, device, lr, max_epoch, eval_period, store_model, save_model, reset_model, save_loc):

		self.batch_size = batch_size
		self.sample_size = sample_size
		self.harmony_type = harmony_type
		self.device = device
		self.lr = lr
		self.max_epoch = max_epoch
		self.eval_period = eval_period

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

		self.evaluator = tools.Evaluator(harmony_type, batch_size, sample_size)

	def train(self, train_loader, validation_loader):

		self.model.to(self.device)
		optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

		# Initialize a tensor board
		tb = SummaryWriter()

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

				train_loss = self.model.get_loss(train_batch)
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
						
						if isinstance(self.model, Harmony2HarmonySoftmax):
							frame_gt = self.model.harmony_encoder.harmony_vector
						else:
							frame_gt = self.evaluator.get_gt(train_batch)
					frame_gt = frame_gt.to(self.device)
					self.evaluator.update_metrics(train_loss, frame_gt, decode_result_train, stage)

			stage = 'Validation'
			# Run evaluation on the validation dataset
			if eval_epoch:
				for batch_idx, validation_batch in enumerate(validation_loader):
					self.evaluator.num_batch[stage] += 1
					with torch.no_grad():
						print(f"\nValidation... epoch: {epoch}, batch: {batch_idx}")
						print("Evaluating on the validation batch")

						validation_loss = self.model.get_loss(validation_batch)
						print(f"Validation loss: {validation_loss}")

						decode_result_validation = self.model.decode()
						if isinstance(self.model, Harmony2HarmonySoftmax):
							frame_gt = self.model.harmony_encoder.harmony_vector
						else:
							frame_gt = self.evaluator.get_gt(validation_batch)
				frame_gt = frame_gt.to(self.device)
				self.evaluator.update_metrics(validation_loss, frame_gt, decode_result_validation, stage)

				###################################
				#   Report Evaluation statistics   #
				####################################
				best_model = self.evaluator.summarize_epoch(epoch)
				if best_model:
					torch.save(self.model.state_dict(), self.get_model_path(with_filename=True))
				eval_epoch = False

class Note2HarmonySoftmaxTrainer(TrainingPipeline):
	
	def __init__(self, batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, 
					harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE, lr=tools.DEFAULT_LR, 
					max_epoch=tools.DEFAULT_MAX_EPOCH, eval_period=tools.DEFAULT_EVAL_PERIOD,
					store_model=True, save_model=True, reset_model=True, save_loc=None):

		super().__init__(batch_size, sample_size, harmony_type, device, lr, max_epoch, 
							eval_period, store_model, save_model, reset_model, save_loc)

	def load(self):

		model = Note2HarmonySoftmax(batch_size=self.batch_size, sample_size=self.sample_size, harmony_type=self.harmony_type, device=self.device)
		
		model_path = self.get_model_path(with_filename=True)
		
		if self.save_model and os.path.exists(self.get_model_path(with_filename=True)):
			model.load_state_dict(torch.load(model_path), strict=False)

		return model

	def run(self, train_loader, validation_loader):

		self.train(training_loader, validation_loader)

	def get_model_path(self, with_filename=False):
		# Get the path to the model directory
		path = os.path.join(self.save_loc, f'{self.model_name()}')
		
		if with_filename:
			path = os.path.join(self.save_loc, f'{self.model_name()}', f'checkpoint.{tools.PT_EXT}')
		return path
	
	@classmethod
	def model_name(cls):

		tag = cls.__name__.rstrip('Trainer')

		return tag

class Harmony2HarmonySoftmaxTrainer(TrainingPipeline):
	def __init__(self, batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, 
					harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE, lr=tools.DEFAULT_LR, 
					max_epoch=tools.DEFAULT_MAX_EPOCH, eval_period=tools.DEFAULT_EVAL_PERIOD,
					store_model=True, save_model=True, reset_model=True, save_loc=None):

		super().__init__(batch_size, sample_size, harmony_type, device, lr, max_epoch, 
							eval_period, store_model, save_model, reset_model, save_loc)

	def load(self):
		
		model = Harmony2HarmonySoftmax(batch_size=self.batch_size, sample_size=self.sample_size, harmony_type=self.harmony_type, device=self.device)

		model_path = self.get_model_path(with_filename=True)

		if self.save_model and os.path.exists(self.get_model_path(with_filename=True)):
			model.load_state_dict(torch.load(model_path), strict=False)

		return model

	def run(self, train_loader, validation_loader):

		self.train(training_loader, validation_loader)

	def get_model_path(self, with_filename=False):
		# Get the path to the model directory
		path = os.path.join(self.save_loc, f'{self.model_name()}')

		if with_filename:
			path = os.path.join(self.save_loc, f'{self.model_name()}', f'checkpoint.{tools.PT_EXT}')


		return path
	
	@classmethod
	def model_name(cls):

		tag = cls.__name__.rstrip('Trainer')

		return tag

class Note2HarmonySemiCRFTrainer(TrainingPipeline):
	def __init__(self, batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, 
					harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE, lr=tools.DEFAULT_LR, 
					max_epoch=tools.DEFAULT_MAX_EPOCH, eval_period=tools.DEFAULT_EVAL_PERIOD,
					store_model=True, save_model=True, reset_model=True, pretrain=True, save_loc=None):
		
		self.pretrain = pretrain
		
		super().__init__(batch_size, sample_size, harmony_type, device, lr, max_epoch, 
							eval_period, store_model, save_model, reset_model, save_loc)

	def load(self):
		
		model = Note2HarmonySemiCRF(batch_size=self.batch_size, sample_size=self.sample_size, harmony_type=self.harmony_type, device=self.device)

		model_path = self.get_model_path(with_filename=True)
		
		if self.save_model and os.path.exists(self.get_model_path(with_filename=True)):
			model.load_state_dict(torch.load(model_path), strict=False)

		if self.pretrain:
			note_encoder_path = os.path.join(self.save_loc, 'Note2HarmonySoftmax', f'checkpoint.{tools.PT_EXT}')
			harmony_encoder_path = os.path.join(self.save_loc, 'Harmony2HarmonySoftmax', f'checkpoint.{tools.PT_EXT}')
			if os.path.exists(note_encoder_path):
				note_encoder_incompatible_keys = model.load_state_dict(torch.load(note_encoder_path), strict=False)
			if os.path.exists(harmony_encoder_path):
				harmony_encoder_incompatible_keys = model.load_state_dict(torch.load(harmony_encoder_path), strict=False)


		return model

	def run(self, train_loader, validation_loader):

		self.train(training_loader, validation_loader)

	def get_model_path(self, with_filename=False):
		# Get the path to the model directory
		path = os.path.join(self.save_loc, f'{self.model_name()}')

		if with_filename:
			path = os.path.join(self.save_loc, f'{self.model_name()}', f'checkpoint.{tools.PT_EXT}')


		return path
	
	@classmethod
	def model_name(cls):

		tag = cls.__name__.rstrip('Trainer')

		return tag