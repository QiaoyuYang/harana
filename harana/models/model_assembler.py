# Building the model

# My import
from .. import tools
from .encoder import NoteEncoder, HarmonyEncoder
from .decoder import SoftmaxDecoder, SemiCRFDecoder
from .semi_crf import SemiCRF
from .densenet import DenseNet

# Regular import
import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

######################
#   Complete Model   #
######################
# Wrapper for the complete model
class HaranaModel(nn.Module):
	def __init__(self, batch_size, sample_size, harmony_type, device):
		super().__init__()

		self.batch_size = batch_size
		self.sample_size = sample_size
		self.harmony_type = harmony_type
		self.device = device
	
	@abstractmethod
	def get_loss(self, harmony_index_gt, harmony_component_gt):

		return NotImplementedError

	@staticmethod
	def prepare_data(batch, device):
		
		pc_act = batch[tools.KEY_PC_ACT].float()
		bass_pc = batch[tools.KEY_BASS_PC].float()
		harmony_index_gt = batch[tools.KEY_HARMONY_INDEX_GT]
		harmony_component_gt = batch[tools.KEY_HARMONY_COMPONENT_GT].float()


		pc_act = pc_act.to(device)
		bass_pc = bass_pc.to(device)
		harmony_index_gt = harmony_index_gt.to(device)
		harmony_component_gt = harmony_component_gt.to(device)

		return pc_act, bass_pc, harmony_index_gt, harmony_component_gt

# Wrapper for the complete model
class Note2HarmonySoftmax(HaranaModel):
	def __init__(self, note_encoder=None, harmony_decoder=None, batch_size=tools.DEFAULT_BATCH_SIZE,
					sample_size=tools.DEFAULT_SAMPLE_SIZE, harmony_type=tools.DEFAULT_HARMONY_TYPE,
					device=tools.DEFAULT_DEVICE):
		
		super().__init__(batch_size, sample_size, harmony_type, device)
		
		# Initialize the note context transform module
		if isinstance(note_encoder, NoteEncoder):
			self.note_encoder = encoder
		else:
			self.note_encoder = NoteEncoder(self.harmony_type)
		
		if isinstance(harmony_decoder, SoftmaxDecoder):
			self.harmony_decoder = decoder
		else:
			self.harmony_decoder = SoftmaxDecoder(batch_size, sample_size, harmony_type)

	def decode(self):
		
		decode_result = self.harmony_decoder.decode(self.note_embedding)

		return decode_result
	
	def get_loss(self, batch):
		
		return self.forward(batch)

	def forward(self, batch):

		pc_act, bass_pc, _, harmony_component_gt = HaranaModel.prepare_data(batch, self.device)
		self.note_embedding = self.note_encoder(torch.cat([pc_act, bass_pc], dim=1))
		loss = self.harmony_decoder.compute_loss(self.note_embedding, harmony_component_gt)
		
		return loss

# Wrapper for the complete model
class Harmony2HarmonySoftmax(HaranaModel):
	def __init__(self, harmony_encoder=None, harmony_decoder=None, batch_size=tools.DEFAULT_BATCH_SIZE,
					sample_size=tools.DEFAULT_SAMPLE_SIZE, harmony_type=tools.DEFAULT_HARMONY_TYPE,
					device=tools.DEFAULT_DEVICE):
		
		super().__init__(batch_size, sample_size, harmony_type, device)
		
		if isinstance(harmony_encoder, HarmonyEncoder):
			self.harmony_encoder = harmony_encoder
		else:
			self.harmony_encoder = HarmonyEncoder(batch_size, harmony_type, device)

		if isinstance(harmony_decoder, SoftmaxDecoder):
			self.harmony_decoder = harmony_decoder
		else:
			self.harmony_decoder = SoftmaxDecoder(batch_size, sample_size, harmony_type)
	
	def decode(self):
		
		decode_result = self.harmony_decoder.decode(self.harmony_embedding)
		
		return decode_result

	def get_loss(self, batch):
		
		return self.forward(batch)

	def forward(self, batch):
		
		self.harmony_encoder.harmony_vector_multi_hot.detach_()
		
		self.harmony_embedding = self.harmony_encoder.forward()

		loss = self.harmony_decoder.compute_loss(self.harmony_embedding, self.harmony_encoder.harmony_vector)

		return loss



# Wrapper for the complete model
class Note2HarmonySemiCRF(HaranaModel):
	def __init__(self, note_encoder=None, harmony_encoder=None, decoder=None,
					batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, 
					harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE):
		super().__init__(batch_size, sample_size, harmony_type, device)
		
		if isinstance(note_encoder, NoteEncoder):
			self.note_encoder = note_encoder
		else:
			self.note_encoder = NoteEncoder(harmony_type)

		if isinstance(harmony_encoder, HarmonyEncoder):
			self.harmony_encoder = harmony_encoder
		else:
			self.harmony_encoder = HarmonyEncoder(batch_size, harmony_type, device)

		if isinstance(decoder, SemiCRFDecoder):
			self.decoder = decoder
		else:
			self.decoder = SemiCRFDecoder(batch_size=batch_size, sample_size=sample_size, 
											num_harmonies=tools.NUM_HARMONIES[self.harmony_type], harmony_type=harmony_type, device=device)
	
	def decode(self):
		
		return self.decoder.decode()
	
	def get_loss(self, batch):
		
		return self.forward(batch)

	def forward(self, batch):

		pc_act, bass_pc, harmony_index_gt, _ = HaranaModel.prepare_data(batch, self.device)
		self.harmony_encoder.harmony_vector.detach_()
		self.decoder.semicrf.segment_score.detach_()
		
		note_embedding = self.note_encoder(torch.cat([pc_act, bass_pc], dim=1))

		harmony_embedding = self.harmony_encoder.forward()

		return self.decoder.compute_loss(note_embedding, harmony_embedding, harmony_index_gt)




