# Building the model

# My import
from .. import tools
from .encoder import NoteEncoder
from .decoder import FrameLevelDecoder, SemiCRFDecoder

# Regular import
import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn

######################
#   Complete Model   #
######################
# Wrapper for the complete model
class HaranaModel(nn.Module):
	def __init__(self, note_encoder_type, harmony_type, batch_size, device):
		
		super().__init__()
		
		self.harmony_type = harmony_type
		self.batch_size = batch_size
		self.device = device

		# Initialize the note encoder
		self.note_encoder = NoteEncoder(note_encoder_type, harmony_type)


		self.compute_harmony_vector()
	
	@abstractmethod
	def get_loss(self):

		return NotImplementedError

	def prepare_data(self, batch):
		
		pc_act = batch[tools.KEY_PC_ACT].float()
		bass_pc = batch[tools.KEY_BASS_PC].float()
		harmony_index_gt = batch[tools.KEY_HARMONY_INDEX_GT]
		harmony_component_gt = batch[tools.KEY_HARMONY_COMPONENT_GT].float()


		pc_act = pc_act.to(self.device)
		bass_pc = bass_pc.to(self.device)
		harmony_index_gt = harmony_index_gt.to(self.device)
		harmony_component_gt = harmony_component_gt.to(self.device)

		return pc_act, bass_pc, harmony_index_gt, harmony_component_gt

	# Compute the input vecotr for all harmonies
	def compute_harmony_vector(self):

		self.num_harmonies = tools.NUM_HARMONIES[self.harmony_type]

		self.harmony_pc_vector = torch.zeros(self.num_harmonies, tools.NUM_PC)
		self.harmony_pc_vector = self.harmony_pc_vector.to(self.device)

		self.harmony_component_dims = tools.HARMONY_COMPONENT_DIMS[self.harmony_type]
		
		self.harmony_component_vector = []
		for i, component_dim_cur in enumerate(self.harmony_component_dims):
			self.harmony_component_vector.append(torch.zeros(self.num_harmonies, component_dim_cur))
			self.harmony_component_vector[-1] = self.harmony_component_vector[-1].to(self.device)

		# Loop through all harmony labels
		for harmony_index in range(self.num_harmonies):

			# Update the corresponding harmony vector
			self.compute_harmony_vector_single(harmony_index)

		self.harmony_pc_vector = self.harmony_pc_vector.repeat(self.batch_size, 1, 1)

		for i in range(len(self.harmony_component_dims)):
			self.harmony_component_vector[i] = self.harmony_component_vector[i].repeat(self.batch_size, 1, 1)
	
	# Compute the input vector for a single harmony label
	def compute_harmony_vector_single(self, harmony_index):

		component_indexes = tools.Harmony.parse_harmony_index(harmony_index, self.harmony_type)

		if self.harmony_type is tools.HARMONY_TYPE_K:
			key_index = harmony_index
			active_pc = tools.Key.get_key_pc(harmony_index)
		elif self.harmony_type is tools.HARMONY_TYPE_RQ:
			root_pc, quality_index = component_indexes
			active_pc = tools.Chord.get_chordal_pc(root_pc, quality_index)

		for i, pc in enumerate(active_pc):
			self.harmony_pc_vector[harmony_index, pc] = 1

		for i, component_index in enumerate(component_indexes):
			self.harmony_component_vector[i][harmony_index, component_index] = 1

# Wrapper for the CRNN model
class Note2HarmonySoftMax(HaranaModel):
	def __init__(self, note_encoder_type, harmony_type, batch_size, sample_size, device):
		
		super().__init__(note_encoder_type, harmony_type, batch_size, device)

		self.decoder = FrameLevelDecoder(harmony_type, batch_size, sample_size, device)

	def decode(self, stage):
		
		decode_result = self.decoder.decode(self.active_note_embedding)

		return decode_result
	
	def get_loss(self, batch, stage):
		
		return self.forward(batch)

	def forward(self, batch):

		self.harmony_pc_vector.detach_()

		pc_act, bass_pc, harmony_index_gt, harmony_component_gt = self.prepare_data(batch)
		
		self.decoder.batch_size = pc_act.shape[0]
		
		#note_input = pc_act
		note_input = torch.cat([pc_act, bass_pc], dim=1)
		self.active_note_embedding, self.inactive_note_embedding = self.note_encoder(note_input)
		
		loss = self.decoder.compute_loss(self.active_note_embedding, self.inactive_note_embedding, harmony_index_gt, harmony_component_gt, self.harmony_pc_vector, PC_only=True)
		
		return loss


# Wrapper for the NADE model
class Note2HarmonyNADE(HaranaModel):
	def __init__(self, note_encoder_type, harmony_type, batch_size, sample_size, device):
		
		super().__init__(note_encoder_type, harmony_type, batch_size, device)

		self.decoder = FrameLevelDecoder(harmony_type, batch_size, sample_size, device, with_note_inactive=True, nade_output=True)

	def decode(self, stage):
		
		decode_result = self.decoder.decode(self.active_note_embedding)

		return decode_result
	
	def get_loss(self, batch, stage):
		
		return self.forward(batch)

	def forward(self, batch):

		self.harmony_pc_vector.detach_()

		pc_act, bass_pc, harmony_index_gt, harmony_component_gt = self.prepare_data(batch)
		
		self.decoder.batch_size = pc_act.shape[0]
		
		#note_input = pc_act
		note_input = torch.cat([pc_act, bass_pc], dim=1)
		self.active_note_embedding, self.inactive_note_embedding = self.note_encoder(note_input)
		
		loss = self.decoder.compute_nade_loss(self.active_note_embedding, harmony_component_gt)
		
		return loss

# Wrapper for the complete model
class Note2HarmonyRuleSemiCRF(HaranaModel):
	def __init__(self, harmony_type, batch_size, sample_size, max_seg_len, device, note_encoder_type="PC"):
		
		super().__init__(note_encoder_type, harmony_type, batch_size, device)
		
		self.frame_level_decoder = FrameLevelDecoder(harmony_type, batch_size, sample_size, device)

		self.decoder = SemiCRFDecoder(harmony_type, batch_size, sample_size, max_seg_len, device)

		self.softmax = nn.Softmax(dim=2)
	
	def decode(self):
		
		return self.decoder.decode()
	
	def get_loss(self, batch, stage):
		
		return self.forward(batch, stage)

	def forward(self, batch, stage):

		self.decoder.semicrf.segment_score.detach_()


		pc_act, bass_pc, harmony_index_gt, _ = self.prepare_data(batch)
		
		self.decoder.batch_size = pc_act.shape[0]
		if stage == "Training":
			self.decoder.semicrf.transitions = self.decoder.semicrf.train_transitions
		elif stage == "Validation" or stage == "":
			self.decoder.semicrf.transitions = self.decoder.semicrf.validation_transitions
		
		#note_input = pc_act
		note_input = torch.cat([pc_act, bass_pc], dim=1)
		active_note_embedding, inactive_note_embedding = self.note_encoder(note_input)
		active_frame_out = {}
		active_frame_out['pc'] = active_note_embedding
		inactive_frame_out = {}
		inactive_frame_out['pc'] = inactive_note_embedding
		
		return self.decoder.compute_loss(active_frame_out, inactive_frame_out, self.harmony_pc_vector, self.harmony_component_vector, harmony_index_gt, PC_only=True)

# Wrapper for the complete model
class Note2HarmonySemiCRF(HaranaModel):
	def __init__(self, harmony_type, batch_size, sample_size, max_seg_len, device, note_encoder_type="CRNN"):
		
		super().__init__(note_encoder_type, harmony_type, batch_size, device)
		
		self.frame_level_decoder = FrameLevelDecoder(harmony_type, batch_size, sample_size, device)

		self.decoder = SemiCRFDecoder(harmony_type, batch_size, sample_size, max_seg_len, device)

		self.softmax = nn.Softmax(dim=2)
	
	def decode(self):
		
		return self.decoder.decode()
	
	def get_loss(self, batch, stage):
		
		return self.forward(batch, stage)

	def forward(self, batch, stage):

		self.decoder.semicrf.segment_score.detach_()


		pc_act, bass_pc, harmony_index_gt, _ = self.prepare_data(batch)
		
		self.decoder.batch_size = pc_act.shape[0]
		if stage == "Training":
			self.decoder.semicrf.transitions = self.decoder.semicrf.train_transitions
		elif stage == "Validation":
			self.decoder.semicrf.transitions = self.decoder.semicrf.validation_transitions

		
		#note_input = pc_act
		note_input = torch.cat([pc_act, bass_pc], dim=1)
		active_note_embedding, inactive_note_embedding = self.note_encoder(note_input)
		active_frame_out = self.frame_level_decoder(active_note_embedding)
		inactive_frame_out = self.frame_level_decoder(inactive_note_embedding)
		
		return self.decoder.compute_loss(active_frame_out, inactive_frame_out, self.harmony_pc_vector, self.harmony_component_vector, harmony_index_gt)



