# Building the model

# My import
from .. import tools
from .semi_crf import SemiCRF
from .densenet import DenseNet

# Regular import
import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper for the decoder
class Decoder(nn.Module):
	def __init__(self, batch_size, sample_size, harmony_type):
		super().__init__()
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.harmony_type = harmony_type

	@abstractmethod
	def decode(self):

		return NotImplementedError

	def segment2frame(self, segment_decode, device):
		
		frame_decode = {}
		for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
			frame_decode[component] = torch.zeros(self.batch_size, self.sample_size)
			frame_decode[component] = frame_decode[component].to(device)
		for sample_index, segments_sample in enumerate(segment_decode):
			for i, [harmony_index, start_frame, end_frame] in enumerate(segments_sample):
				component_indexes = tools.Harmony.parse_harmony_index(harmony_index, self.harmony_type)
				for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
					frame_decode[component][sample_index, start_frame : end_frame + 1] = component_indexes[i]

		return frame_decode

	def frame2segment(self, frame_decode):

		segment_decode = {}
		for component in tools.HARMONY_COMPONENTS[self.harmony_type]:
			segment_decode_component = []
			for sample_index in range(self.batch_size):
				segment_decode_component_sample = []
				start_frame = 0
				end_frame = 0
				for frame_index in range(1, self.sample_size):
					if frame_decode[component][sample_index, frame_index] != frame_decode[component][sample_index, frame_index - 1]:
						end_frame = frame_index - 1
						segment_decode_component_sample.append([frame_decode[component][sample_index, frame_index - 1], start_frame, end_frame])
						start_frame = frame_index
				segment_decode_component.append(segment_decode_component_sample)
			segment_decode[component] = segment_decode_component

		return segment_decode
		


####################
#   Score Module   #
#####################

# Wrapper to compute the score for each segment
class SegmentScore(nn.Module):
	def __init__(self, batch_size, sample_size, num_label, embedding_size, device):
		super().__init__()
		
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.num_label = num_label
		self.device = device

		# Initialize the cross attention module of the query harmony embedding to the note embedding
		self.harmony_note_attention = ScaledDotProductAttention(embedding_size)
	
	# Compute the segment score for all segments (with all harmony labels)
	# Since the computation is the same for all harmony labels, we parallelize across them
	def compute_segment_score(self, note_embedding, harmony_embedding):
		
		# The variable to store all the segment scores
		segment_score = torch.zeros(self.batch_size, self.sample_size, self.sample_size, self.num_label)
		segment_score = segment_score.to(self.device)


		# Loop through start_frame
		for start_frame in range(self.sample_size):
			# Loop through end_frame
			for end_frame in range(start_frame, self.sample_size):
				# Update the segment scores of the current segment boundary
				segment_score[:, start_frame, end_frame, :] = self.compute_segment_score_single(note_embedding[:, start_frame : end_frame + 1, :], harmony_embedding)

		return segment_score


	# Compute the segment score for a single segment (with all chord labels)
	def compute_segment_score_single(self, note_embedding_segment, harmony_embedding):

		# Compute the attended note embedding of the segment
		attended_note_embedding, attention = self.harmony_note_attention(harmony_embedding, note_embedding_segment, note_embedding_segment, mask=None)

		# compute the segment score
		segment_score_single = (harmony_embedding * attended_note_embedding).sum(-1)
		
		return segment_score_single

	def forward(self, note_embedding, harmony_embedding):

		# Compute the segment score
		segment_score = self.compute_segment_score(note_embedding, harmony_embedding)
		
		return segment_score


# Scaled dot product attention adapted from github shreydesai/dotproduct_attention.py			
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, query_dim):
        super(ScaledDotProductAttention, self).__init__()
        
        self.scale = 1.0 / torch.sqrt(torch.tensor(query_dim))
        self.softmax = nn.Softmax(dim=2)
       
    def forward(self, query, keys, values, mask):
        # query: [B, Tt, Q] (target)
        # keys: [B, Ti, K] (input)
        # values: [B, Ti, V] (input)
        # assume Q == K
        
        # Compute attention
        keys = keys.permute(0, 2, 1) # [B, Ti, K] -> [B, K, Ti]
        attention = torch.bmm(query, keys) # [B, Tt, Q] * [B, K, Ti] = [B, Tt, Ti]
        attention = self.softmax(attention.mul_(self.scale))

        # Apply mask if there is one and then renormalize the attention
        if mask:
        	attention = attention * mask
        	attention.div(attention.sum(2, keepdim=True))

        # Weighted sum of the values with attention as the weight
        attended_values = torch.bmm(attention, values).squeeze(1) # [B, Tt, Ti] * [B, Ti, V] -> [B, Tt, V]

        return (attended_values, attention)


###############
#   Decoder   #
###############

class SoftmaxDecoder(Decoder):

	def __init__(self, batch_size, sample_size, harmony_type):
		super().__init__(batch_size, sample_size, harmony_type)

		self.linear_heads = nn.ModuleList()
		self.softmax = nn.Softmax(dim=2)
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		# A separate decoder for each component
		for component_dim_cur in tools.HARMONY_COMPONENT_DIMS[harmony_type]:
			self.linear_heads.append(nn.Linear(tools.EMBEDDING_SIZE[harmony_type], component_dim_cur))

	def forward(self, note_embedding):
		
		head_output = dict()
		cum_component_dim = 0
		for i, component_dim_cur in enumerate(tools.HARMONY_COMPONENT_DIMS[self.harmony_type]):
			head_output[tools.HARMONY_COMPONENTS[self.harmony_type][i]] = self.linear_heads[i](note_embedding)
			cum_component_dim += component_dim_cur
		
		return head_output

	def decode(self, note_embedding):
		frame_decode = dict()
		head_output = self.forward(note_embedding)
			
		for component in tools.HARMONY_COMPONENTS[self.harmony_type]:
			frame_decode[component] = self.softmax(head_output[component]).argmax(dim=2)

		segment_decode = self.frame2segment(frame_decode)
			
		return [frame_decode, segment_decode]

	def compute_loss(self, note_embedding, harmony_component_gt):

		component_weight = {
			'root' : 1,
			'quality' : 1,
			'inversion' : 1,
			'key' : 1,
			'degree' : 1
		}
		head_output = self.forward(note_embedding)
		loss = 0
		for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
			loss += component_weight[component] * self.cross_entropy_loss(head_output[component].permute(0, 2, 1), harmony_component_gt[:, i, :].long())
			
		return loss


# Wrapper for the decoder
class SemiCRFDecoder(Decoder):
	def __init__(self, batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, max_seg_len=tools.DEFAULT_MAX_SEG_LEN,
					num_harmonies=0, embedding_size=0, harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE, 
					transition_importance=0.01, with_transition=False):
		super().__init__(batch_size, sample_size, harmony_type)

		self.device = device
		self.with_transition = with_transition
		num_harmonies = num_harmonies if harmony_type=='test' else tools.NUM_HARMONIES[harmony_type]
		embedding_size = embedding_size if harmony_type=='test' else tools.EMBEDDING_SIZE[harmony_type]
		self.segment_score = SegmentScore(batch_size, sample_size, num_harmonies, embedding_size, device)
		self.semicrf = SemiCRF(batch_size, sample_size, max_seg_len, num_harmonies, device, transition_importance, with_transition)
		self.softmax = nn.Softmax(dim=2)

	def decode(self):
		
		segment_decode = self.semicrf.decode()
		for sample_index in range(self.batch_size):
			print(segment_decode[sample_index])
		
		frame_decode = dict()
		if self.harmony_type != 'test':
			frame_decode = self.segment2frame(segment_decode, self.device)

		return [frame_decode, segment_decode]

	def compute_loss(self, note_embedding, harmony_embedding, harmony_index_gt):

		print("Computing segment score")
		self.semicrf.segment_score = self.segment_score(note_embedding, harmony_embedding)
			
		print("Computing path score")
		path_score = self.semicrf.compute_path_score(harmony_index_gt)

		transition_weight_L2_loss = self.semicrf.transitions.flatten().square().sum()
			
		print("Computing log z")
		log_z = self.semicrf.compute_log_z()

		nll = - torch.sum(path_score - log_z) / self.batch_size
			
		return nll + transition_weight_L2_loss

# Wrapper for the decoder
class SemiCRFDecoder(Decoder):
	def __init__(self, batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, max_seg_len=tools.DEFAULT_MAX_SEG_LEN,
					num_harmonies=0, embedding_size=0, harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE, 
					transition_importance=0.01, with_transition=False):
		super().__init__(batch_size, sample_size, harmony_type)

		self.device = device
		num_harmonies = num_harmonies if harmony_type=='test' else tools.NUM_HARMONIES[harmony_type]
		embedding_size = embedding_size if harmony_type=='test' else tools.EMBEDDING_SIZE[harmony_type]
		self.segment_score = SegmentScore(batch_size, sample_size, num_harmonies, embedding_size, device)
		self.semicrf = SemiCRF(batch_size, sample_size, max_seg_len, num_harmonies, device, transition_importance, with_transition)
		self.softmax = nn.Softmax(dim=2)

	def decode(self):
		
		segment_decode = self.semicrf.decode()
		for sample_index in range(self.batch_size):
			print(segment_decode[sample_index])
		
		frame_decode = dict()
		if self.harmony_type != 'test':
			frame_decode = self.segment2frame(segment_decode, self.device)

		return [frame_decode, segment_decode]

	def compute_loss(self, note_embedding, harmony_embedding, harmony_index_gt):

		print("Computing segment score")
		self.semicrf.segment_score = self.segment_score(note_embedding, harmony_embedding)
			
		print("Computing path score")
		path_score = self.semicrf.compute_path_score(harmony_index_gt)

		transition_weight_L2_loss = self.semicrf.transitions.flatten().square().sum()
			
		print("Computing log z")
		log_z = self.semicrf.compute_log_z()

		nll = - torch.sum(path_score - log_z) / self.batch_size

		if with_transition:
			return nll + transition_weight_L2_loss
		else:
			return nll

# Wrapper for the decoder
class HierarchicalSemiCRFDecoder(Decoder):
	def __init__(self, batch_size=tools.DEFAULT_BATCH_SIZE, sample_size=tools.DEFAULT_SAMPLE_SIZE, max_seg_len=tools.DEFAULT_MAX_SEG_LEN,
					num_harmonies=0, embedding_size=0, harmony_type=tools.DEFAULT_HARMONY_TYPE, device=tools.DEFAULT_DEVICE, 
					transition_importance=0.01, with_transition=False):
		super().__init__(batch_size, sample_size, harmony_type)

		self.device = device
		num_harmonies = num_harmonies if harmony_type=='test' else tools.NUM_HARMONIES[harmony_type]
		embedding_size = embedding_size if harmony_type=='test' else tools.EMBEDDING_SIZE[harmony_type]
		self.segment_score = SegmentScore(batch_size, sample_size, num_harmonies, embedding_size, device)
		self.semicrf = SemiCRF(batch_size, sample_size, max_seg_len, num_harmonies, device, transition_importance, with_transition)
		self.softmax = nn.Softmax(dim=2)

	def decode(self):
		
		segment_decode = self.semicrf.decode()
		for sample_index in range(self.batch_size):
			print(segment_decode[sample_index])
		
		frame_decode = dict()
		if self.harmony_type != 'test':
			frame_decode = self.segment2frame(segment_decode, self.device)

		return [frame_decode, segment_decode]

	def compute_loss(self, note_embedding, harmony_embedding, harmony_index_gt):

		print("Computing segment score")
		self.semicrf.segment_score = self.segment_score(note_embedding, harmony_embedding)
			
		print("Computing path score")
		path_score = self.semicrf.compute_path_score(harmony_index_gt)

		transition_weight_L2_loss = self.semicrf.transitions.flatten().square().sum()
			
		print("Computing log z")
		log_z = self.semicrf.compute_log_z()

		nll = - torch.sum(path_score - log_z) / self.batch_size

		if with_transition:
			return nll + transition_weight_L2_loss
		else:
			return nll