# Building the model

# My import
from .. import tools
from .semi_crf import SemiCRF
from .densenet import DenseNet

# Regular import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




####################
#   Note Encoder   #
####################

# Wrapper of the note encoder (Note context transform)
class NoteEncoder(nn.Module):
	'''
	Init Arguments:
		num_pc: 			number of pitch class
		embedding_size:     size of the encoded embedding vector of each frame
		drop_rate:          probability of dropping out a parameter in the module

	'''
	def __init__(self, embedding_size):
		super(NoteEncoder, self).__init__()

		# Initialize a DenseGRU module
		self.dense_gru = DenseGRU(embedding_size=embedding_size, drop_rate=0.6)

	def forward(self, x):
		
		out = self.dense_gru(x)
		
		return out


# A 1D CRNN module with DenseNet and GRU
class DenseGRU(nn.Module):
	'''
	Init Arguments:
		drop_rate: 			probability of elements in each input tensor to be zero
		gru_hidden_size: 	dimension of hidden state output of GRU at each time step

	The parameter values follow the setup in Micchi et al, 2021
	'''
	def __init__(self, in_channels=tools.NUM_PC, block_depths=[3, 2, 2], growth_rates=[10, 4, 4], kernel_sizes=[7, 3, 3], \
			bottleneck=True, gru_hidden_size=128, embedding_size=64, drop_rate=0.6):
		super(DenseGRU, self).__init__()

		# Initialize the DenseNet module
		self.densenet = DenseNet(in_channels=tools.NUM_PC, block_depths=[3, 2, 2], growth_rates=[10, 4, 4], kernel_sizes=[7, 3, 3], bottleneck=True, drop_rate=0.6)
		
		# The output embedding dimension of each frame after DenseNet
		feature_size_after_densenet = tools.NUM_PC + sum(pair[0] * pair[1] for pair in zip(block_depths, growth_rates))

		# Initialize the GRU layer
		self.gru = nn.GRU(input_size=feature_size_after_densenet, hidden_size=gru_hidden_size, batch_first=True)
		self.gru_bn = nn.BatchNorm1d(gru_hidden_size)
		
		# Project the embedding size of each frame back to the number of pitch class
		self.fc = nn.Linear(gru_hidden_size, embedding_size)
		self.fc_bn = nn.BatchNorm1d(embedding_size)

	def forward(self, x):
		
		# x.shape: (batch_size, num_pc, sample_size)
		
		out = self.densenet(x)
		# out.shape: (batch_size, feature_size_after_densenet, sample_size)
		
		out = out.permute(0, 2, 1)
		# out.shape: (batch_size, sample_size, feature_size_after_densenet)

		out, _ = self.gru(out)
		# out.shape: (batch_size, sample_size, gru_hidden_size)
		
		out = self.gru_bn(out.permute(0, 2, 1)).permute(0, 2, 1)
		# out.shape: (batch_size, sample_size, gru_hidden_size)

		out = self.fc(out)
		# out.shape: (batch_size, sample_size, embedding_size)

		out = self.fc_bn(out.permute(0, 2, 1)).permute(0, 2, 1)
		# out.shape: (batch_size, sample_size, embedding_size)

		return out




#####################
#   Harmony Encoder   #
#####################

# Wrapper of the harmony encoder
class HarmonyEncoder(nn.Module):
	'''
	Init Arguments:
		input_dim: 			dimension of input vector
		embedding_size:     size of the encoded embedding vector of each frame
	'''
	def __init__(self, harmony_type, device):
		super(HarmonyEncoder, self).__init__()
		
		self.harmony_type = harmony_type
		self.category_linear_proj = nn.ModuleList()

		# The encoder is two stage
		# First, create a separate linear projection for each component and concatenate the output
		for component_dim_cur in tools.HARMONY_COMPONENT_DIMS[harmony_type]:
			self.category_linear_proj.append(nn.Linear(component_dim_cur, component_dim_cur))
		
		# Second, use a single linear projection layer to transform the concatenated output to the embedding
		self.combined_linear_proj = nn.Linear(tools.EMBEDDING_SIZE[harmony_type], tools.EMBEDDING_SIZE[harmony_type])
		self.device = device
	
	def forward(self, harmony_vector):
		out = torch.empty_like(harmony_vector)
		out = out.to(self.device)

		cum_component_dim = 0
		for i, component_dim_cur in enumerate(tools.HARMONY_COMPONENT_DIMS[self.harmony_type]):
			component_cur_start = cum_component_dim
			component_cur_end = component_cur_start + component_dim_cur
			out[:, :, component_cur_start : component_cur_end] = \
				self.category_linear_proj[i](harmony_vector[:, :, component_cur_start:component_cur_end])
		cum_component_dim += component_dim_cur
		
		out = self.combined_linear_proj(out)
		
		return out


####################
#   Score Module   #
#####################

# Wrapper to compute the score for each segment
class SegmentScore(nn.Module):
	def __init__(self, batch_size, harmony_type, device):
		super(SegmentScore, self).__init__()
		
		self.batch_size = batch_size
		self.harmony_type = harmony_type
		self.device = device

		# Initialize the harmony transform module
		self.harmony_encoder = HarmonyEncoder(harmony_type, self.device)

		# Initialize the harmony vectors
		self.harmony_vector = torch.zeros(tools.NUM_HARMONIES[harmony_type], tools.EMBEDDING_SIZE[harmony_type], requires_grad=False)
		self.compute_harmony_vector()
		self.harmony_vector = self.harmony_vector.to(self.device)

		# Initialize the cross attention module of the query harmony embedding to the note embedding
		self.harmony_note_attention = ScaledDotProductAttention(tools.EMBEDDING_SIZE[harmony_type])

	# Compute the input vecotr for all harmony labels
	def compute_harmony_vector(self):

		# Loop through all harmony labels
		for harmony_index in range(tools.NUM_HARMONIES[self.harmony_type]):

			# Update the corresponding harmony vector
			self.compute_harmony_vector_single(harmony_index)
		
		# Repeat the harmony vector for each sample in the batch for parallel processing
		self.harmony_vector = self.harmony_vector.unsqueeze(0).repeat(self.batch_size, 1, 1)

	# Compute the input vector for a single harmony labels
	def compute_harmony_vector_single(self, harmony_index):

		# Extract the index of components of the harmony
		component_indexes = tools.Harmony.parse_harmony_index(harmony_index, self.harmony_type)		

		cum_component_dim = 0
		for i, component_dim_cur in enumerate(tools.HARMONY_COMPONENT_DIMS[self.harmony_type]):
			self.harmony_vector[harmony_index, cum_component_dim + component_indexes[i]] = 1
			cum_component_dim += component_dim_cur
	
	# Compute the segment score for all segments (with all harmony labels)
	# Since the computation is the same for all harmony labels, we parallelize across them
	def compute_segment_score(self, note_embedding_seq, harmony_embedding):
		harmony_embedding = self.harmony_encoder(self.harmony_vector)
		
		# The variable to store all the segment scores
		segment_score = torch.zeros(self.batch_size, tools.FRAMES_PER_SAMPLE, tools.FRAMES_PER_SAMPLE, tools.NUM_HARMONIES[self.harmony_type])
		segment_score = segment_score.to(self.device)


		# Loop through start_frame
		for start_frame in range(tools.FRAMES_PER_SAMPLE):
			# Loop through end_frame
			for end_frame in range(start_frame, tools.FRAMES_PER_SAMPLE):

				# Update the segment scores of the current segment boundary
				segment_score[:, start_frame, end_frame, :] = self.compute_segment_score_single(note_embedding_seq[:, start_frame : end_frame + 1, :], harmony_embedding)

		return segment_score

	# Compute the segment score for a single segment (with all chord labels)
	def compute_segment_score_single(self, note_embedding_seq_segment, harmony_embedding):

		# Compute the attended note embedding of the segment
		attended_note_embedding, attention = self.harmony_note_attention(harmony_embedding, note_embedding_seq_segment, note_embedding_seq_segment, mask=None)

		# compute the segment score
		segment_score_single = (harmony_embedding * attended_note_embedding).sum(-1)
		
		return segment_score_single

	def forward(self, note_embedding_seq):
		# Compute the chord pc weight
		harmony_embedding = self.harmony_encoder(self.harmony_vector)

		# Compute the segment score
		segment_score = self.compute_segment_score(note_embedding_seq, harmony_embedding)
		
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

class LinearDecoder(nn.Module):

	def __init__(self, harmony_type):
		super(LinearDecoder, self).__init__()

		self.linear_heads = nn.ModuleList()
		self.harmony_type = harmony_type
		# A separate decoder for each component
		for component_dim_cur in tools.HARMONY_COMPONENT_DIMS[harmony_type]:
			self.linear_heads.append(nn.Linear(component_dim_cur, component_dim_cur))

	def forward(self, note_embedding):
		
		head_output = dict()
		cum_component_dim = 0
		for i, component_dim_cur in enumerate(tools.HARMONY_COMPONENT_DIMS[self.harmony_type]):
			head_output[tools.HARMONY_COMPONENTS[self.harmony_type][i]] = self.linear_heads[i](note_embedding[:, :, cum_component_dim : cum_component_dim + component_dim_cur])
			cum_component_dim += component_dim_cur
		
		return head_output

# Wrapper for the decoder
class Decoder(nn.Module):
	def __init__(self, batch_size, segment_max_len, harmony_type, decode_type, device):
		super(Decoder, self).__init__()

		self.batch_size = batch_size
		self.harmony_type = harmony_type
		self.decode_type = decode_type
		self.device = device

		self.note_embedding = torch.zeros(batch_size, tools.FRAMES_PER_SAMPLE, tools.EMBEDDING_SIZE[harmony_type])
		self.note_embedding = self.note_embedding.to(self.device)

		self.segment_score = SegmentScore(batch_size, harmony_type, device)
		transition_importance = 0.001 / (tools.FRAMES_PER_SAMPLE - 1)
		self.semicrf = SemiCRF(batch_size, tools.NUM_HARMONIES[harmony_type], segment_max_len, transition_importance, device)

		self.linear_decoder = LinearDecoder(harmony_type)
		self.softmax = nn.Softmax(dim=2)
		self.cross_entropy_loss = nn.CrossEntropyLoss()

	def decode(self):
		
		decode_result = {}
		if self.decode_type == "semi_crf":
			return self.semicrf.decode()

		if self.decode_type == "softmax":
			head_output = self.linear_decoder(self.note_embedding)
			
			for component in tools.HARMONY_COMPONENTS[self.harmony_type]:
				decode_result[component] = self.softmax(head_output[component]).argmax(dim=2)
			
			return decode_result

	def compute_loss_semi_crf(self, harmony_index_gt):

		print("Computing segment score")
		self.semicrf.segment_score = self.segment_score(self.note_embedding)
			
		print("Computing path score")
		path_score = self.semicrf.compute_path_score(harmony_index_gt)

		transition_weight_L2_loss = self.semicrf.transitions.flatten().square().sum()
			
		print("Computing log z")
		log_z = self.semicrf.compute_log_z()

		nll = - torch.sum(path_score - log_z) / self.batch_size
			
		return nll + transition_weight_L2_loss

	def compute_loss_cross_entropy(self, harmony_component_gt):
		harmony_component_gt = harmony_component_gt.to(self.device)
		head_output = self.linear_decoder(self.note_embedding)
		loss = 0
		for i, component in enumerate(tools.HARMONY_COMPONENTS[self.harmony_type]):
			loss += self.cross_entropy_loss(head_output[component].permute(0, 2, 1), harmony_component_gt[:, i, :].long())
			
		return loss





######################
#   Complete Model   #
######################

# Wrapper for the complete model
class ModelComplete(nn.Module):
	def __init__(self, batch_size, segment_max_len, harmony_type, decode_type, device):
		super(ModelComplete, self).__init__()
		
		self.harmony_type = harmony_type
		self.decode_type = decode_type
		self.device = device
		
		# Initialize the note context transform module
		self.note_encoder = NoteEncoder(tools.EMBEDDING_SIZE[harmony_type])
		self.decoder = Decoder(batch_size, segment_max_len, harmony_type, decode_type, device)
	
	def get_loss(self, batch):
		return self.forward(batch)

	def forward(self, batch):
		self.decoder.segment_score.harmony_vector.detach_()
		self.decoder.semicrf.segment_score.detach_()
		
		pc_act = batch[tools.KEY_PC_ACT].float()
		chord_index_gt = batch[tools.KEY_CHORD_INDEX_GT]
		rn_index_gt = batch[tools.KEY_RN_INDEX_GT]
		chord_component_gt = batch[tools.KEY_CHORD_COMPONENT_GT].float()
		rn_component_gt = batch[tools.KEY_RN_COMPONENT_GT].float()

		pc_act = pc_act.to(self.device)
		
		if self.harmony_type == tools.HARMONY_TYPE_CHORD:
			harmony_index_gt = chord_index_gt
			harmony_component_gt = chord_component_gt
		elif self.harmony_type == tools.HARMONY_TYPE_RN:
			harmony_index_gt = rn_index_gt
			harmony_component_gt = rn_component_gt

		note_embedding = self.note_encoder(pc_act)

		self.decoder.note_embedding = note_embedding
		if self.decode_type == "softmax":
			return self.decoder.compute_loss_cross_entropy(harmony_component_gt)
		if self.decode_type == "semi_crf":
			return self.decoder.compute_loss_semi_crf(harmony_index_gt)






