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

class Encoder(nn.Module):

	def __init__(self, harmony_type):
		super().__init__()

		self.harmony_type = harmony_type
		self.harmony_components = tools.HARMONY_COMPONENTS[harmony_type]
		self.harmony_component_dims = tools.HARMONY_COMPONENT_DIMS[harmony_type]
		self.num_harmonies = tools.NUM_HARMONIES[harmony_type]
		self.vec_size = tools.HARMONY_VEC_SIZE[harmony_type]
		self.embedding_size = tools.EMBEDDING_SIZE[harmony_type]



####################
#   Note Encoder   #
####################

# Wrapper of the note encoder (Note context transform)
class NoteEncoder(Encoder):
	'''
	Init Arguments:
		num_pc: 			number of pitch class
		embedding_size:     size of the encoded embedding vector of each frame
		drop_rate:          probability of dropping out a parameter in the module

	'''
	def __init__(self, harmony_type):
		super().__init__(harmony_type)

		# Initialize a DenseGRU module
		self.dense_gru = DenseGRU(embedding_size=self.embedding_size, drop_rate=0.6)

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
	def __init__(self, in_channels=tools.NUM_PC*2, block_depths=[4, 2, 2], growth_rates=[10, 4, 4], kernel_sizes=[7, 5, 3], \
			bottleneck=True, gru_hidden_size=64, embedding_size=16, drop_rate=0.6):
		super(DenseGRU, self).__init__()

		# Initialize the DenseNet module
		self.densenet = DenseNet(in_channels=in_channels, block_depths=block_depths, growth_rates=growth_rates, 
									kernel_sizes=kernel_sizes, bottleneck=bottleneck, drop_rate=drop_rate)
		
		# The output embedding dimension of each frame after DenseNet
		feature_size_after_densenet = tools.NUM_PC*2 + sum(pair[0] * pair[1] for pair in zip(block_depths, growth_rates))

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
class HarmonyEncoder(Encoder):
	
	def __init__(self, batch_size, harmony_type, device):
		super().__init__(harmony_type)
		self.batch_size = batch_size
		self.device = device

		# Initialize the harmony vectors
		self.harmony_vector = torch.zeros(len(self.harmony_components), self.num_harmonies, requires_grad=False)
		self.harmony_vector_multi_hot = torch.zeros(self.num_harmonies, self.vec_size, requires_grad=False)
		self.compute_harmony_vector()
		self.harmony_vector = self.harmony_vector.to(device)
		self.harmony_vector_multi_hot = self.harmony_vector_multi_hot.to(device)

		self.hidden_size = torch.tensor(self.vec_size / 2).int()
		self.linear1 = nn.Linear(self.vec_size, self.hidden_size)
		self.linear1_bn = nn.BatchNorm1d(self.hidden_size)

		self.linear2 = nn.Linear(self.hidden_size, self.embedding_size)
		self.linear2_bn = nn.BatchNorm1d(self.embedding_size)

		self.relu = nn.ReLU(inplace=True)
	
	# Compute the input vecotr for all harmony labels
	def compute_harmony_vector(self):

		# Loop through all harmony labels
		for harmony_index in range(self.num_harmonies):

			# Update the corresponding harmony vector
			self.compute_harmony_vector_single(harmony_index)

		# Repeat the harmony vector for each sample in the batch for parallel processing
		self.harmony_vector = self.harmony_vector.unsqueeze(0).repeat(self.batch_size, 1, 1)
		self.harmony_vector_multi_hot = self.harmony_vector_multi_hot.unsqueeze(0).repeat(self.batch_size, 1, 1)

	# Compute the input vector for a single harmony labels
	def compute_harmony_vector_single(self, harmony_index):

		# Extract the index of components of the harmony
		component_indexes = tools.Harmony.parse_harmony_index(harmony_index, self.harmony_type)		

		cum_component_dim = 0
		for i, component_dim_cur in enumerate(self.harmony_component_dims):
			self.harmony_vector[i, harmony_index] = component_indexes[i]
			self.harmony_vector_multi_hot[harmony_index, cum_component_dim + component_indexes[i]] = 1
			cum_component_dim += component_dim_cur

	def forward(self):

		out = self.relu(self.linear1_bn(self.linear1(self.harmony_vector_multi_hot).permute(0, 2, 1))).permute(0, 2, 1)
		out = self.relu(self.linear2_bn(self.linear2(out).permute(0, 2, 1))).permute(0, 2, 1)

		return out