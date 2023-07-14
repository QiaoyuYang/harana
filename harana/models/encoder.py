# Building the model

# My import
from ..tools import *
from .semi_crf import SemiCRF
from .densenet import DensePoolNet

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
		self.embedding_size = tools.EMBEDDING_SIZE




####################
#   Note Encoder   #
####################

class NoteEncoder(Encoder):

	def __init__(self, note_encoder_type, harmony_type):
		super().__init__(harmony_type)

		if note_encoder_type == "PC":
			self.note_transform = PCTransform(self.embedding_size)
		elif note_encoder_type == "CRNN":
			self.note_transform = CRNNTransform(self.embedding_size)

	def forward(self, x):

		batch_size, feature_size, _ = x.shape
		
		active_note_embedding = x
		inactive_note_embedding = x
		inactive_note_embedding[:, tools.NUM_PC:, :] = 1 - inactive_note_embedding[:, tools.NUM_PC:, :]

		active_note_embedding = self.note_transform(active_note_embedding, batch_size, feature_size)
		inactive_note_embedding = self.note_transform(inactive_note_embedding, batch_size, feature_size)

		return active_note_embedding, inactive_note_embedding

class PCTransform(nn.Module):

	def __init__(self, embedding_size):
		super().__init__()
	
	def forward(self, note_embedding, batch_size, feature_size):

		note_embedding = note_embedding.view(batch_size, feature_size, -1, tools.HARMONY_UNIT_SPAN).sum(dim=-1).transpose(1, 2)[..., :tools.NUM_PC]
		
		return note_embedding

# Wrapper of the CRNN note encoder
class CRNNTransform(nn.Module):

	def __init__(self, embedding_size):
		super().__init__()

		# Initialize a DenseGRU module
		self.dense_gru = DenseGRU(embedding_size=embedding_size, drop_rate=0.6)
		self.layernorm = nn.LayerNorm(embedding_size)
	
	def forward(self, note_embedding, batch_size, feature_size):

		note_embedding = self.dense_gru(note_embedding)
		note_embedding = self.layernorm(note_embedding)
		return note_embedding

# A 1D CRNN module with DenseNet and GRU
class DenseGRU(nn.Module):

	def __init__(self, in_channels=tools.NUM_PC*2, block_depths=[16, 8, 4], growth_rates=[32, 16, 8], kernel_sizes=[3, 5, 7], \
			bottleneck=True, gru_hidden_size=256, embedding_size=128, drop_rate=0.6):
		
		super().__init__()

		# Initialize the DenseNet module
		self.densenet = DensePoolNet(in_channels=in_channels, block_depths=block_depths, growth_rates=growth_rates, 
									kernel_sizes=kernel_sizes, bottleneck=bottleneck, drop_rate=drop_rate)
		
		# The output embedding dimension of each frame after DenseNet
		feature_size_after_densenet = tools.NUM_PC*2 + sum(pair[0] * pair[1] for pair in zip(block_depths, growth_rates))

		# Initialize the GRU layer
		self.gru = nn.GRU(input_size=feature_size_after_densenet, hidden_size=gru_hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
		self.gru_ln = nn.LayerNorm(gru_hidden_size*2)
		
		# Project the embedding size of each frame back to the number of pitch class
		self.fc1 = nn.Linear(gru_hidden_size*2, gru_hidden_size)
		self.fc1_ln = nn.LayerNorm(gru_hidden_size)

		self.fc2 = nn.Linear(gru_hidden_size, embedding_size)
		self.fc2_ln = nn.LayerNorm(embedding_size)

		self.lkrelu = nn.LeakyReLU(inplace=True)

		self.dropout = nn.Dropout(p=0.2)

	def forward(self, x):
		
		# x.shape: (batch_size, num_pc, sample_size)
		
		out = self.densenet(x).transpose(1, 2)
		# out.shape: (batch_size, sample_size, feature_size_after_densenet)

		out, _ = self.gru(out)
		# out.shape: (batch_size, sample_size, gru_hidden_size)
		
		out = self.lkrelu(self.gru_ln(out))
		# out.shape: (batch_size, sample_size, gru_hidden_size)

		out = self.dropout(out)

		out = self.lkrelu(self.fc1_ln(self.fc1(out)))
		# out.shape: (batch_size, sample_size, embedding_size)

		out = self.dropout(out)

		out = self.lkrelu(self.fc2_ln(self.fc2(out)))
		# out.shape: (batch_size, sample_size, embedding_size)


		return out



