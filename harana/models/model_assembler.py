# Building the model

# My import
from ..utils import chord
from .semi_crf import SemiCRF
from .densenet import DenseNet, TransitionBlock
from .nade import BlockNADE

# Regular import
import torch
import torch.nn as nn
import torch.nn.functional as F




####################
#   Note Encoder   #
####################

# Wrapper of the note encoder (Note context transform)
class NoteContextTransform(nn.Module):
	'''
	Init Arguments:
		num_pc: 			number of pitch class
		transform_type:		type of note context transformation

	'''
	def __init__(self, num_pc, transform_type):
		super(NoteContextTransform, self).__init__()

		# Initialize a DenseGRU module
		self.dense_gru = DenseGRU(num_pc)

		# Initialize a channel-wise 1d CNN module
		self.cnn = CNN1d(num_pc)
		
		# A softmax layer to convert the frame-level embedding to a distribution-like vector
		self.softmax = nn.Softmax(dim=2)
		
		self.transform_type = transform_type

	def forward(self, x):

		if self.transform_type == "none":
			out = x
		elif self.transform_type == "cnn":
			out = self.cnn(x)
		elif self.transform_type == "dense_gru":
			out = self.dense_gru(x)

		out = self.softmax(out)
		
		return out


# A plain channel-wise 1D CNN module
class CNN1d(nn.Module):
	'''
	Init Arguments:
		num_pc: 			number of pitch class
		in_planes: 			the number of input planes (planes with shape num_pc * sample_size)
		out_planes: 		the number of input planes (planes with shape num_pc * sample_size), could be interpreted as the number of channel-wise conv1d filters of each block
		kernel_size: 		length of the 1d kernel of each block (same for all the conv layers in each block)
		bottleneck: 		whether to use the bottleneck layers, see more details in the BottleneckBlock class of densenet.py
		drop_rate: 			the probability of elements in each input tensor to be zero

	Note that all in_planes and out_planes are 1. This is the simple case where there is only single 1d filter in each conv layer.
	More generally, the shapes are consistent as long as:
		in_planes = 1
		in_planes[k] = out_planes[k - 1]
	'''
	def __init__(self, num_pc, in_planes=[1, 1, 1], out_planes=[1, 1, 1], kernel_sizes=[3, 3, 3, 3], drop_rate=0.0):
		super(CNN1d, self).__init__()
		self.num_pc = num_pc
		self.conv1 = TransitionBlock(num_pc, in_planes=in_planes[0], out_planes=out_planes[0], kernel_size=kernel_sizes[0], drop_rate=drop_rate)
		self.conv2 = TransitionBlock(num_pc, in_planes=in_planes[1], out_planes=out_planes[1], kernel_size=kernel_sizes[1], drop_rate=drop_rate)
		self.conv3 = TransitionBlock(num_pc, in_planes=in_planes[2], out_planes=out_planes[2], kernel_size=kernel_sizes[2], drop_rate=drop_rate)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		
		# x.shape: (batch_size, sample_size, num_pc)
		# Set the pitch class dimension before the time dimension, as this representation is used in torch.nn.Conv1d
		# Need to unsqueeze the second to last dimension to set up the transformation to multiple planes as used in the channel-wise conv1d. 
		out = x.permute(0, 2, 1).unsqueeze(-2)
		# x.shape: (batch_size, num_pc, 1, sample_size)

		out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.relu(out)
		# x.shape: (batch_size, num_pc, 1, sample_size)
		
		out = out.permute(0, 3, 1, 2)
		# x.shape: (batch_size, sample_size, num_pc, 1)
		
		out = out.flatten(start_dim=2)
		# x.shape: (batch_size, sample_size, num_pc)

		return out


# A 1D CRNN module with DenseNet and GRU
class DenseGRU(nn.Module):
	'''
	Init Arguments:
		num_pc: 			number of pitch class
		block_depth: 		number of convolutional layers of each block
		growth_rates: 		number filters of the convolutional layers in each block (same for all the conv layers in each block)
		kernel_size: 		length of the 1d kernel of each block (same for all the conv layers in each block)
		bottleneck: 		whether to use the bottleneck layers, see more details in the BottleneckBlock class of densenet.py
		drop_rate: 			the probability of elements in each input tensor to be zero
		gru_hidden_size: 	the dimension of hidden state output of GRU at each time step

	The parameter values follow the setup in Micchi et al, 2021
	'''
	def __init__(self, num_pc, block_depths=[3, 2, 2], growth_rates=[10, 4, 4], kernel_sizes=[7, 3, 3],
                 bottleneck=True, drop_rate=0.0, gru_hidden_size=178):
		super(DenseGRU, self).__init__()

		# Initialize the DenseNet module
		self.densenet = DenseNet(num_pc, block_depths, growth_rates, kernel_sizes, bottleneck, drop_rate)
		
		# The output embedding dimension of each frame after DenseNet
		num_feature_after_densenet = num_pc * (block_depths[-1] + 1) * growth_rates[-1]
		
		# Initialize the GRU layer
		self.gru = nn.GRU(input_size=num_feature_after_densenet, hidden_size=gru_hidden_size, batch_first=True)

		# Project the embedding size of each frame back to the number of pitch class
		self.linear = nn.Linear(gru_hidden_size, num_pc)

	def forward(self, x):
		
		# x.shape: (batch_size, sample_size, num_pc)
		# Set the pitch class dimension before the time dimension, as this representation is used in torch.nn.Conv1d
		out = x.permute(0, 2, 1).unsqueeze(-2)
		# x.shape: (batch_size, num_pc, sample_size)
		
		out = self.densenet(out)
		# out.shape: (batch_size, num_pc, num_feature_each_pc, sample_size)
		
		# Set the time dimension before feature dimensions, as this representation is used in torch.nn.GRU
		out = out.permute(0, 3, 1, 2)
		# out.shape: (batch_size, sample_size, num_pc, num_feature_each_pc)
		
		# Flatten the feature plane of each frame into a 1d vector
		out = out.flatten(start_dim=2)
		# out.shape: (batch_size, sample_size, num_feature_each_frame)

		out, _ = self.gru(out)
		# out.shape: (batch_size, sample_size, gru_hidden_size)

		out = self.linear(out)
		# out.shape: (batch_size, sample_size, num_pc)

		return out




#####################
#   Chord Encoder   #
#####################

# Wrapper of the chord encoder (reweight the chordal pitch class vector based on scale degree)
class ChordalPCWeightTransform(nn.Module):
	'''
	Init Arguments:
		num_label: 			number of chord labels
		num_pc: 			number of pitch class
		transform_type:		type of chordal pc weight transformation
		device:				the device where the model is running
	'''
	def __init__(self, num_label, num_pc, chord_transform_type, device):
		super(ChordalPCWeightTransform, self).__init__()
		
		self.num_label = num_label
		self.num_pc = num_pc
		
		'''
		Three types of chord transform:
			weight_vector: a single weight vector to be element-wise multiplied with the chordal pc vectors
			fc1: one fully-connected layer
			fc2: two fully-connected layer
		'''
		
		# The weight vector is initialized to be approximately the importance of each scale degree
		self.scale_degree_weight = nn.Parameter(torch.tensor([1, 0.01, 0.01, 0.9, 0.9, 0.01, 0.1, 0.8, 0.1, 0.1, 0.9, 0.7, 0]))
		
		self.linear1 = nn.Linear(num_pc, num_pc)
		self.linear2 = nn.Linear(num_pc, num_pc)
		self.softmax = nn.Softmax(dim=2)
		
		self.chord_transform_type = chord_transform_type
		self.device = device

	# Permute the pitch class vector
	def permute_pc(self, chordal_pc_vector, mode):

		# The variable to store the permuted chordal pc vectors
		chordal_pc_vector_permuted = torch.tensor([])
		chordal_pc_vector_permuted = chordal_pc_vector_permuted.to(self.device)

		# Loop through all the chord labels
		for label_idx in range(self.num_label):

			# Get the root pc of the chord
			root_pc, _, _ = chord.parse_symbol(chord.index2symbol(label_idx))

			# Slice the 12 actual pitch classes from the vector
			chordal_pc_vector_cur_12 = chordal_pc_vector[:, label_idx, :self.num_pc - 1]

			# If the target pc vector has the root pc at the first entry
			if mode == "root_first":
				chordal_pc_vector_permuted_cur_12 = chordal_pc_vector_cur_12.roll(self.num_pc - 1 - root_pc, dims=1)
			# If the target pc vector has the original order
			elif mode == "original":
				chordal_pc_vector_permuted_cur_12 = chordal_pc_vector_cur_12.roll(root_pc, dims=1)
			
			# Concatenate the no-pitch class back to the vector
			chordal_pc_vector_permuted_cur_13 = torch.cat([chordal_pc_vector_permuted_cur_12, chordal_pc_vector[:, label_idx, -1].unsqueeze(-1)], dim=1)
			
			# Stach each permuted chordal pc vector at the num_label dimension
			chordal_pc_vector_permuted = torch.cat([chordal_pc_vector_permuted, chordal_pc_vector_permuted_cur_13[:, None, :]], dim=1)
		
		return chordal_pc_vector_permuted
	
	def forward(self, chordal_pc_vector):

		# chordal_pc_vector.shape: (num_label, num_pc)
		out = self.permute_pc(chordal_pc_vector, mode="root_first")
		# out.shape: (num_label, num_pc) and remains the same in the following steps

		if self.chord_transform_type == "None":
			out = out
		elif self.chord_transform_type == "weight_vector":
			out = out * self.scale_degree_weight
		elif self.chord_transform_type == "fc1":
			out = self.linear1(out)
		elif self.chord_transform_type == "fc1":
			out = self.linear1(out)
			out = self.linear2(out)
		
		out = self.permute_pc(out, mode="original")
		
		out = self.softmax(out)
		
		return out


####################
#   Score Module   #
#####################

# Wrapper to compute the score for each segment
class SegmentScore(nn.Module):
	def __init__(self, batch_size, sample_size, num_label, num_pc, chord_transform_type, device):
		super(SegmentScore, self).__init__()
		
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.num_label = num_label
		self.num_pc = num_pc

		self.device = device

		# Initialize the chordal pc weight transform module
		self.chordal_pc_weight_transform = ChordalPCWeightTransform(self.num_label, self.num_pc, chord_transform_type, self.device)

		# Initialize the chordal pc vectors
		self.chordal_pc_vector = torch.zeros(self.num_label, self.num_pc, requires_grad=False)
		self.compute_chordal_pc_vector()
		self.chordal_pc_vector = self.chordal_pc_vector.to(self.device)

		'''
		self.chord_prior = torch.zeros(self.num_label)
		self.init_chord_prior()
		self.chord_prior = nn.Parameter(self.chord_prior)
		self.chordal_prior = self.chord_prior.to(device)
		'''

		# Initialize the cross attention module of the query chord labels to the note embedding
		self.chord_note_attention = ScaledDotProductAttention(self.num_pc)

	# Compute the chordal pc vector for all chords
	def compute_chordal_pc_vector(self):

		# Loop through chord labels
		for label_idx in range(self.num_label - 1):

			# Update the chordal pc vector for the chord
			self.chordal_pc_vector[label_idx, :] = self.compute_chordal_pc_vector_single(label_idx)
		
		# Repeat the chordal pc vectors for each sample in the batch for parallel processing
		self.chordal_pc_vector = self.chordal_pc_vector.unsqueeze(0).repeat(self.batch_size, 1, 1)

	# Compute the chordal pc vector for a single chord
	def compute_chordal_pc_vector_single(self, label_idx):

		# Extract the chordal pc of the chord
		chordal_pc = chord.index2chordal_pc(label_idx)		

		# Create a 1-hot vector for the chordal pc
		chordal_pc_vector = torch.zeros(self.num_pc, 1)
		for pitch_class in range(self.num_pc):
			if pitch_class in chordal_pc:
				chordal_pc_vector[pitch_class, 0] = 1

		# Normalize the vectors to be distribution-like
		return chordal_pc_vector.squeeze(-1) / chordal_pc_vector.sum()

	
	# Compute the segment score for all segments (with all chord labels)
	# Since the computation is the same for all chord labels, we parallelize across them
	def compute_segment_score(self, pc_embedding_seq):
		
		# The variable to store all the segment scores
		segment_score = torch.zeros(self.batch_size, self.sample_size, self.sample_size, self.num_label)
		segment_score = segment_score.to(self.device)


		# Loop through start_frame
		for start_frame in range(self.sample_size):
			# Loop through end_frame
			for end_frame in range(start_frame, self.sample_size):

				# Update the segment scores of the current segment boundary
				segment_score[:, start_frame, end_frame, :] = self.compute_segment_score_single(start_frame=start_frame, pc_embedding_seq_segment=pc_embedding_seq[:, start_frame : end_frame + 1, :])

		return segment_score
	

	# Compute the segment score for a single segment (with all chord labels)
	def compute_segment_score_single(self, start_frame, pc_embedding_seq_segment):

		# Compute the attended note embedding of the segment
		attended_pc_embedding, attention = self.chord_note_attention(self.chordal_pc_vector, pc_embedding_seq_segment, pc_embedding_seq_segment, mask=None)

		# compute the segment score
		segment_score_single = (self.chordal_pc_weight * attended_pc_embedding).sum(-1)
		
		#segment_score_single = segment_score_single.mul(self.chord_prior)
		
		return segment_score_single

	def forward(self, pc_embedding_seq):
		
		# Compute the chord pc weight
		self.chordal_pc_weight = self.chordal_pc_weight_transform(self.chordal_pc_vector)

		# Compute the segment score
		segment_score = self.compute_segment_score(pc_embedding_seq)
		
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

class SoftmaxDecoder(nn.Module):

	def __init__(self):
		super(SoftmaxDecoder, self).__init__()

	def forward(self, pc_embedding_seq):
		return 0

# Wrapper for the decoder
class Decoder(nn.Module):
	def __init__(self, decode_type, batch_size, sample_size, num_label, num_pc, segment_max_len, device, chord_transform_type):
		super(Decoder, self).__init__()

		self.batch_size = batch_size
		self.decode_type = decode_type
		self.device = device
		self.segment_score = SegmentScore(self.batch_size, sample_size, num_label, num_pc, chord_transform_type, self.device)
		self.semicrf = SemiCRF(self.batch_size, sample_size, num_label, segment_max_len, self.device)

		self.softmax = SoftmaxDecoder()

		#label_dim = {'key': 24, 'tonocisation': 7, 'degree': 7, 'quality':12, 'inversion': 4, 'root': 24}
		#self.nade = BlockNADE(embedding_size=num_pc, visible_dim_list=[24, 7, 7, 12, 4, 24], hidden_dim=350)

	def decode(self):
		
		if self.decode_type == "semi_crf":
			return self.semicrf.decode()
		'''
		elif decode_type == "nade":
			return self.nade(self.pc_embedding_seq)
		'''

	def compute_loss(self, pc_embedding_seq, chord_seq):
		
		self.pc_embedding_seq = pc_embedding_seq
		if self.decode_type == "semi_crf":
			print("Computing segment score")
			self.semicrf.segment_score = self.segment_score(pc_embedding_seq)
			
			print("Computing path score")
			path_score = self.semicrf.compute_path_score(chord_seq)
			
			print("Computing log z")
			log_z = self.semicrf.compute_log_z()
			
			return - torch.sum(path_score - log_z) / self.batch_size
		elif self.decode_type == "softmax":
			chord_seq_pred = self.softmax(pc_embedding_seq)

		'''
		elif self.decode_type == "nade":
			chord_seq_pred = self.nade(pc_embedding_seq)
		'''




######################
#   Complete Model   #
######################

# Wrapper for the complete model
class ModelComplete(nn.Module):
	def __init__(self, batch_size, sample_size, segment_max_len, num_label, note_transform_type, chord_transform_type, decode_type, device):
		super(ModelComplete, self).__init__()

		self.batch_size = batch_size
		self.decode_type = decode_type
		self.device = device
		
		self.num_pc = 13
		
		# Initialize the note context transform module
		self.note_context_transform = NoteContextTransform(self.num_pc, note_transform_type)
		self.decoder = Decoder(decode_type, batch_size, sample_size, num_label, self.num_pc, segment_max_len, self.device, chord_transform_type)

	def forward(self, batch):
		self.decoder.segment_score.chordal_pc_vector.detach_()
		
		pc_dist_seq = batch["pc_dist_seq"]
		chord_seq = batch["chord_seq"]
		pc_dist_seq = pc_dist_seq.to(self.device)
		chord_seq = chord_seq.to(self.device)

		'''
		song_idx = batch["song_idx"]
		sample_idx_in_song = batch["sample_idx_in_song"]


		sample_idx = 1
		sample_idx_in_song_cur = sample_idx_in_song[sample_idx]
		print(f"song index: {song_idx[sample_idx]}")

		qn_offset = 0

		sample_size = 48
		fpqn = 4
		qnps = sample_size/fpqn

		start_qn = qn_offset + qnps * sample_idx_in_song_cur

		print(f"start_qn: {start_qn}")
		print(chord_seq[sample_idx])
		print(chord.index2symbol(1))
		'''

		pc_embedding_seq = self.note_context_transform(pc_dist_seq)

		return self.decoder.compute_loss(pc_embedding_seq, chord_seq)






