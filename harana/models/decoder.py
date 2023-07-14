# Building the model

# My import
from .. import tools
from .densenet import DenseNet
from .semi_crf import SemiCRF
from .nade import NADE

# Regular import
import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper for the decoder
class Decoder(nn.Module):
	
	def __init__(self, harmony_type, batch_size, sample_size, device):
		
		super().__init__()
		
		self.harmony_type = harmony_type
		self.num_harmonies = 0
		self.embedding_size = tools.EMBEDDING_SIZE
		if harmony_type != "toy":
			self.num_harmonies = tools.NUM_HARMONIES[harmony_type]
			self.harmony_components = tools.HARMONY_COMPONENTS[harmony_type]
			self.harmony_component_dims = tools.HARMONY_COMPONENT_DIMS[harmony_type]

		self.batch_size = batch_size
		self.harmony_sample_size = sample_size
		if harmony_type != "toy":
			self.harmony_sample_size = round(sample_size / tools.HARMONY_UNIT_SPAN)
		self.device = device


	@abstractmethod
	def decode(self):
		return NotImplementedError

	def segment_postprocess(self, segment_result):
		
		interval_result = []
		label_result = []
		
		frame_result = torch.zeros(self.batch_size, len(self.harmony_components), self.harmony_sample_size)
		frame_result = frame_result.to(self.device)
		for sample_index, segment_result_sample in enumerate(segment_result):
			
			interval_result_sample = torch.zeros(len(segment_result_sample), 2)
			label_result_sample = []
			
			for i, [harmony_index, start_frame, end_frame] in enumerate(segment_result_sample):
				
				interval_result_sample[i, :] = torch.tensor([start_frame, end_frame])
				label_result_sample.append(tools.Harmony.index2symbol(harmony_index, self.harmony_type))
				
				component_indexes = tools.Harmony.parse_harmony_index(harmony_index, self.harmony_type)
				for i, _ in enumerate(self.harmony_components):
					frame_result[sample_index, i, start_frame : end_frame + 1] = component_indexes[i]

			label_result.append(label_result_sample)
			interval_result.append(interval_result_sample)

		return frame_result, interval_result, label_result

	def frame_postprocess(self, frame_result):

		frame_result = frame_result.to(self.device)
		interval_result = []
		label_result = []
		
		for sample_index in range(self.batch_size):
			segment_result_sample = []
			label_result_sample = []
			
			start_frame = 0
			end_frame = 0
			
			for frame_index in range(1, self.harmony_sample_size):

				component_indexes_cur = []
				new_segment = False
				for i, _ in enumerate(self.harmony_components):
					component_indexes_cur.append(frame_result[sample_index, i, frame_index - 1].item())
					if frame_result[sample_index, i, frame_index] != frame_result[sample_index, i, frame_index - 1]:
						new_segment = True
				
				if new_segment:
					end_frame = frame_index - 1
					harmony_index = tools.Harmony.combine_component_indexes(component_indexes_cur, self.harmony_type)
					segment_result_sample.append([harmony_index, start_frame, end_frame])
					start_frame = frame_index
			
			end_frame = frame_index - 1	
			harmony_index = tools.Harmony.combine_component_indexes(component_indexes_cur, self.harmony_type)
			segment_result_sample.append([harmony_index, start_frame, end_frame])

			interval_result_sample = torch.zeros(len(segment_result_sample), 2)
			for i, [harmony_index, start_frame, end_frame] in enumerate(segment_result_sample):
					
				interval_result_sample[i, :] = torch.tensor([start_frame, end_frame])
				label_result_sample.append(tools.Harmony.index2symbol(harmony_index, self.harmony_type))

			interval_result.append(interval_result_sample)
			label_result.append(label_result_sample)
				
		return frame_result, interval_result, label_result


		


####################
#   Score Module   #
#####################

# Wrapper to compute the score for each segment
class SegmentScore(nn.Module):
	def __init__(self, with_attention, with_adversarial, batch_size, sample_size, num_label, embedding_size, device):
		super().__init__()
		
		self.with_attention = with_attention
		self.with_adversarial = with_adversarial

		self.batch_size = batch_size
		self.sample_size = sample_size
		self.num_label = num_label
		self.device = device

		self.cs_sim = nn.CosineSimilarity(dim=-1)

		# Initialize the cross attention module of the query harmony embedding to the note embedding
		self.harmony_note_attention = ScaledDotProductAttention(embedding_size)

		self.pc_non_matching_score_weight = 0.1
		self.adversarial_score_weight = 0.001

	
	# Compute the segment score for all segments (with all harmony labels)
	# Since the computation is the same for all harmony labels, we parallelize across them
	def compute_segment_score(self, active_frame_out, inactive_frame_out, harmony_pc_vector, harmony_component_vector, harmony_components, PC_only):
		
		# The variable to store all the segment scores
		segment_score = torch.zeros(self.batch_size, self.sample_size, self.sample_size, self.num_label)
		segment_score = segment_score.to(self.device)

		active_pc_out = active_frame_out['pc']
		inactive_pc_out = inactive_frame_out['pc']

		# Loop through start_frame
		for start_frame in range(self.sample_size):
			# Loop through end_frame
			for end_frame in range(start_frame, self.sample_size):
				# Update the segment scores of the current segment boundary

				active_pc_out_segment = active_pc_out[:, start_frame : end_frame + 1, :]
				inactive_pc_out_segment = active_pc_out[:, start_frame : end_frame + 1, :]
				harmony_pc_vector_batch = harmony_pc_vector[0:self.batch_size, ...]

				# The similarity between active pc prediction and gt harmony pc
				segment_score[:, start_frame, end_frame, :] += self.compute_similarity_single(active_pc_out_segment, harmony_pc_vector_batch)

				if self.with_adversarial:
					# The similarity between active pc prediction and the complement of gt harmony pc (extra non-chordal note)
					segment_score[:, start_frame, end_frame, :] -= self.pc_non_matching_score_weight * self.compute_similarity_single(active_pc_out_segment, 1 - harmony_pc_vector_batch)

					# The similarity between inactive pc prediction and gt harmony pc (missing chordal note)
					segment_score[:, start_frame, end_frame, :] -= self.pc_non_matching_score_weight * self.compute_similarity_single(1 - active_pc_out_segment, harmony_pc_vector_batch)

					# The the average activation value of the estimated pc vector at the true activation positions from the inactive input (adversarial score)
					avg_soft_act_of_harmony_pc = torch.bmm(inactive_pc_out_segment, harmony_pc_vector_batch.transpose(1, 2)) / harmony_pc_vector_batch.sum(dim=-1)[:, None, :]
					segment_score[:, start_frame, end_frame, :] -= self.adversarial_score_weight * avg_soft_act_of_harmony_pc.sum(dim = 1) / (end_frame - start_frame + 1)

				if not PC_only:
					for i, component in enumerate(harmony_components):

						active_frame_out_segment = active_frame_out[component][:, start_frame : end_frame + 1, :]
						inactive_frame_out_segment = inactive_frame_out[component][:, start_frame : end_frame + 1, :]
						harmony_component_vector_batch = harmony_component_vector[i][0:self.batch_size, ...]
					
						# the probility of the ground truth label when the active music notes are given
						active_prob_of_label = torch.bmm(active_frame_out_segment, harmony_component_vector_batch.transpose(1, 2))
						segment_score[:, start_frame, end_frame, :] += active_prob_of_label.sum(dim=1) / (end_frame - start_frame + 1)

						if self.with_adversarial:
							# the probility of the ground truth label when the inactive music notes are given
							inactive_prob_of_label = torch.bmm(inactive_frame_out_segment, harmony_component_vector_batch.transpose(1, 2))
							segment_score[:, start_frame, end_frame, :] -= self.adversarial_score_weight * inactive_prob_of_label.sum(dim=1) / (end_frame - start_frame + 1)


		return segment_score


	# Compute the similarity score between pitch class vectors for a single segment (with all harmony labels)
	def compute_similarity_single(self, note_embedding_segment, harmony_embedding):
		

		if self.with_attention:

			# Compute the attended note embedding of the segment
			attended_note_embedding, attention = self.harmony_note_attention(harmony_embedding, note_embedding_segment, note_embedding_segment, mask=None)
			segment_score_single = self.cs_sim(attended_note_embedding, harmony_embedding)
		
		else:
			average_note_embedding = note_embedding_segment.sum(dim=1) / (note_embedding_segment.shape[1])
			
			# average_note_embedding.unsqueeze(-1).shape: (batch_size, embedding_size, 1)
			# harmony_embedding.shape: (batch_size, num_harmonies, embedding_size)
			segment_score_single = torch.bmm(harmony_embedding, average_note_embedding.unsqueeze(-1)).squeeze(-1)
		
		return segment_score_single

	def forward(self, active_frame_out, inactive_frame_out, harmony_pc_vector, harmony_component_vector, harmony_components, PC_only):

		# Compute the segment score
		segment_score = self.compute_segment_score(active_frame_out, inactive_frame_out, harmony_pc_vector, harmony_component_vector, harmony_components, PC_only)
		
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

# The frame-level decoder that classify the harmony of each each frame individually
class FrameLevelDecoder(Decoder):

	def __init__(self, harmony_type, batch_size, sample_size, device, with_note_inactive=False, nade_output=False):
		
		super().__init__(harmony_type, batch_size, sample_size, device)
		
		# Whether the inactive note input is used
		self.with_note_inactive = with_note_inactive
		self.nade_output = nade_output

		self.pc_non_matching_loss_weight = 0.1
		self.adversarial_loss_weight = 0.001
		self.small_offset = 0.000001
		
		# The linear heead used to find the class probabilities of each harmony component
		self.linear_heads = nn.ModuleList()
		# A separate decoder for each component
		for component_dim_cur in self.harmony_component_dims:
			self.linear_heads.append(tools.LinearLayers(2, self.embedding_size, component_dim_cur))
		self.softmax = nn.Softmax(dim=2)

		self.nade_decoder = NADE(harmony_type, self.embedding_size, self.embedding_size)
		
		# as well as the soft activations of the harmony pitch classes. 
		self.pc_linear_head = tools.LinearLayers(2, self.embedding_size, tools.NUM_PC)
		self.sigmoid = nn.Sigmoid()

		self.cce_loss = nn.CrossEntropyLoss()
		self.cs_sim = nn.CosineSimilarity(dim=-1)
	
	def forward(self, note_embedding, harmony_component_gt=None):
		
		out = dict()

		if self.nade_output:
			out = self.nade_decoder(note_embedding, harmony_component_gt=harmony_component_gt, training=True)
		else:
			for i, component in enumerate(self.harmony_components):
				out[component] = self.softmax(self.linear_heads[i](note_embedding))
			out['pc'] = self.sigmoid(self.pc_linear_head(note_embedding)).float()
		
		return out

	def decode(self, note_embedding):
		
		# The frame level harmony estimation for each frame
		frame_decode = torch.zeros(self.batch_size, len(self.harmony_components), self.harmony_sample_size)
		
		if self.nade_output:
			out = self.nade_decoder(note_embedding, training=False)
			for i, component in enumerate(self.harmony_components):
				frame_decode[:, i, :] = out[component]
		else:
			out = self.forward(note_embedding)
			for i, component in enumerate(self.harmony_components):
				frame_decode[:, i, :] = self.softmax(out[component]).argmax(dim=2)

		return self.frame_postprocess(frame_decode)

	def compute_loss(self, active_note_embedding, inactive_note_embedding, harmony_index_gt, harmony_component_gt, harmony_pc_vector):

		# Compute the output of the decoder for both active and inactive note input
		active_out = self.forward(active_note_embedding)
		inactive_out = self.forward(inactive_note_embedding)
		
		loss = 0

		num_entry = self.batch_size * self.harmony_sample_size

		# For each harmony component
		for i, component in enumerate(self.harmony_components):
			
			print(component)
			# compute the categorical cross entropy loss between estimated class probability distributions and one-hot gt labels
			print(f"cce_loss: {self.cce_loss(active_out[component].transpose(1, 2), harmony_component_gt[:, i, :].long())}")

			loss += self.cce_loss(active_out[component].transpose(1, 2), harmony_component_gt[:, i, :].long())
			
			# For the inactive note output,
			if self.with_note_inactive:
				# find the probability of the ground truth label
				prob_of_gt_label = (inactive_out[component] * F.one_hot(harmony_component_gt[:, i, :].long(), num_classes=self.harmony_component_dims[i])).sum(dim=-1)
				
				# and we want to minimize it. 
				nll = - torch.log(1 - prob_of_gt_label)
				print(f"inactive_prob_loss: {self.adversarial_loss_weight * nll.sum() / num_entry}")

				loss += self.adversarial_loss_weight * nll.sum() / num_entry
		
		out_pc_gt = torch.bmm(F.one_hot(harmony_index_gt.long(), num_classes=self.num_harmonies).float(), harmony_pc_vector[0:self.batch_size, ...])

		# compute the binary cross entropy between the gt and estimated pc vector on each pitch class
		print(f"active_pc_loss: {(1 - self.cs_sim(active_out['pc'], out_pc_gt)).sum() / num_entry}")
		
		loss += (1 - self.cs_sim(active_out['pc'], out_pc_gt)).sum() / num_entry

		if self.with_note_inactive:


			# missing chordal note
			loss += self.pc_non_matching_loss_weight * self.cs_sim((1 - active_out['pc']), out_pc_gt).sum() / num_entry

			# extra non-chordal note
			loss += self.pc_non_matching_loss_weight * self.cs_sim(active_out['pc'], 1 - out_pc_gt).sum() / num_entry


			avg_soft_act_of_gt_pc = (inactive_out['pc'] * out_pc_gt).sum(dim=-1) / out_pc_gt.sum(dim=-1)

			avg_soft_act_of_gt_pc -= 0.000001
			nll = - torch.log(1 - avg_soft_act_of_gt_pc)

			print(f"inactive_pc_loss: {self.adversarial_loss_weight * nll.sum() / num_entry}")
			loss += self.adversarial_loss_weight * nll.sum() / num_entry

			
		return loss

	def compute_nade_loss(self, active_note_embedding, harmony_component_gt):

		active_out = self.forward(active_note_embedding, harmony_component_gt=harmony_component_gt)

		loss = 0
		# For each harmony component
		for i, component in enumerate(self.harmony_components):
			
			print(component)
			# compute the categorical cross entropy loss between estimated class probability distributions and one-hot gt labels
			print(f"cce_loss: {self.cce_loss(active_out[component].transpose(1, 2), harmony_component_gt[:, i, :].long())}")

			loss += self.cce_loss(active_out[component].transpose(1, 2), harmony_component_gt[:, i, :].long())

		return loss


# Wrapper for the decoder
class SemiCRFDecoder(Decoder):
	def __init__(self, harmony_type, batch_size, sample_size, max_seg_len, device,
					with_attention=True, with_adversarial=True, with_transition=True,
					transition_importance = 0.001, num_harmonies=0, embedding_size=0):
		
		super().__init__(harmony_type, batch_size, sample_size, device)

		self.with_transition = with_transition
		
		self.num_harmonies = num_harmonies if harmony_type == 'toy' else self.num_harmonies
		self.embedding_size = embedding_size if harmony_type == 'toy' else self.embedding_size
		
		self.segment_score = SegmentScore(with_attention, with_adversarial, batch_size, self.harmony_sample_size, self.num_harmonies, self.embedding_size, device)
		self.semicrf = SemiCRF(with_transition, transition_importance, batch_size, self.harmony_sample_size, max_seg_len, self.num_harmonies, device)
		self.relu = nn.ReLU()

		self.sigmoid = nn.Sigmoid()

	def decode(self):
		
		segment_decode = self.semicrf.decode()
		if self.harmony_type != 'toy':
			return self.segment_postprocess(segment_decode)
		else:
			return segment_decode

	def compute_loss(self, active_frame_out, inactive_frame_out, harmony_pc_vector, harmony_component_vector, harmony_index_gt, PC_only=False):

		self.segment_score.batch_size = self.batch_size
		self.semicrf.batch_size = self.batch_size
		
		print("Computing segment score")
		self.semicrf.segment_score = self.segment_score(active_frame_out, inactive_frame_out, harmony_pc_vector, harmony_component_vector, self.harmony_components, PC_only)
			
		print("Computing path score")
		path_score = self.semicrf.compute_path_score(harmony_index_gt)
			
		print("Computing log z")
		log_z = self.semicrf.compute_log_z()
		
		nll = - torch.sum(path_score - log_z) / self.batch_size

		return nll
