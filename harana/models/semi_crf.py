# my import
from ..utils import core
from ..utils import chord

# regular import
import numpy as np
import torch
from torch import nn


class SemiCRF(nn.Module):

    def __init__(self, batch_size, sample_size, num_label, segment_max_len, device):
        super().__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_label = num_label
        self.segment_max_len = segment_max_len
        
        self.device = device

        self.segment_score = torch.zeros(self.batch_size, self.sample_size, self.sample_size, self.num_label)
        self.segment_score = self.segment_score.to(self.device)

        # the last number in the chord label index list is BOS tag.
        self.BOS_TAG_ID = num_label

        self.transitions = nn.Parameter(torch.empty(self.num_label + 1, self.num_label))
        self.init_weights()

    def init_weights(self):
        # initialize weights between -0.1 and 0.1
        nn.init.uniform_(self.transitions, 0, 1)


    def compute_path_score(self, chord_seq):
        path_score = torch.zeros(self.batch_size)
        path_score = path_score.to(self.device)
        for batch_idx in range(self.batch_size):
            label_idx_pre = self.BOS_TAG_ID
            label_idx_cur = int(chord_seq[batch_idx, 0].item())
            start_frame = 0
            end_frame = 0
            for frame_idx in range(self.sample_size):
                if chord_seq[batch_idx, frame_idx] != label_idx_cur:
                    end_frame = frame_idx - 1
                    path_score[batch_idx] += self.transitions[label_idx_pre, label_idx_cur] * (end_frame - start_frame + 1)
                    path_score[batch_idx] += self.segment_score[batch_idx, start_frame, end_frame, label_idx_cur]
                    start_frame = frame_idx
                    label_idx_pre = label_idx_cur
                    label_idx_cur = int(chord_seq[batch_idx, frame_idx].item())
            
            end_frame = frame_idx
            label_idx_cur = int(chord_seq[batch_idx, frame_idx].item())
            path_score[batch_idx] += self.transitions[label_idx_pre, label_idx_cur] * (end_frame - start_frame + 1)
            path_score[batch_idx] += self.segment_score[batch_idx, start_frame, end_frame, label_idx_cur]
        
        return path_score

    def compute_log_z(self):
        log_alpha = torch.zeros(self.batch_size, self.sample_size, self.num_label)
        log_alpha = log_alpha.to(self.device)
        
        in_segment_score_cur = self.segment_score[:, 0, 0, :]
        transition_score_cur = self.transitions[self.BOS_TAG_ID, :]
        log_alpha[:, 0, :] = torch.exp(in_segment_score_cur + transition_score_cur[None, :])

        
        for end_frame in range(1, self.sample_size):
            for label_cur_idx in range(self.num_label):
                max_span = min(end_frame + 1, self.segment_max_len)
                
                if end_frame + 1 <= max_span:
                    segment_len = end_frame + 1
                    segment_score_cur_starting_at_0 = self.segment_score[:, 0, end_frame, label_cur_idx] * segment_len
                    transition_score_from_BOS = self.transitions[self.BOS_TAG_ID, label_cur_idx] * segment_len
                    complete_score_cur_starting_at_0 = segment_score_cur_starting_at_0.unsqueeze(-1) + transition_score_from_BOS.unsqueeze(-1)[None, :]
                
                # The score of segments not starting at the first frame
                # j + 1 is the length of the frame
                segment_score_cur_not_starting_at_0 = torch.stack([self.segment_score[:, end_frame - max_span + 2 + j, end_frame, label_cur_idx] * (j + 1) for j in range(max_span - 1)], dim=1)
                transition_score_from_chord = torch.stack([self.transitions[torch.tensor(range(self.num_label)), label_cur_idx] * (j + 1) for j in range(max_span - 1)], dim=0)
                complete_score_cur_not_starting_at_0 = segment_score_cur_not_starting_at_0[:, :, None] + transition_score_from_chord[None, :, :]
                
                log_alpha_pre_from_chord = log_alpha[:, end_frame - max_span + 1 : end_frame, :]

                if end_frame + 1 <= max_span:
                    log_alpha[:, end_frame, label_cur_idx] = torch.cat([complete_score_cur_starting_at_0, (log_alpha_pre_from_chord + complete_score_cur_not_starting_at_0).flatten(start_dim=1)], dim=1).logsumexp(dim=1)
                else:
                    log_alpha[:, end_frame, label_cur_idx] = (log_alpha_pre_from_chord + complete_score_cur_not_starting_at_0).flatten(start_dim=1).logsumexp(dim=1)
        
        log_z = log_alpha[:, self.sample_size - 1, :].logsumexp(dim=1)

        return log_z

    def decode(self):
        print(f"Decoding the segments...")
        decoded_segments = []
        for sample_idx in range(self.batch_size):
            print(f"\tsample {sample_idx}")
            V = torch.zeros(self.sample_size, self.num_label)
            V = V.to(self.device)
            pre_pos = torch.zeros(self.sample_size, self.num_label, 2)
            pre_pos = pre_pos.to(self.device)
            
            in_segment_score_cur = self.segment_score[sample_idx, 0, 0, :]
            transition_score_cur = self.transitions[self.BOS_TAG_ID, :]
            V[0, :] = in_segment_score_cur + transition_score_cur
            
            for end_frame in range(1, self.sample_size):
                for label_cur_idx in range(self.num_label):
                    max_span = min(end_frame + 1, self.segment_max_len)

                    if end_frame + 1 <= max_span:
                        segment_len = end_frame + 1
                        segment_score_cur_starting_at_0 = self.segment_score[sample_idx, 0, end_frame, label_cur_idx] * segment_len
                        transition_score_from_BOS = self.transitions[self.BOS_TAG_ID, label_cur_idx] * segment_len
                        complete_score_cur_starting_at_0 = segment_score_cur_starting_at_0 + transition_score_from_BOS
                    
                    segment_score_cur_not_starting_at_0 = torch.stack([self.segment_score[sample_idx, end_frame - max_span + 2 + j, end_frame, label_cur_idx] * (j + 1) for j in range(max_span - 1)], dim=0)
                    transition_score_from_chord = torch.stack([self.transitions[torch.tensor(range(self.num_label)), label_cur_idx] * (j + 1) for j in range(max_span - 1)], dim=0)
                    complete_score_cur_not_starting_at_0 = segment_score_cur_not_starting_at_0[:, None] + transition_score_from_chord

                    V_pre = V[end_frame - max_span + 1 : end_frame, :]
                    
                    total_score_with_chord_change = V_pre + complete_score_cur_not_starting_at_0
                    
                    pre_pos_with_chord_change = (total_score_with_chord_change==torch.max(total_score_with_chord_change)).nonzero()
                    pre_pos_with_chord_change = pre_pos_with_chord_change[0, :]
                    total_score_with_chord_change_max = total_score_with_chord_change[pre_pos_with_chord_change[0], pre_pos_with_chord_change[1]]
                    
                    if total_score_with_chord_change_max > complete_score_cur_starting_at_0:
                        V[end_frame, label_cur_idx] = total_score_with_chord_change_max
                        pre_pos_with_chord_change[0] += end_frame - max_span + 1
                        pre_pos[end_frame, label_cur_idx] = pre_pos_with_chord_change
                    else:
                        V[end_frame, label_cur_idx] = complete_score_cur_starting_at_0
                        pre_pos[end_frame, label_cur_idx] = torch.tensor([-1, self.BOS_TAG_ID])

            end_frame_list = []
            end_frame_cur = torch.tensor(self.sample_size - 1)
            end_frame_list.append(int(end_frame_cur.item()))
            label_idx_list = []
            label_cur_idx = torch.argmax(V[end_frame_cur, :])
            label_idx_list.append(int(label_cur_idx.item()))
            while end_frame_cur > 0 and label_cur_idx != self.BOS_TAG_ID:
                end_frame_pre, label_idx_pre = pre_pos[end_frame_cur.long(), label_cur_idx.long()]
                if end_frame_pre != -1:
                    end_frame_list.append(int(end_frame_pre.item()))
                    label_idx_list.append(int(label_idx_pre.item()))
                end_frame_cur = end_frame_pre
                label_idx_cur = label_idx_pre


            
            end_frame_list.reverse()
            label_idx_list.reverse()

            segments_sample = []
            segment_start_frame = 0
            label_cur_idx = label_idx_list[0]
            for k in range(1, len(end_frame_list)):
                if label_idx_list[k] != label_cur_idx:
                    segment_end_frame = end_frame_list[k - 1]
                    segments_sample.append([label_cur_idx, segment_start_frame, segment_end_frame])
                    segment_start_frame = segment_end_frame + 1
                    label_cur_idx = label_idx_list[k]
            segment_end_frame = end_frame_list[-1]
            segments_sample.append([label_cur_idx, segment_start_frame, segment_end_frame])
            decoded_segments.append(segments_sample)
        return decoded_segments

    
    


