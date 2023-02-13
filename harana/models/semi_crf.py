# my import
from .. import tools

# regular import
import numpy as np
import torch
from torch import nn

class SemiCRF(nn.Module):

    def __init__(self, batch_size, sample_size, max_seg_len, num_label, device, transition_importance, with_transition):
        super().__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.max_seg_len = max_seg_len
        self.num_label = num_label
        self.transition_importance = transition_importance
        self.with_transition = with_transition
        
        self.device = device

        self.segment_score = torch.zeros(self.batch_size, self.sample_size, self.sample_size, self.num_label)
        self.segment_score = self.segment_score.to(self.device)

        # the last number in the label index list is BOS tag.
        #self.BOS_TAG_ID = num_label + 1

        self.transitions = nn.Parameter(torch.empty(self.num_label, self.num_label))
        self.init_weights()

    def init_weights(self):
        # initialize weights between -0.1 and 0.1
        nn.init.uniform_(self.transitions, 0, 1)


    def compute_path_score(self, label_gt):
        path_score = torch.zeros(self.batch_size)
        path_score = path_score.to(self.device)


        for sample_index in range(self.batch_size):
            
            label_pre = int(label_gt[sample_index, 0].item())
            f_start = 0
            for f_i in range(1, self.sample_size):
                if label_gt[sample_index, f_i] != label_pre:
                    label_cur = int(label_gt[sample_index, f_i].item())
                    f_end = f_i - 1
                    segment_score_pre = self.segment_score[sample_index, f_start, f_end, label_pre] * (f_end - f_start + 1)
                    path_score[sample_index] += segment_score_pre
                    if self.with_transition:
                        transition_score_cur = self.transitions[label_pre, label_pre] * (f_end - f_start) + self.transitions[label_pre, label_cur]
                        path_score[sample_index] += transition_score_cur * self.transition_importance
                    f_start = f_i
                    label_pre = label_cur
            
            f_end = f_i
            transition_score_cur = self.transitions[label_pre, label_pre] * (f_end - f_start)
            segment_score_cur = self.segment_score[sample_index, f_start, f_end, label_pre] * (f_end - f_start + 1)
            path_score[sample_index] += transition_score_cur * self.transition_importance + segment_score_cur
        
        return path_score

    def compute_log_z(self):
        log_alpha = torch.zeros(self.batch_size, self.sample_size, self.num_label)
        log_alpha = log_alpha.to(self.device)

        # There is no transition for the single-frame segment at the begining of the sample
        # There is only one possible assignment for each current label so logsumexp is equivalent to identity
        log_alpha[:, 0, :] = self.segment_score[:, 0, 0, :]
        
        for f_end in range(1, self.sample_size):
            max_span = min(f_end + 1, self.max_seg_len)

            # shape: batch_size, span, num_label_cur
            segment_score = torch.stack([self.segment_score[:, f_end - j + 1, f_end, :] * j for j in range(max_span, 0, -1)], dim=1)

            # shape: span, num_label_pre, num_label_cur
            transition_score = torch.stack([(self.transitions[:, :] + self.transitions.diag()[None, :] * (j - 1)) for j in range(max_span, 0, -1)], dim=0)
            
            if f_end < self.max_seg_len:
                # shape: num_label_cur
                transition_score_init = self.transitions.diag() * f_end
                
                # shape: batch_size, num_label_cur
                feature_score_init = segment_score[:, 0, :]

                # shape: batch_size, max_span - 1, num_label_pre, num_label_cur
                feature_score = segment_score[:, 1:, None, :]

                if self.with_transition:
                    # shape: batch_size, num_label_cur
                    feature_score_init += transition_score_init[None, :]

                    # shape: batch_size, max_span - 1, num_label_pre, num_label_cur
                    feature_score += transition_score[None, 1:, :, :]

                # shape: batch_size, max_span - 1, num_label_pre
                log_alpha_pre = log_alpha[:, f_end - max_span + 1 : f_end, :]

            else:
                # shape: batch_size, max_span, num_label_pre, num_label_cur
                feature_score = segment_score[:, :, None, :]
                
                if self.with_transition:
                    # shape: batch_size, max_span, num_label_pre, num_label_cur
                    feature_score += transition_score[None, :, :, :]

                # shape: batch_size, max_span, num_label_pre
                log_alpha_pre = log_alpha[:, f_end - max_span: f_end, :]
            
            # shape: batch_size, max_span * num_label_pre, num_label_cur
            temp = (log_alpha_pre[:, :, :, None] + feature_score).flatten(start_dim=1, end_dim=2)

            if f_end < self.max_seg_len:
                log_alpha[:, f_end, :] = torch.cat([feature_score_init[:, None, :], temp], dim=1).logsumexp(dim=1)
            else:
                log_alpha[:, f_end, :] = temp.logsumexp(dim=1)
        
        log_z = log_alpha[:, self.sample_size - 1, :].logsumexp(dim=1)

        return log_z

    def decode(self):
        print(f"Decoding the segments...")
       
        decoded_segments = list()
        V = torch.zeros(self.batch_size, self.sample_size, self.num_label)
        V = V.to(self.device)
        pos_pre = torch.zeros(self.batch_size, self.sample_size, self.num_label, 2)
        pos_pre = pos_pre.to(self.device)
        
        # There is no transition for the single-frame segment at the begining of the sample
        # There is only one possible assignment for each current label
        V[:, 0, :] = self.segment_score[:, 0, 0, :]
        pos_pre[:, 0, :, :] = torch.tensor([-1, -1])[None, None, :]

        for f_end in range(1, self.sample_size):
            #print(f'f_end: {f_end}')
            max_span = min(f_end + 1, self.max_seg_len)

            # shape: batch_size, span, num_label_cur
            segment_score = torch.stack([self.segment_score[:, f_end - j + 1, f_end, :] * j for j in range(max_span, 0, -1)], dim=1)

            # shape: span, num_label_pre, num_label_cur
            transition_score = torch.stack([(self.transitions[:, :] + self.transitions.diag()[None, :] * (j - 1)) for j in range(max_span, 0, -1)], dim=0)

            if f_end < self.max_seg_len:
                # shape: num_label_cur
                transition_score_init = self.transitions.diag() * f_end
                
                # shape: batch_size, num_label_cur
                feature_score_init = segment_score[:, 0, :]

                # shape: batch_size, max_span - 1, num_label_pre, num_label_cur
                feature_score = segment_score[:, 1:, None, :]

                if self.with_transition:
                    # shape: batch_size, num_label_cur
                    feature_score_init += transition_score_init[None, :]

                    # shape: batch_size, max_span - 1, num_label_pre, num_label_cur
                    feature_score += transition_score[None, 1:, :, :]

                # shape: batch_size, max_span - 1, num_label_pre

                V_pre = V[:, f_end - max_span + 1: f_end, :]

            else:
                # shape: batch_size, max_span, num_label_pre, num_label_cur
                feature_score = segment_score[:, :, None, :]
                
                if self.with_transition:
                    # shape: batch_size, max_span, num_label_pre, num_label_cur
                    feature_score += transition_score[None, :, :, :]

                # shape: batch_size, max_span, num_label_pre
                V_pre = V[:, f_end - max_span: f_end, :]

            '''
            print('V_pre')
            print(V_pre[0, ...])
            print('feature_score')
            print(feature_score[0, ...])
            '''

            # shape: batch_size, max_span * num_label_pre, num_label_cur
            temp = (V_pre[:, :, :, None] + feature_score).flatten(start_dim=1, end_dim=2)

            '''
            print('temp')
            print(temp[0, ...])
            '''


            V[:, f_end, :], max_index = temp.max(dim=1)
            pos_pre[:, f_end, :, :] = torch.stack([f_end - max_span + (max_index / self.num_label).floor(), (max_index % self.num_label)], dim=-1)

            if f_end < self.max_seg_len:
                for sample_index in range(self.batch_size):
                    for label_cur in range(self.num_label):
                        if V[sample_index, f_end, label_cur] <= feature_score_init[sample_index, label_cur]:
                            V[sample_index, f_end, label_cur] = feature_score_init[sample_index, label_cur]
                            pos_pre[sample_index, f_end, label_cur, :] = torch.tensor([-1, -1])

        for sample_index in range(self.batch_size):
            
            segment_offsets = list()
            offset_cur = torch.tensor(self.sample_size - 1)
            segment_offsets.append(int(offset_cur.item()))
            
            labels = list()
            label_cur = torch.argmax(V[sample_index, offset_cur, :])
            labels.append(int(label_cur.item()))
            
            while offset_cur != -1:
                offset_pre, label_pre = pos_pre[sample_index, offset_cur.long(), label_cur.long(), :]
                if offset_pre != -1:
                    segment_offsets.append(int(offset_pre.item()))
                    labels.append(int(label_pre.item()))
                offset_cur, label_cur = offset_pre, label_pre

            segment_offsets.reverse()
            labels.reverse()
            '''
            print(segment_offsets)
            print(labels)
            '''

            segments_sample = list()
            onset_cur = 0
            label_cur = labels[0]
            for i in range(1, len(segment_offsets)):
                if labels[i] != label_cur:
                    offset_cur = segment_offsets[i - 1]
                    segments_sample.append([label_cur, onset_cur, offset_cur])
                    onset_cur = offset_cur + 1
                    label_cur = labels[i]
            
            offset_cur = segment_offsets[-1]
            segments_sample.append([label_cur, onset_cur, offset_cur])
            decoded_segments.append(segments_sample)

        return decoded_segments

    
    


