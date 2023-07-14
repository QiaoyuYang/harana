# The entry point of the training code

from harana.train import main
from harana.tools import *
from harana.models.semi_crf import SemiCRF
from harana.models.decoder import SemiCRFDecoder
from harana.tools import *

from harana.datasets.preprocessing import AllDatasets

import sys
import torch
import numpy as np

import shutil
import os
import tarfile
import math

def segment2frame(label_segment, batch_size, sample_size):
    
    label_frame = torch.zeros(batch_size, sample_size)
    for sample_index in range(batch_size):
        for label_index, f_start, f_end in label_segment[sample_index]:
            label_frame[sample_index, f_start : f_end + 1] = label_index
    
    return label_frame

def default_segment_score(batch_size, sample_size, num_label, label_segment):
    
    segment_score = torch.zeros(batch_size, sample_size, sample_size, num_label)
    for sample_index in range(batch_size):
        for label_index, f_start, f_end in label_segment[sample_index]:
            for i in range(f_start, f_end + 1):
                for j in range(i, f_end + 1):
                    segment_score[sample_index, i, j, label_index] = 1

    return segment_score

def default_note_embedding(batch_size, sample_size, embedding_size, label_segment):

    note_embedding = torch.zeros(batch_size, sample_size, embedding_size)
    for sample_index in range(batch_size):
        for label_index, f_start, f_end in label_segment[sample_index]:
            note_embedding[sample_index, f_start : f_end + 1, label_index] = 1

    return note_embedding

def default_harmony_embedding(batch_size, num_label, embedding_size):
    
    harmony_embedding = torch.zeros(batch_size, num_label, embedding_size)
    for i in range(num_label):
        harmony_embedding[:, i, i] = 1

    return harmony_embedding

if __name__ == "__main__":
    main(sys.argv[1:])

    '''
    #Toy Example parameters
    batch_size = 1
    sample_size = 3
    max_seg_len = 2
    num_label = 2

    label_gt_segment = [
        [[1, 0, 0], [0, 1, 2]],
        ]
    label_gt_frame = segment2frame(label_gt_segment, batch_size, sample_size)

    device = torch.device('cpu')
    transition_importance = 0
    with_transition=False
    transition_importance=1
    semicrf = SemiCRF(with_transition, transition_importance, batch_size, sample_size, max_seg_len, num_label, device)



    # Toy Example 1
    segment_score = default_segment_score(batch_size, sample_size, num_label, label_gt_segment)
    print(segment_score)
    semicrf.segment_score = segment_score
    path_score = semicrf.compute_path_score(label_gt_frame)
    print(path_score)
    log_z = semicrf.compute_log_z()
    print(log_z)
    
    segment_decode = semicrf.decode()
    print('segments decoded')
    for sample_index in range(batch_size):
        print(segment_decode[sample_index])
    '''


    '''
    # Toy Example 2
    embedding_size = num_label
    note_embedding = default_note_embedding(batch_size, sample_size, embedding_size, label_gt_segment)
    harmony_embedding = default_harmony_embedding(batch_size, num_label, embedding_size)

    harmony_type = 'toy'
    decoder = SemiCRFDecoder(harmony_type, batch_size, sample_size, max_seg_len, device, num_harmonies=num_label, embedding_size=embedding_size)
    
    path_score = semicrf.compute_path_score(label_gt_frame)

    loss = decoder.compute_loss(note_embedding, harmony_embedding, harmony_embedding, label_gt_frame)

    [frame_decode, segment_decode] = decoder.decode()
    print('segments decoded')
    for sample_index in range(batch_size):
        print(segment_decode[sample_index])
    '''

