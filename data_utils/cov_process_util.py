import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from data_utils.data_util import pad_and_truncate
import pandas as  pd

class CovProcess(Dataset):
    def __init__(self, data_path, tokenizer):
        cov_data = pd.read_csv(data_path, engine = 'python', encoding = 'utf-8')
        dim = cov_data.shape[0]
        all_data = []
        for line in range(dim):
            text = cov_data.at[line, 'text']
            time = cov_data.at[line, 'time']

            text_raw_indices = tokenizer.text_to_sequence(text)
            text_bert = '[CLS]{}[SEP]'.format(text)
            text_bert_indices = tokenizer.text_to_sequence(text_bert)
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
            

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'time': time,
            }
            all_data.append(data)
        
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)