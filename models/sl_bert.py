# -*- coding: utf-8 -*-
# file: sl_bert.py
# author: Jade <jadezheng@qq.com>
# Copyright (C) 2019. All Rights Reserved.


import torch
import torch.nn as nn
import copy
import numpy as np

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class SL_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SL_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert   # Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)


    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]

        bert_spc_out, _ = self.bert_spc(text_bert_indices, bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)

        mean_pool = self.linear_single(bert_spc_out)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(self_attention_out)

        return dense_out
