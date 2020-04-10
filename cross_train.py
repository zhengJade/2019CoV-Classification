# -*- coding: utf-8 -*-
# file: train.py
# author: jade <zhengjade@qq.com>
# Copyright (C) 2020. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np

from pytorch_transformers import BertModel
from loss_function import focal_loss

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from data_utils import CovData

from models import SL_BERT, BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class ModelTrained(nn.Module):
    def __init__(self, opt, model_path: str):
        super(ModelTrained, self).__init__()
        self.opt = opt
        self.model_path = model_path
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.model.load_state_dict(torch.load(model_path))

    def output(self, inputs):
        return self.model(inputs)

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='weibo')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=5, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--cross_fold', default=4, type=int, help='交叉验证次数')
    parser.add_argument('model_path', default='model', type=str, help='save model name')
    parser.
    opt = parser.parse_args()

    model_classes = {
        'bert_spc': BERT_SPC,
        'sl_bert': SL_BERT,
    }

    input_colses = {
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'lsat_bert': ['text_bert_indices', 'bert_segments_ids'],
        'sl_bert': ['text_bert_indices', 'bert_segments_ids'],
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    dataset_files = {
        'weibo':{
            'train': './datasets/2019CoV/CoV_train.csv',
            'test': './datasets/2019CoV/CoVTest.csv'
        }
    }

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.data_file = dataset_files[opt.dataset]
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    testset = CovData(opt.dataset_file['test'], tokenizer)
    test_data_loader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False)
    
    model_list = []
    for fold in range(opt.cross_fold):
        model = ModelTrained(opt, './state_dict/{}{}'.format(opt.model_path, fold))
        #model._reset_params
        model_list.append(model)

    n_correct, n_total = 0, 0
    targets_all, outputs_all = None, None
    with torch.no_grad():
        logger.info('>' * 100)
        for batch, sample_batched in enumerate(test_data_loader):
            if batch % 100 == 0:
                logger.info('batch: {}'.format(batch))
            inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
            targets = sample_batched['polarity'].to(opt.device)
            result_list = []
            for model in model_list:
                result_list.append(model.output(inputs))
                
            outputs = sum(result_list)

            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            acc = n_correct / n_total
            if batch % 10 == 0:
                logger.info('acc: {:.4f}'.format(acc))

            if targets_all is None:
                targets_all = targets
                outputs_all = outputs
            else:
                targets_all = torch.cat((targets_all, targets), dim=0)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        logger.info('acc: {:.4f} f1: {:.4f}'.format(train, f1))
if __name__ == '__main__':
  main()
