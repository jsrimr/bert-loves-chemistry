import numpy as np 
import pandas as pd
import random
import os
import dill
import matplotlib.pyplot as plt
import json
from ast import literal_eval
from tqdm.auto import tqdm

import re
import sys
import copy
import time
import math
import string
import pickle
import joblib
import itertools
import platform
import collections
import scipy as sp
import gc
import warnings

from model import ChemoModel; warnings.filterwarnings("ignore")
from scipy.stats import gmean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from config import ModelCfg, DATA_PATH, SUBMISSION_PATH
from utils import make_tokenizer, cacl_max_lenth
from data_model import ChemiDataset


# print('torch version : ' , torch.__version__)
# print('tokenizers version : ', tokenizers.__version__)
# print('transformers version : ', transformers.__version__)
# %env TOKENIZERS_PARALLELISM=true

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def read_data_list():
    train = pd.read_csv(f'{DATA_PATH}train_set.ReorgE.csv')
    test = pd.read_csv(f'{DATA_PATH}test_set.csv')
    sample_submission = pd.read_csv(f'{DATA_PATH}sample_submission.csv')
    
    print('train.shape: ', train.shape)
    print('test.shape: ', test.shape)
    print('sample_submission.shape: ', sample_submission.shape)

    return train, test, sample_submission


def get_optimizer_params(model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    return scheduler

class LoseMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = LoseMeter()
    epoch_display = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    epoch_display.set_description(f'Epoch [{epoch+1}/{ModelCfg.epochs}]')
    for step, (inputs, labels) in epoch_display:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(inputs)
        y_preds = y_preds[::, 0]
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        epoch_display.set_postfix(train_loss=losses.avg, lr=scheduler.get_lr()[0])

    return losses.avg

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    input_values = tqdm(test_loader, total=len(test_loader))
    for inputs, _ in input_values:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        y_preds = y_preds[::, 0]
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


def MY_RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2)) * 100



if __name__ == "__main__":

    reset_seeds(ModelCfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train, test, sample_submission = read_data_list()


    # tokenizer = make_tokenizer()
    # max_seq_length = cacl_max_lenth(train)

    # temp_train_dataset = ChemiDataset(tokenizer, max_seq_length, train)
    # temp_train_loader = DataLoader(temp_train_dataset, batch_size=1, shuffle=False, drop_last=True)

    # 데이타 읽기
    print('read data.............')
    train, test, sample_submission = read_data_list()

    # tokenizer 생성
    print('create tokenizer.............')
    tokenizer = make_tokenizer()
    ModelCfg.TOKENIZER = tokenizer

    # tokenizer에 입력하는 최대 길이 계산
    print('calculate max sequence length.............')
    max_seq_length = cacl_max_lenth(train)
    ModelCfg.MAX_SEQ_LENGTH = max_seq_length

    # 훈련
    print(f'TRAINING [{ModelCfg.MODEL_NAME}] MODEL.............')
    # train_loop(train)

    # ====================================================
    # data loader
    # ====================================================
    x_train_ = train.copy()
    train_dataset = ChemiDataset(ModelCfg, x_train_)
    train_loader = DataLoader(train_dataset, batch_size=ModelCfg.batch_size, shuffle=True, drop_last=True)

    # ====================================================
    # model
    # ====================================================
    model = ChemoModel(config_path=None, pretrained=True)
    torch.save(model.config, ModelCfg.MODEL_PATH + 'config.pth')
    model.to(device)

    # ====================================================
    # optimizer
    # ====================================================
    optimizer_parameters = get_optimizer_params(model)
    optimizer = AdamW(optimizer_parameters, lr=2e-5, eps=1e-8, betas=(0.9, 0.999))
    
    # ====================================================
    # scheduler
    # ====================================================
    num_train_steps = int(len(x_train_) / ModelCfg.batch_size * ModelCfg.epochs)
    scheduler = get_scheduler(ModelCfg, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = MY_RMSELoss
    # criterion = nn.MSELoss()
    best_score = np.inf
    for epoch in range(ModelCfg.epochs):
        start_time = time.time()

        # train
        avg_train_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # scoring
        epoch_display = tqdm([0], leave=True)
        for _ in epoch_display:
            epoch_display.set_description(f'Epoch [{epoch+1}/{ModelCfg.epochs}]')

            score = avg_train_loss
            if best_score > score:
                best_score = score
                torch.save({'model': model.state_dict()}, f"{ModelCfg.MODEL_PATH}{ModelCfg.MODEL_NAME.replace('/', '-')}_best.pth")

            elapsed = time.time() - start_time
            epoch_display.set_postfix(avg_train_loss=f'{avg_train_loss:.4f}', time=f'{elapsed/60:.0f}m', Score=f'{score:.4f}')

    torch.cuda.empty_cache()
    gc.collect()