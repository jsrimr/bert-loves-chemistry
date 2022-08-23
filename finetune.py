import argparse
from utils import reset_seeds, read_data_list, get_optimizer_params, get_scheduler, LoseMeter, MY_RMSELoss
from data_model import ChemiDataset
from utils import make_tokenizer, cacl_max_lenth
from config import ModelCfg, DATA_PATH, SUBMISSION_PATH

from transformers import AutoTokenizer, AutoConfig, AutoModel
import transformers
import tokenizers
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD, AdamW
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch
from scipy.stats import gmean
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

from model import ChemoModel
warnings.filterwarnings("ignore")

import wandb

# print('torch version : ' , torch.__version__)
# print('tokenizers version : ', tokenizers.__version__)
# print('transformers version : ', transformers.__version__)
# %env TOKENIZERS_PARALLELISM=true


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = LoseMeter()
    epoch_display = tqdm(enumerate(train_loader),
                         total=len(train_loader), leave=False)
    epoch_display.set_description(f'Epoch [{epoch+1}/{ModelCfg.epochs}]')
    for step, (inputs, labels) in epoch_display:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(inputs)
        y_preds = y_preds[::, 0]
        loss = criterion(y_preds, labels)
        wandb.log({"train_loss":loss.item()})

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        epoch_display.set_postfix(
            train_loss=losses.avg, lr=scheduler.get_lr()[0])

    return losses.avg

from inference import inference_fn
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--partial", action='store_true')
    # parser.add_argument("--sanity_check", action='store_true')
    parser.add_argument('wandb_tags', type=str,
                        nargs='*', help='tags for wandb')
    args = parser.parse_args()
    wandb.init(
        project="dacon-SMILE-challenge",
        notes="",
        tags=args.wandb_tags,
        config={k:getattr(ModelCfg,k) for k in dir(ModelCfg) if not k.startswith("__")},
    )
    # wandb.watch(finetune_model, log_freq=100)
    # use_wandb = True

    reset_seeds(ModelCfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train_loader = DataLoader(
        train_dataset, batch_size=ModelCfg.batch_size, shuffle=True, drop_last=True)

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
    # AdamW : https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html
    optimizer = AdamW(optimizer_parameters, lr=2e-5,
                      eps=1e-8, betas=(0.9, 0.999))

    # ====================================================
    # scheduler
    # ====================================================
    num_train_steps = int(
        len(x_train_) / ModelCfg.batch_size * ModelCfg.epochs)
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
        avg_train_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # scoring
        epoch_display = tqdm([0], leave=True)
        for _ in epoch_display:
            epoch_display.set_description(
                f'Epoch [{epoch+1}/{ModelCfg.epochs}]')

            score = avg_train_loss
            wandb.log({"score": score})

            if best_score > score:
                best_score = score
                wandb.log({"best_score":best_score})
                torch.save({'model': model.state_dict(
                )}, f"{ModelCfg.MODEL_PATH}{ModelCfg.MODEL_NAME.replace('/', '-')}_best.pth")

            elapsed = time.time() - start_time
            epoch_display.set_postfix(
                avg_train_loss=f'{avg_train_loss:.4f}', time=f'{elapsed/60:.0f}m', Score=f'{score:.4f}')

    # 예측
    print(f'INFERENCE [{ModelCfg.MODEL_NAME}] MODEL.............')
    test_dataset = ChemiDataset(ModelCfg, test, True)
    test_loader = DataLoader(test_dataset, batch_size=ModelCfg.batch_size, shuffle=False, drop_last=False)
    model = ChemoModel(config_path=None, pretrained=True)
    state = torch.load(f"{ModelCfg.MODEL_PATH}{ModelCfg.MODEL_NAME.replace('/', '-')}_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    # model = ChemoModel(ModelCfg, config_path=None, pretrained=False)
    prediction = inference_fn(test_loader, model, device)
    print(f'[{ModelCfg.MODEL_NAME}] predictions shape : ', np.array(prediction).shape)

    torch.cuda.empty_cache()
    gc.collect()
