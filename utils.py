import os
import numpy as np
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm

# from config import MODEL_NAME, MODEL_PATH
from config import ModelCfg, DATA_PATH


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
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    return scheduler


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
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
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


def cacl_max_lenth(train_data):
    print(f'calculate max sequence length....[{ModelCfg.MODEL_NAME}]')
    for text_col in ['SMILES']:
        SMILES_lengths = []
        text_values = tqdm(train_data[text_col].fillna(
            '').values, total=len(train_data))
        for text in text_values:
            length = len(ModelCfg.TOKENIZER(
                text, add_special_tokens=False)['input_ids'])
            SMILES_lengths.append(length)

    max_seq_length = max(SMILES_lengths) + 2
    print('SMILES max length : ', max(SMILES_lengths))
    print('final max data length : ', max_seq_length)

    return max_seq_length


def MY_RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2)) * 100


def make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        f'{ModelCfg.MODEL_NAME}', normalization=True, caeche_dir='./cache')
    tokenizer.save_pretrained(f'{ModelCfg.MODEL_PATH}')
    print(f'tokenizer object load & save......[{ModelCfg.MODEL_NAME}]')

    example = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    tokens = tokenizer(example)  # example
    tokens_ = tokenizer.tokenize(example)  # example
    print(tokens)
    print(tokens_)

    return tokenizer


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
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
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


def MY_RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2)) * 100
