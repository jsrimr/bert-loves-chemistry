import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def prepare_input(tokenizer, max_seq_length, smiles):
    inputs = tokenizer(smiles, add_special_tokens=True, truncation=True, max_length=max_seq_length, padding="max_length", return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

def create_label(Reorg_g, Reorg_ex):
    return torch.tensor([Reorg_g, Reorg_ex], dtype=torch.float)

class ChemiDataset(Dataset):
    def __init__(self, cfg, df, eval_mode=False):
        self.cfg = cfg
        # self.tokenizer = tokenizer
        # self.max_seq_length = max_seq_length
        self.eval_mode = eval_mode
        self.smiles = df['SMILES'].values
        if not self.eval_mode:
            self.Reorg_g = df['Reorg_g'].values
            self.Reorg_ex = df['Reorg_ex'].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg.TOKENIZER, self.cfg.MAX_SEQ_LENGTH, self.smiles[item])
        label = np.array([])
        if not self.eval_mode:
            label = create_label(self.Reorg_g[item], self.Reorg_ex[item])
        return inputs, label