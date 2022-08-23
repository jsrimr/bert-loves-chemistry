import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from tqdm import tqdm

# from config import MODEL_NAME, MODEL_PATH
from config import ModelCfg

def cacl_max_lenth(train_data):
    print(f'calculate max sequence length....[{ModelCfg.MODEL_NAME}]')
    for text_col in ['SMILES']:
        SMILES_lengths = []
        text_values = tqdm(train_data[text_col].fillna('').values, total=len(train_data))
        for text in text_values:
            length = len(ModelCfg.TOKENIZER(text, add_special_tokens=False)['input_ids'])
            SMILES_lengths.append(length)

    max_seq_length = max(SMILES_lengths) + 2
    print('SMILES max length : ', max(SMILES_lengths))
    print('final max data length : ', max_seq_length)

    return max_seq_length



def MY_RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2)) * 100


def make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(f'{ModelCfg.MODEL_NAME}', normalization=True, caeche_dir='./cache')
    tokenizer.save_pretrained(f'{ModelCfg.MODEL_PATH}')
    print(f'tokenizer object load & save......[{ModelCfg.MODEL_NAME}]')

    example = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    tokens = tokenizer(example) # example
    tokens_ = tokenizer.tokenize(example) # example
    print(tokens)
    print(tokens_)
    
    return tokenizer

