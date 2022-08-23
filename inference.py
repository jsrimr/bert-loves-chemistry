import numpy as np
import torch
import gc

from data_model import ChemiDataset
from config import ModelCfg
from model import ChemoModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import read_data_list, make_tokenizer, cacl_max_lenth

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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, test, sample_submission = read_data_list()

        # tokenizer 생성
    print('create tokenizer.............')
    tokenizer = make_tokenizer()
    ModelCfg.TOKENIZER = tokenizer
    # tokenizer에 입력하는 최대 길이 계산
    print('calculate max sequence length.............')
    max_seq_length = cacl_max_lenth(train)
    ModelCfg.MAX_SEQ_LENGTH = max_seq_length
    # 예측
    print(f'INFERENCE [{ModelCfg.MODEL_NAME}] MODEL.............')
    test_dataset = ChemiDataset(ModelCfg, test, True)
    test_loader = DataLoader(test_dataset, batch_size=ModelCfg.batch_size, shuffle=False, drop_last=False)
    model = ChemoModel(config_path=None, pretrained=True)
    state = torch.load(f"{ModelCfg.MODEL_PATH}{ModelCfg.MODEL_NAME.replace('/', '-')}_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    # model = ChemoModel(ModelCfg, config_path=None, pretrained=False)
    prediction = inference_fn(test_loader, model, device)
    # del model, state
    gc.collect()
    torch.cuda.empty_cache()

    print(f'[{ModelCfg.MODEL_NAME}] predictions shape : ', np.array(prediction).shape)