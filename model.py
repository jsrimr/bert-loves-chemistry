import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


from config import ModelCfg
class ChemoModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(ModelCfg.MODEL_NAME, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(ModelCfg.MODEL_NAME, config=self.config, cache_dir='./cache')
        else:
            self.model = AutoModel.from_config(self.config)

        self.fc_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output
