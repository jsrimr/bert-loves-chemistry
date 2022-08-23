DATA_PATH = './data/'
SUBMISSION_PATH = './submit/'


class ModelCfg:
    # [hyperparams]
    epochs = 5
    seed = 42
    n_fold = 5
    batch_size = 24

    # [default]
    # 모델명 https://huggingface.co/models 참고
    MODEL_NAME = 'seyonec/PubChem10M_SMILES_BPE_180k'
    MAX_SEQ_LENGTH = 0
    TOKENIZER = None
    MODEL_PATH = './model/'
    CLEAN_DATA_PATH = './cleaned/'
    CONFIG_PATH = MODEL_PATH + 'model_config.pth'