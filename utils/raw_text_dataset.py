from torch.utils.data import Dataset
from nlp import load_dataset

class RawTextDataset(Dataset):
    """
    Custom Torch Dataset for tokenizing large (up to 100,000,000+ sequences) text corpuses,
    by not loading the entire dataset into cache and using lazy loading from disk (using huggingface's
    'NLP' library. See 'https://github.com/huggingface/nlp' for more details on the NLP package.
    Examples
    --------
    >>> from raw_text_dataset import RawTextDataset
    >>> dataset = RawTextDataset(tokenizer=tokenizer, file_path="shard_00_selfies.txt", block_size=512)
    Downloading: 100%
    1.52k/1.52k [00:03<00:00, 447B/s]
    Using custom data configuration default
    Downloading and preparing dataset text/default-f719ef2eb3ab586b (download: Unknown size, generated: Unknown size, post-processed: Unknown sizetotal: Unknown size) to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b...
    Dataset text downloaded and prepared to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b. Subsequent calls will reuse this data.
    Loaded Dataset
    Number of lines: 999988
    Block size: 512
    """
    def __init__(self, tokenizer, file_path: str, block_size: int):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        self.dataset = load_dataset("text", data_files=file_path)["train"]
        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def preprocess(self, text):
        batch_encoding = self.tokenizer(str(text), add_special_tokens=True, truncation=True, max_length=self.block_size)
        return torch.tensor(batch_encoding["input_ids"])

    def __getitem__(self, i):
        line = self.dataset[i]
        example = self.preprocess(line)
        return example
