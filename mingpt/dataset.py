import json

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class JSONLDataset(Dataset):
    def __init__(self, file_path, split='train', test_size=1000):
        self.file_path = file_path
        self.tokenizer = Tokenizer.from_file("gpt2_tokenizer.json")

        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        print(f'Length of lines: {len(lines)}')
        prelength = len(lines)
        data = []
        for i, line in enumerate(lines):
            try:
                # Processes the line and extracts the 'text' field
                d = json.loads(line)['text']
                # Skips short text entries
                if len(d) < 10:
                    continue
                data.append(d)
            except json.JSONDecodeError:
                # Silently skip lines that fail to parse
                pass 
        print(f'Length of data: {len(data) / prelength * 100:.2f}%')

        # calulate the vocab size
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.block_size = 512  # or any other fixed size you want
        
        # Implements the train/test split
        if split == 'train':
            self.data = data[:-test_size]
        else:
            self.data = data[-test_size:]
    def __len__(self):
        # Returns the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Returns the raw text at the specified index
        # Tokenize the text here as tensors
        text = self.data[idx]
        tokens = self.tokenizer.encode(text).ids

        # truncate tokens if too long
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            # pad tokens if too short
            tokens = tokens + [0] * (self.block_size - len(tokens))

        # return x, y where y is the next token prediction
        return torch.tensor(tokens), torch.tensor(tokens)

    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        return self.block_size

file_path = '/nobackup/autodelete/usr/rsinema/pile_data_10_min.jsonl'
# file_path = 'pile_data_10_first_50000.jsonl'
# file_path = '100.jsonl'
# Initialize the dataset with a test size of 1000 lines
train_dataset = JSONLDataset(file_path, split='train', test_size=10)
print(len(train_dataset))
print(train_dataset[0])