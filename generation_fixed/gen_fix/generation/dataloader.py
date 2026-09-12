import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import parse_opt
from torch.nn.functional import pad

class GenDataset(Dataset):
    def __init__(self, data_file_list):
        self.token_stream = []
        for data_file in data_file_list:
            with open(data_file, 'r') as f:
                for line in f:
                    tokens = [int(x) for x in line.strip().split()]
                    self.token_stream.append(tokens)

    def __len__(self):
        return len(self.token_stream)

    def __getitem__(self, idx):
        return torch.tensor(self.token_stream[idx], dtype=torch.long)

opt = parse_opt()

def pad_batch(batch, pad_token=0, max_len=opt.MAX_SEQ_LENGTH):
    return [torch.cat([item, torch.full((max_len - len(item),), pad_token)]) if len(item) < max_len else item[:max_len] for item in batch]

def gen_collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(pad_batch(batch), batch_first=True, padding_value=0)

class DisDataset(Dataset):
    def __init__(self, positive_file_list, negative_file_list, max_len):
        self.samples = []
        self.labels = []

        for pos_file in positive_file_list:
            with open(pos_file, 'r') as f:
                for line in f:
                    tokens = [int(x) for x in line.strip().split()][:max_len]
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))
                    self.labels.append(torch.tensor([0, 1], dtype=torch.float))

        for neg_file in negative_file_list:
            with open(neg_file, 'r') as f:
                for line in f:
                    tokens = [int(x) for x in line.strip().split()][:max_len]
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))
                    self.labels.append(torch.tensor([1, 0], dtype=torch.float))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def dis_collate_fn(batch):
    opt = parse_opt()
    max_len = opt.MAX_SEQ_LENGTH

    sentences, labels = zip(*batch)
    padded = [
        pad(s.cpu(), (0, max(0, max_len - len(s))), value=0)[:max_len]
        for s in sentences
    ]
    labels = torch.stack(labels).cpu()
    return torch.stack(padded), labels
