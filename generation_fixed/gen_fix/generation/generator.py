import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import temporary_eval
from torch.autograd import Variable
import numpy as np

class Generator(nn.Module):
    """LSTM-based Generator for ToxicGAN"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, max_seq_len):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.normal_(0.0, 0.05)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return h, c

    def forward_step(self, input_token, hidden):
        emb = self.embedding(input_token)  # input_token: [B, 1]
        output, hidden = self.lstm(emb, hidden)
        logit = self.linear(output.squeeze(1))
        return logit, hidden

    def forward(self, input_seq, lengths):
        emb = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        h0, c0 = self.init_hidden(input_seq.size(0), input_seq.device)
        outputs, _ = self.lstm(packed, (h0, c0))
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=self.max_seq_len)
        logits = self.linear(unpacked)
        return logits

    def sample(self, batch_size, start_token, device):
        """Generate samples up to max_seq_len tokens"""
        samples = []
        input_token = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)  # [B, 1]
        h, c = self.init_hidden(batch_size, device)

        for _ in range(self.max_seq_len):
            logit, (h, c) = self.forward_step(input_token, (h, c))

            next_token = torch.multinomial(F.softmax(logit, dim=-1), 1)  # [B, 1]
            samples.append(next_token)
            input_token = next_token

        output_seq = torch.cat(samples, dim=1)  # [B, max_seq_len]
        return output_seq
