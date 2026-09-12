import torch
import torch.nn as nn
from utils import temporary_eval
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from penalty_loss import SemanticPenalty


class Rollout:
    def __init__(self, generator, update_rate=0.8):
        self.generator = generator
        self.update_rate = update_rate
        
        self.lambda_penalty = 1
        self.penalty_module = None
        self.id2word = None
        
    def set_penalty_context(self, neutral_sentences, id2word):
        self.penalty_module = SemanticPenalty(neutral_sentences)
        self.id2word = id2word

    def get_penalty(self, samples, rollout_num, discriminator, start_token, device='cpu', current_class=0, penalty_type='dis'):
        batch_size, max_seq_len = samples.size()
        penalties = torch.zeros(batch_size, max_seq_len).to(device)
    
        with temporary_eval(self.generator):
            for i in range(rollout_num):
                for t in range(1, max_seq_len + 1):
                    current_tokens = samples[:, :t]
                    rollout_samples = []
                    for b in range(batch_size):
                        cur = current_tokens[b]
                        h, c = None, None
                        for step in range(t, max_seq_len):
                            inp = cur[-1].unsqueeze(0).unsqueeze(0)  # [1,1]
                            emb = self.generator.embedding(inp.to(device))
                            if h is not None:
                                out, (h, c) = self.generator.lstm(emb, (h, c))
                            else:
                                out, (h, c) = self.generator.lstm(emb)
                            logits = self.generator.linear(out.squeeze(1))
                            next_token = torch.multinomial(F.softmax(logits / 0.8, dim=-1), 1)
                            cur = torch.cat([cur, next_token.squeeze(1)], dim=0)
                        # pad to max length
                        if cur.size(0) < max_seq_len:
                            cur = F.pad(cur, (0, max_seq_len - cur.size(0)), value=0)
                        rollout_samples.append(cur)
                    
                    rollout_tensor = torch.stack(rollout_samples).to(device)
    
                    with torch.no_grad():
                        if penalty_type=='dis':
                            _, ypred, _ = discriminator(rollout_tensor)
                            D_i = 1-ypred[:, current_class]
                            penalty_score = D_i
                        elif penalty_type=='tox':
                            if self.penalty_module is not None and self.id2word is not None:
                                decoded = [
                                    ' '.join([
                                        self.id2word.get(idx.item(), '<UNK>')
                                        for idx in rollout_tensor[i]
                                        if idx.item() != 0 and self.id2word.get(idx.item(), '') != '<START>'
                                    ])
                                    for i in range(rollout_tensor.size(0))
                                ]
                                penalty_vals = self.penalty_module.compute_penalty(decoded).to(rollout_tensor.device)
                                penalty_score = penalty_vals

    
                    penalties[:, t - 1] += penalty_score
    
        penalties = penalties / rollout_num
        return penalties

    def update_params(self, new_generator):
        self.generator.load_state_dict(new_generator.state_dict())
        self.generator.lstm.flatten_parameters()