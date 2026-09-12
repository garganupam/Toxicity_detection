import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class Highway(nn.Module):
    def __init__(self, size, num_layers=1, activation_fn=F.relu):
        super().__init__()
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        for nonlinear_layer, gate_layer in zip(self.nonlinear, self.gate):
            gate = torch.sigmoid(gate_layer(x))
            nonlinear = self.activation_fn(nonlinear_layer(x))
            x = gate * nonlinear + (1 - gate) * x
        return x

# BERT Discriminator
class BertDiscriminator(nn.Module):
    def __init__(self, vocab_dict, id2word, num_classes, dropout_keep_prob=0.75):
        super(BertDiscriminator, self).__init__()
        self.vocab_dict = vocab_dict
        self.id2word = id2word
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_size = self.bert.config.hidden_size  # typically 768

        self.highway = Highway(self.hidden_size, num_layers=1)
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def decode_sentence(self, token_ids, id2word, start_id, eos_id):
        words = []
        for idx in token_ids:
            if idx == eos_id:
                break
            if idx == start_id:
                continue
            words.append(id2word.get(idx, '<UNK>'))
        return ' '.join(words)

    def trans_vocab(self, input_ids):
        decode_from_gen = [
            self.decode_sentence(seq.tolist(), self.id2word, self.vocab_dict['<START>'], self.vocab_dict['<EOS>']) 
            for seq in input_ids
        ]
        encode_to_bert = self.tokenizer(
            decode_from_gen, padding=True, truncation=True, max_length=32, return_tensors="pt"
        )

        device = next(self.parameters()).device
        input_ids = encode_to_bert["input_ids"].to(device)
        attention_mask = encode_to_bert["attention_mask"].to(device)

        return input_ids, attention_mask

    def forward(self, input_ids):
        input_ids, attention_mask = self.trans_vocab(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        h_highway = self.highway(cls_output)
        h_drop = self.dropout(h_highway)
        scores = self.fc(h_drop)
        ypred_for_auc = F.softmax(scores, dim=1)
        predictions = torch.argmax(scores, dim=1)

        return scores, ypred_for_auc, predictions
