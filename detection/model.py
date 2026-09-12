"""
Toxicity detection model — single definition, imported by all scripts.

Supports:
  - DistilBERT (default)
  - Any HuggingFace AutoModel-compatible model name
"""

import torch
import torch.nn as nn


class DistilBertClassifier(nn.Module):
    """DistilBERT with a binary classification head."""

    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
