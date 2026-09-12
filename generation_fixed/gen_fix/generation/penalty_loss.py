import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class SemanticPenalty:
    def __init__(self, neutral_sentences, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SentenceTransformer(model_name).to(self.device)
        self.anchor_embeddings = self.encode_sentences(neutral_sentences)

    def encode_sentences(self, sentences):
        with torch.no_grad():
            embeddings = self.encoder.encode(sentences, convert_to_tensor=True, device=self.device)
        return embeddings

    def compute_penalty(self, generated_texts):
        with torch.no_grad():
            gen_embeddings = self.encoder.encode(generated_texts, convert_to_tensor=True, device=self.device)
            cosine_sim = F.cosine_similarity(
                gen_embeddings.unsqueeze(1),
                self.anchor_embeddings.unsqueeze(0),
                dim=-1
            )  # [B, N]
            max_sim = cosine_sim.max(dim=1)[0]  # take most similar anchor
        return max_sim