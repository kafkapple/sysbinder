import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# 1. TextProcessor class definition
class TextProcessor(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        # Adjust intermediate layer size based on embedding dimension
        intermediate_size = max(1024, embedding_dim * 2)  # 2x embedding dim or 1024, whichever is larger
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, text_embedding):
        return self.projection(text_embedding)

# 2. TextEmbeddingModel class definition
class TextEmbeddingModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        # Define embedding dimensions for each model
        self.embedding_dims = {
            'bert-base-uncased': 768,
            'bge-m3': 768,
            'bge-large': 1024,
            'bge-small': 384
        }
        
        if 'bge' in model_name:
            self.tokenizer = None
            self.model = SentenceTransformer(model_name).cuda()
            self.embedding_dim = self.embedding_dims[model_name]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).cuda()
            self.embedding_dim = self.embedding_dims[model_name]
    
    def encode(self, texts):
        if 'bge' in self.model_name:
            # Encoding for BGE models
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings
        else:
            # Encoding for BERT-based models
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)