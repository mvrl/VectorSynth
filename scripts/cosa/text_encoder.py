import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    BertTokenizer, BertModel, 
    CLIPTokenizer, CLIPTextModel
)
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from abc import ABC, abstractmethod
import random

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def taglist_index_to_sentence(taglist_vocab, tag_vocab, taglist_indices, subsample: bool = True):
    """
    Convert a tensor or list of taglist indices to a list of tag sentences.
    Optionally, randomly shuffle and sample a subset of tags for each sentence.
    
    Args:
        taglist_vocab: List of tuples of tag IDs.
        tag_vocab: Dictionary mapping tag ID to tag string.
        taglist_indices: Tensor or list of indices into taglist_vocab.
        seed: Random seed for reproducibility.
        subsample: If True, randomly subsample tags in each sentence.
    
    Returns:
        tag_sentences: List of strings (tag sentences).
    """
    if isinstance(taglist_indices, torch.Tensor):
        taglist_indices = taglist_indices.view(-1).tolist()

    tag_sentences = []
    
    for idx in taglist_indices:
        tag_ids = taglist_vocab[idx]
        tags = [tag_vocab[tid].lower().replace('=', ' ') for tid in tag_ids]
        
        if subsample and len(tags) > 1:
            n_sample = random.randint(1, len(tags))  # Choose how many tags to keep
            tags = random.sample(tags, n_sample)     # Sample without replacement

        random.shuffle(tags)  # Randomize order
        sentence = ' '.join(tags)
        tag_sentences.append(sentence)

    return tag_sentences


def average_pool(last_hidden_states, attention_mask):
    masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)


class BaseTextEncoder(nn.Module, ABC):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embedding_dim = None

    @abstractmethod
    def encode(self, sentences: List[str], device: str = 'cpu') -> torch.Tensor:
        """
        Encode a list of sentences into a tensor of embeddings.
        Must be implemented by subclasses.
        """
        pass

class BertTextEncoder(BaseTextEncoder):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, sentences, device='cpu'):
        self.model.to(device)
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        return self.model(**inputs).pooler_output


class CLIPTextEncoder(BaseTextEncoder):
    def __init__(self, model_name='openai/clip-vit-large-patch14'):
        super().__init__(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, sentences, device='cpu'):
        self.model.to(device)
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = inputs['input_ids']
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, T, D]

        batch_size = input_ids.size(0)
        embeddings = []

        for i in range(batch_size):
            input_seq = input_ids[i]
            eos_positions = (input_seq == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                eos_idx = eos_positions[-1]  # take last EOS (safe for duplicates)
            else:
                eos_idx = (input_seq != pad_token_id).sum() - 1  # fallback to last non-padding token

            embeddings.append(last_hidden[i, eos_idx, :])

        return torch.stack(embeddings)

class E5TextEncoder(BaseTextEncoder):
    def __init__(self, model_name='intfloat/e5-base'):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.pooler = None
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, sentences, device='cpu'):
        self.model.to(device)
        sentences = [f"query: {s}" for s in sentences]  # official prompt for e5 (for features as per documentation)
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.model(**inputs)
        return average_pool(outputs.last_hidden_state, inputs['attention_mask'])

class GritLMTextEncoder(BaseTextEncoder):
    def __init__(self, model_name='nomic-ai/nomic-bert-base-punc'):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size
        self.proj_head = nn.Linear(self.embedding_dim, 768) # to match other encoders

    def encode(self, sentences, device='cpu'):
        self.model.to(device)
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.model(**inputs)
        pooled = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return self.proj_head(pooled)


class TextEncoder(pl.LightningModule):
    def __init__(self, taglist_vocab: List[tuple], tag_vocab: dict, model_name='bert'):
        super().__init__()
        self.taglist_vocab = taglist_vocab
        self.tag_vocab = tag_vocab

        model_name = model_name.lower()
        encoder_map = {
            'bert': lambda: BertTextEncoder('bert-base-uncased'),
            'clip': lambda: CLIPTextEncoder('openai/clip-vit-large-patch14'),
            'e5': lambda: E5TextEncoder('intfloat/e5-base'),
            'gritlm': lambda: GritLMTextEncoder('nomic-ai/nomic-bert-base-punc')
        }

        if model_name not in encoder_map:
            raise ValueError(f"Unsupported model_name: {model_name}. Choose from {list(encoder_map.keys())}")
        print(f"Text backbone: {model_name}")
        self.encoder = encoder_map[model_name]()  # Instantiate the selected encoder
        # self.embedding_dim = 768

    def forward(self, taglist_tensor: torch.Tensor) -> torch.Tensor:
        tag_indices = taglist_tensor.tolist()
        tag_sentences = taglist_index_to_sentence(self.taglist_vocab, self.tag_vocab, tag_indices, subsample=True) # randomize subsampling tags
        embeddings = self.encoder.encode(tag_sentences, device=self.device)
        return embeddings

    def encode_raw_text(self, raw_text: str) -> torch.Tensor:
        """
        Encode a single raw string into an embedding for queries
        """
        return self.encoder.encode([raw_text], device=self.device)[0] 
