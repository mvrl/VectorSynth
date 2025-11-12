import os
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel,
    BertTokenizer, BertModel,
    CLIPTokenizer, CLIPTextModel,
    T5Tokenizer, T5EncoderModel
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "osm_clip")))
from model import OSMBind


def average_pool(last_hidden_states, attention_mask):
    """Computes average pooling of hidden states, masking padding tokens."""
    masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_tokenizer_and_model(encoder_type='bert', checkpoint_path=None, taglist_path = None, tagvocab_path = None):
    if encoder_type == 'bert':
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        embedding_fn = lambda outputs, batch_dict: outputs.pooler_output.squeeze()

    elif encoder_type == 'clip':
        model_name = 'openai/clip-vit-large-patch14'
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPTextModel.from_pretrained(model_name)

        def clip_embedding_fn(outputs, batch_dict):
            input_ids = batch_dict['input_ids']
            eos_token_id = tokenizer.eos_token_id
            seq_lengths = (input_ids == eos_token_id).nonzero(as_tuple=True)[1]

            embeddings = []
            for i in range(input_ids.size(0)):
                eos_pos = seq_lengths[i] if i < len(seq_lengths) else (input_ids[i] != tokenizer.pad_token_id).sum() - 1
                embeddings.append(outputs.last_hidden_state[i, eos_pos, :])
            return torch.stack(embeddings)

        embedding_fn = clip_embedding_fn

    elif encoder_type == 'e5':
        model_name = 'intfloat/e5-base-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embedding_fn = lambda outputs, batch_dict: average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    elif encoder_type == 't5':
        model_name = 't5-base'
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name)
        embedding_fn = lambda outputs, batch_dict: average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    elif 'osm' in encoder_type:
        text_backbone = encoder_type.split('-')[1] if '-' in encoder_type else 'clip'
        model = OSMBind(taglist_path=taglist_path, tagvocab_path=tagvocab_path, text_backbone=text_backbone)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval().cuda()
        tokenizer = None

        def osm_embedding_fn(outputs, batch_dict):
            return model.text_encoder.encode_batch(batch_dict['sentences'])

        embedding_fn = osm_embedding_fn

    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    model.eval()
    return tokenizer, model, embedding_fn


def generate_embeddings(taglist_path, tag_vocab_path, output_path,
                                     encoder_type='bert', checkpoint_path=None):
    # Load taglist and vocab
    taglist = torch.load(taglist_path, weights_only = True)  # list of tuples of tag indices
    tag_vocab = torch.load(tag_vocab_path, weights_only = True)
    tag_index = {v: k for k, v in tag_vocab.items()}  # index -> tag string

    # Convert taglist tuples to "sentences" of tag strings
    sentences = []
    for tl in taglist:
        words = [tag_index[idx] for idx in tl]
        sentences.append(" ".join(words))

    # Optional prompt formatting
    if encoder_type == 'e5':
        sentences = [f"query: {s}" for s in sentences]
    elif encoder_type == 't5':
        sentences = [f"embedding: {s}" for s in sentences]

    # Load model
    tokenizer, model, embedding_fn = get_tokenizer_and_model(encoder_type, checkpoint_path, taglist_path = taglist_path, tagvocab_path = tag_vocab_path)
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')

    # Generate embeddings
    embeddings = []
    print("Encoding taglists...")
    for sentence in tqdm(sentences):
        if 'osm' in encoder_type:
            batch_dict = {'sentences': [sentence]}
            outputs = None
        else:
            inputs = tokenizer([sentence], return_tensors='pt', padding=True, truncation=True)
            batch_dict = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**batch_dict)

        with torch.inference_mode():
            emb = embedding_fn(outputs, batch_dict)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved {len(sentences)} taglist embeddings to {output_path}")


# ========================
# Command Line Interface
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for taglists")
    parser.add_argument("--taglist_path", type=str, required=True, help="Path to taglist_vocab.pt")
    parser.add_argument("--tag_vocab_path", type=str, required=True, help="Path to tag_vocab.pt")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save embeddings tensor")
    parser.add_argument("--encoder_type", type=str,
                        choices=["bert", "clip", "e5", "t5", "osm-clip", "osm-e5", "osm-bert"],
                        default="bert")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional checkpoint for OSMBind")

    args = parser.parse_args()

    generate_embeddings(
        taglist_path=args.taglist_path,
        tag_vocab_path=args.tag_vocab_path,
        output_path=args.output_path,
        encoder_type=args.encoder_type,
        checkpoint_path=args.checkpoint_path
    )