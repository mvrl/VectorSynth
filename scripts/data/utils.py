import json
import ast
import torch
import os
import numpy as np

def process_osm_tags(tags_list):
    cleaned = []

    allowed_boundary_tags = {
        "boundary protected_area",
        "boundary parking",
        "boundary historic",
        "boundary national_park"
    }

    tags_to_remove = {
        "building colour",
        "building proposed",
        "height -1m",
        'place municipality', 
        'place locality', 
        'place island', 
        'place county', 
        'place region', 
        'place quarter', 
    }

    for tag_pair in tags_list:
        if not isinstance(tag_pair, (list, np.ndarray)):
            continue
        if len(tag_pair) == 0:
            continue

        key = tag_pair[0]
        val = tag_pair[1] if len(tag_pair) > 1 else None

        if key is None:
            continue

        tag_string = f"{key} {val}" if val else key

        # Boundary filtering
        if tag_string.startswith("boundary ") and tag_string not in allowed_boundary_tags:
            continue

        if tag_string in tags_to_remove:
            continue

        # Remove tags ending with _no
        if tag_string.endswith(" no"):
            continue

        # Rename keys ending with _yes by stripping suffix
        if tag_string.endswith(" yes"):
            new_tag_string = tag_string[:-4]
            cleaned.append(new_tag_string)
            continue

        # Rename craft → workplace
        if key == "craft":
            new_tag_string = f"workplace {val}" if val else "workplace"
            cleaned.append(new_tag_string)
            continue

        # Default: keep as is
        cleaned.append(tag_string)

    # Final cleanup: remove None, empty, or whitespace-only strings
    cleaned = [t for t in cleaned if isinstance(t, str) and t.strip()]

    # Replace underscores with spaces
    cleaned = [t.replace("_", " ") for t in cleaned]
    
    return cleaned

def load_tag_data(tag_list_path, tag_vocab_path):
    def load_file(path):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif ext == '.pt':
            return torch.load(path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    tag_list_index = load_file(tag_list_path)
    tag_vocab = load_file(tag_vocab_path)

    if type(tag_list_index) is not dict:
        inverted_tag_list_index = {i: tag for i, tag in enumerate(tag_list_index)}
    else:
        inverted_tag_list_index = {v: k for k, v in tag_list_index.items()}

    inverted_tag_vocab = {v: k for k, v in tag_vocab.items()}

    return inverted_tag_list_index, inverted_tag_vocab

def generate_cleaned_tag_descriptions(inverted_tag_list_index, inverted_tag_vocab):
    cleaned_results = {}
    for idx, tag_list in inverted_tag_list_index.items():

        if isinstance(tag_list, str):
            tag_list_val = ast.literal_eval(tag_list)
        else:
            tag_list_val = tag_list

        tags = [inverted_tag_vocab.get(i) for i in tag_list_val]
        # cleaned_tags = process_osm_tags(tags)
        cleaned_results[idx] = tags
    return cleaned_results