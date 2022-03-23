from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoTokenizer, BatchEncoding

from src.datamodules import ModelType
from src.model import BiEncoder


def generate_inference_function(
    model_dir: Path,
    model_type: ModelType
) -> Callable[[str], int]:

    def predict(
        change_item_str: str,
        model: BiEncoder,
        tokenize_fn: Callable[[str], BatchEncoding],
        company_embeddings: torch.Tensor,
        company_ids: np.ndarray,
        metric: str
    ):
        encoded_change_item = tokenize_fn(change_item_str)
        embedded_change_item = model.embed_change_item(encoded_change_item)

        if metric == "euclidean_distance":
            distances = torch.cdist(embedded_change_item, company_embeddings, p=2)
            closest_idx = torch.argmin(distances, dim=1)

        closest_org_id = company_ids[closest_idx]
        return closest_org_id

    # Load model according to model type
    if model_type == ModelType.cross_encoder:
        model = CrossEncoder.load_from_checkpoint(model_dir / "checkpoints" / "model.ckpt").eval()
    else:
        # model_type == ModelType.bi_encoder:
        model = BiEncoder.load_from_checkpoint(model_dir / "checkpoints" / "model.ckpt").eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    tokenize_fn = partial(tokenizer, padding="max_length", truncation=True, return_tensors="pt")

    # Load company embeddings and ids
    company_embeddings = torch.load(model_dir / "embeddings" / "company_embeddings.pt")
    with open(model_dir / "embeddings" / "company_ids.txt") as f:
        raw_ids = [int(line.strip()) for line in f.readlines()]
    company_ids = np.array(raw_ids)

    return partial(predict, model=model, tokenize_fn=tokenize_fn, company_embeddings=company_embeddings,
                   company_ids=company_ids, metric="euclidean_distance")
