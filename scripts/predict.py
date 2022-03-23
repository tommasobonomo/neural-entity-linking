import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from settings import ConfigSchema
from src.datamodules import MatchesDataModule, ModelType
from src.model import NeuralMatcher, BiEncoder, BatchEncoding, PolyEncoder

logging.basicConfig(level=logging.INFO)

config = ConfigSchema()


def compute_and_write_predictions(
    model: Union[NeuralMatcher, BiEncoder],
    dataloader: torch.utils.data.DataLoader,
    use_gpu: bool,
    metric: str,
    embeddings_dir: Path,
    output_name: str = "all_orgs_evaluation.csv",
    chunking: Union[int, bool] = 1000,
    cdist_on_gpu: bool = True
):
    assert embeddings_dir.is_dir(), f"{embeddings_dir} is not a directory."
    assert metric in ["cosine_similarity", "euclidean_distance",
                      "dot_product"], "Metric can only be one of `cosine_similarity`, `euclidean_distance` or `dot_product`"

    company_embeddings = torch.load(embeddings_dir / "company_embeddings.pt")
    if use_gpu:
        company_embeddings = company_embeddings.cuda()

    with open(embeddings_dir / "company_ids.txt") as f:
        raw_ids = [int(line.strip()) for line in f.readlines()]

    company_ids = np.array(raw_ids)

    if chunking and int(chunking) < len(dataloader):
        len_chunks = int(chunking)
        first = True
    else:
        chunking = False

    i = 0
    truth = []
    raw_predictions = []
    rank_of_truth = []
    similarity_score = []
    for batch in tqdm(dataloader, desc="Extracting predictions..."):
        change_item_batch = batch[2]
        if use_gpu:
            change_item_batch = BatchEncoding({key: tensor.cuda() for key, tensor in change_item_batch.items()})

        if isinstance(model, BiEncoder):
            ci_embeds = model.embed_change_item(change_item_batch)

            if metric == "cosine_similarity":
                similarities = (ci_embeds @ company_embeddings.t()).cpu()
                ranks = torch.argsort(similarities, dim=1, descending=True)
                score = similarities.max(dim=1).values.numpy()
            elif metric == "euclidean_distance":
                distances = torch.cdist(ci_embeds, company_embeddings, p=2).cpu()
                ranks = torch.argsort(distances, dim=1, descending=False)
                score = distances.min(dim=1).values.numpy()
            else:
                raise ValueError(f"Invalid metric {metric} for BiEncoder model")
        elif isinstance(model, PolyEncoder):
            transformer_change_item_output = model.change_item_transformer(**change_item_batch).last_hidden_state
            ranks, scores = model.company_embedding_idx_for_change_item(
                transformer_change_item_output, company_embeddings.cpu(), change_item_batch.get("attention_mask"), metric, cdist_on_gpu
            )
            ranks = ranks.cpu().numpy()
            scores = scores.cpu().numpy()
            score = scores[:, 0]

        company_ranks = company_ids[ranks]

        # Add truth and prediction org_id
        truth.append(batch[0].numpy().squeeze())
        raw_predictions.append(company_ranks[:, 0])
        # Add score
        similarity_score.append(score)

        # Compute rank of truth. Will be -1 if not found
        truths_in_batch_rank = np.full_like(company_ranks[:, 0], -1)
        found_orgs_ranks = np.argwhere(company_ranks == batch[0].numpy())
        truths_in_batch_rank[found_orgs_ranks[:, 0]] = found_orgs_ranks[:, 1]
        rank_of_truth.append(truths_in_batch_rank)

        # Chunking
        i += 1
        if chunking and i > len_chunks:
            results = pd.DataFrame({
                "truth": np.concatenate(truth),
                "closest_rank": np.concatenate(raw_predictions),
                "similarity_score": np.concatenate(similarity_score),
                "rank_of_truth": np.concatenate(rank_of_truth),
            })

            if first:
                results.to_csv(embeddings_dir / output_name)
                first = False
            else:
                results.to_csv(embeddings_dir / output_name, mode="a", header=False)

            i = 0
            truth = []
            raw_predictions = []
            rank_of_truth = []
            similarity_score = []

    results = pd.DataFrame({
        "truth": np.concatenate(truth),
        "closest_rank": np.concatenate(raw_predictions),
        "similarity_score": np.concatenate(similarity_score) if any(score is not None for score in similarity_score) else pd.NA,
        "rank_of_truth": np.concatenate(rank_of_truth),
    })

    results.to_csv(embeddings_dir / output_name)

    return results["rank_of_truth"].mean()


def main(args: Namespace):
    logging.info("Loading model...")
    checkpoint_path = args.model_weights if args.model_weights else config.default_model_checkpoint
    if args.model == "triplet":
        model = NeuralMatcher.load_from_checkpoint(checkpoint_path)
        model_type = ModelType.bi_encoder
    elif args.model == "poly-encoder":
        model = PolyEncoder.load_from_checkpoint(checkpoint_path)
        model_type = ModelType.bi_encoder
    else:
        # args.model == "bi-encoder"
        model = BiEncoder.load_from_checkpoint(checkpoint_path)

    model = model.eval()

    if args.gpu:
        model = model.cuda()

    logging.info("Loading data...")
    dm = MatchesDataModule(data_path=args.dataset, model_type=model_type, batch_size=args.batch_size)
    dm.setup()

    compute_and_write_predictions(model, dm.dataloader(), args.gpu, args.metric, args.embeddings_folder,
                                  args.output, args.predictions_chunk, not args.cdist_not_on_gpu)

    logging.info("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output file to write the results of the prediction"
    )
    parser.add_argument(
        "--dataset", type=Path, help="Path to the file of companies to use to calculate the predictions."
    )
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for predictions. Default is low to compute euclidean distance on GPU wihtout CUDA OOM")
    parser.add_argument("--model", choices=["triplet", "bi-encoder", "poly-encoder"], default="bi-encoder",
                        help="The model to use for testing. At the moment the two choices are between a triplet loss network and a bi-encoder.")
    parser.add_argument("--metric", choices=["cosine_similarity", "euclidean_distance", "dot_product"], default="euclidean_distance",
                        help="The metric to use to compare change items embeddings with company embeddings")
    parser.add_argument("--model_weights", type=Path, default=None, help="Can be a path to a model's weights. If not specified will default to the default \
        checkpoint specified in settings.")
    parser.add_argument("--embeddings_folder", type=Path, required=True, help="Path to where the company embeddings and ids used \
        for prediction are stored.")
    parser.add_argument("--gpu", action="store_true", help="Whether to use GPU to compute predictions")
    parser.add_argument("--predictions_chunk", default=False, help="Number of steps of predictions to take before writing to disk the outcomes.")
    parser.add_argument("--cdist_not_on_gpu", action="store_true",
                        help="Compute the euclidean distance on CPU to avoid CUDA OOM errors. Not considered if metric is not `euclidean_distance`")
    args = parser.parse_args()

    main(args)
