import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from settings import ConfigSchema
from src.datamodules import MatchesDataModule, ModelType
from src.model import NeuralMatcher, BiEncoder, PolyEncoder

logging.basicConfig(level=logging.INFO)

config = ConfigSchema()


def compute_and_write_company_embeddings(
    model: Union[NeuralMatcher, BiEncoder, PolyEncoder],
    data_path: Path,
    use_gpu: bool,
    output_dir: Path,
    batch_size: int = 40,
    model_type: ModelType = ModelType.bi_encoder
):
    assert data_path.exists(), f"{data_path} is not a valid JSON file!"
    assert output_dir.is_dir(), f"{output_dir} is not a directory or does not exist."

    logging.info("Loading data...")
    datamodule = MatchesDataModule(data_path=data_path, batch_size=batch_size, model_type=model_type)
    datamodule.setup()

    if use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    list_company_embeddings = []
    for batch in tqdm(datamodule.dataloader(), desc="Embedding companies..."):
        _, company, _ = batch
        if use_gpu:
            company = {key: tensor.cuda() for key, tensor in company.items()}
        embedded_company = model.embed_company(company).cpu()
        list_company_embeddings.append(embedded_company)
    company_embeddings = torch.cat(list_company_embeddings)
    torch.save(company_embeddings, output_dir / "company_embeddings.pt")

    company_ids = np.concatenate(list(map(
        lambda batch: batch[0].numpy().astype(str),
        tqdm(datamodule.dataloader(), desc="Storing org_ids...")
    )))
    with open(output_dir / "company_ids.txt", "w+") as f:
        f.writelines(company_id + "\n" for company_id in company_ids.squeeze())


def main(args: Namespace):

    logging.info(f"Loading {args.model} model...")
    checkpoint_path = args.model_weights if args.model_weights else config.default_model_checkpoint
    model_type = ModelType.bi_encoder
    if args.model == "triplet":
        model = NeuralMatcher.load_from_checkpoint(checkpoint_path)
    elif args.model == "cross-encoder":
        raise ValueError("Does not make sense to compute company embeddings of a cross-encoder!")
    elif args.model == "poly-encoder":
        model = PolyEncoder.load_from_checkpoint(checkpoint_path)
    else:
        model = BiEncoder.load_from_checkpoint(checkpoint_path)

    model = model.eval()

    output_dir = checkpoint_path.parent.parent / "embeddings"
    output_dir.mkdir(exist_ok=True)

    compute_and_write_company_embeddings(model, args.dataset, args.gpu, output_dir, model_type=model_type, batch_size=args.batch_size)
    logging.info("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract company embeddings from selected dataset")
    parser.add_argument("--model", choices=["triplet", "bi-encoder", "poly-encoder"], default="bi-encoder",
                        help="The model to train for this task. At the moment the two choices are between a triplet loss network and a bi-encoder.",)
    parser.add_argument("--gpu", action="store_true", help="Flag to indicate if GPUs should be used for inference")
    parser.add_argument("--dataset", type=Path, help="Path to the file of companies to use to calculate the embeddings.")
    parser.add_argument("--model_weights", type=Path, default=None, help="Can be a path to a model's weights. If not specified will default to the default \
        checkpoint specified in settings.")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size to use for inference on model")

    args = parser.parse_args()

    main(args)
