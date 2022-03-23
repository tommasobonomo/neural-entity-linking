import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Optional

import optuna
import pytorch_lightning as pl
import torch
import wandb
from google.cloud import storage as gcs
from names_generator import generate_name
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from scripts.company_embeddings import compute_and_write_company_embeddings
from scripts.predict import compute_and_write_predictions
from settings import ConfigSchema
from src.datamodules import MatchesDataModule, ModelType
from src.model import BiEncoder, CrossEncoder, NeuralMatcher, PolyEncoder


def upload_to_gcs(name: str, path: Path):
    client = gcs.Client()
    bucket = client.bucket("neural-matcher")
    blob = bucket.blob(name)
    blob.upload_from_filename(path.absolute().as_posix())


def train_fn(args, config: ConfigSchema, trial: Optional[optuna.Trial] = None):

    # Delete from args dictionary arguments that cannot be passed to pl.Trainer
    model_name = args.pop("model")
    pooling_strategy = config.pooling_strategy
    special_token_strategy = args.pop("special_token_strategy")
    orgs_to_embed = args.pop("orgs_to_embed")
    evaluate = args.pop("evaluate")

    if evaluate and not orgs_to_embed:
        raise RuntimeError("Cannot evaluate the model without a data path to organizations to embed (`--orgs_to_embed` parameter)")

    if model_name == "triplet":
        datamodule = MatchesDataModule(model_type=ModelType.bi_encoder)
        model = NeuralMatcher(frozen_transformer=False)
        test_metric = "cosine_similarity"
    elif model_name == "cross-encoder":
        datamodule = MatchesDataModule(model_type=ModelType.cross_encoder, special_token_strategy=special_token_strategy)
        new_vocabulary_size = len(datamodule.tokenizer)
        model = CrossEncoder(pooling_strategy=pooling_strategy, new_vocabulary_size=new_vocabulary_size)
        test_metric = None
    else:
        # model_name == "bi-encoder" or model_name == "poly-encoder":
        datamodule = MatchesDataModule(model_type=ModelType.bi_encoder, special_token_strategy=special_token_strategy,
                                       max_context_length=config.max_mention_length)
        new_vocabulary_size = len(datamodule.tokenizer)
        if model_name == "poly-encoder":
            model = PolyEncoder(
                transformer_model=config.transformer_model,
                learning_rate=config.default_learning_rate,
                use_vic_reg_loss=config.use_vic_reg_loss,
                metric_function=config.metric_function,
                lambda_=config.lambda_,
                mu=config.mu,
                nu=config.nu,
                temperature=config.temperature,
                pooling_strategy=pooling_strategy,
                new_vocabulary_size=new_vocabulary_size
            )
            test_metric = "euclidean_distance" if config.use_vic_reg_loss else "cosine_similarity"
        else:
            # model_name == "bi-encoder"
            model = BiEncoder(
                transformer_model=config.transformer_model,
                learning_rate=config.default_learning_rate,
                use_vic_reg_loss=config.use_vic_reg_loss,
                metric_function=config.metric_function,
                lambda_=config.lambda_,
                mu=config.mu,
                nu=config.nu,
                temperature=config.temperature,
                pooling_strategy=pooling_strategy,
                new_vocabulary_size=new_vocabulary_size
            )
            test_metric = "euclidean_distance" if config.use_vic_reg_loss else "cosine_similarity"

    datamodule.setup()

    model_dir = Path("models") / generate_name(style="hyphen")

    if not args.get("fast_dev_run"):
        datamodule.tokenizer.save_pretrained(model_dir / "tokenizer", legacy_format=False)
        model_paths = {
            "checkpoints": model_dir / "checkpoints",
            "embeddings": model_dir / "embeddings",
            "data": model_dir / "data"
        }
        for path in model_paths.values():
            path.mkdir()

        # Save data files
        (model_paths["data"] / "train.json").write_text(datamodule.train_dataset.file_path.read_text())
        (model_paths["data"] / "val.json").write_text(datamodule.val_dataset.file_path.read_text())
        (model_paths["data"] / "test.json").write_text(datamodule.test_dataset.file_path.read_text())
    else:
        model_paths = defaultdict(lambda: None)

    # Checkpointing
    checkpointing = ModelCheckpoint(
        monitor="val_mean_true_rank", save_top_k=1, dirpath=model_paths["checkpoints"], save_weights_only=True, save_on_train_epoch_end=False
    )

    # Early stopping
    early_stopping = EarlyStopping(monitor="val_mean_true_rank", mode="min", strict=False, patience=3)

    if trial is not None:
        # Hyperparam pruning
        hyperparm_pruning = optuna.integration.PyTorchLightningPruningCallback(trial, "val_mean_true_rank")
        callbacks = [checkpointing, early_stopping, hyperparm_pruning]
        # Trial - model connection
        print(f"Trial {trial.number} - model {model_dir.name}")
    else:
        callbacks = [checkpointing, early_stopping]

    # W&B logger
    wandb.finish()
    wandb_logger = WandbLogger(name=model_dir.name, project="bi-encoder", config=trial.params if trial else config, log_model=False)

    # Tensorboard logger
    tensorboard_logger = TensorBoardLogger("logs", name=model_dir.name)

    # DeepSpeed config
    deepspeed_conf = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": model.hparams.learning_rate
            }
        },
        # "scheduler": {
        #     "type": "WarmupDecayLR",
        #     "params": {
        #         "warmup_max_lr": config.default_learning_rate,
        #         "warmup_num_steps": 500,
        #         "total_num_steps": 10_000,
        #     }
        # },
        "zero_optimization": {
            "stage": 2,
            "cpu_offload": True,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        },
    }

    train_args = {
        **args,
        "gpus": config.gpus,
        "max_epochs": config.max_epochs,
        "callbacks": callbacks,
        "logger": [tensorboard_logger, wandb_logger],
        "log_every_n_steps": 25,
        "check_val_every_n_epoch": 2,
        "strategy": DeepSpeedPlugin(config=deepspeed_conf),
        "precision": 16
    }

    trainer = pl.Trainer(**train_args)

    trainer.fit(model, datamodule=datamodule)

    if not args.get("fast_dev_run"):
        # Rename best model to simply "model.ckpt"
        final_model = Path(checkpointing.best_model_path)
        if final_model.is_dir():
            # Sharded model parameters through deepspeed - gather into one checkpoint
            convert_zero_checkpoint_to_fp32_state_dict(final_model.as_posix(), (model_paths["checkpoints"] / "model.ckpt").as_posix())
        else:
            final_model.rename(model_paths["checkpoints"] / "model.ckpt")

        accuracy = None
        if orgs_to_embed:
            data_path = orgs_to_embed
            compute_and_write_company_embeddings(model, data_path, True, model_paths["embeddings"])

            if evaluate:
                accuracy = compute_and_write_predictions(
                    model, datamodule.test_dataloader(), True, test_metric, model_paths["embeddings"], chunking=False, cdist_on_gpu=False
                )

        # Upload all files in subdirs
        for model_subdir in model_paths.values():
            for file in model_subdir.iterdir():
                if not file.is_dir():
                    try:
                        upload_to_gcs(f"{model_dir.name}/{model_subdir.name}/{file.name}", file)
                    except Exception as e:
                        logging.warning(f"Exception caught when uploading to GCS: {e}")

        return accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", choices=["triplet", "bi-encoder", "cross-encoder", "poly-encoder"], required=True,
                        help="The model to train for this task. At the moment the two choices are between a triplet loss network and a bi-encoder.",)
    parser.add_argument("--orgs_to_embed", type=Path, required=False, help="Path to organizations that we want to compute the embeddings for.")
    parser.add_argument("--evaluate", action="store_true", required=False, help="If specified, will evaluate model on test set")
    parser.add_argument("--special_token_strategy", choices=["wu", "humeau"], required=False,
                        help="If special tokens should be introduced around entities and mentions like in Wu et al, or if no new special tokens should be "
                        + "introduced.")

    args = vars(parser.parse_args())
    train_fn(args, ConfigSchema())
