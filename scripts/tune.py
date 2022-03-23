from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch

from settings import ConfigSchema
from scripts.train import train_fn


def objective(trial: optuna.trial.Trial, args: dict) -> float:
    torch.cuda.empty_cache()
    # Params
    # trial.suggest_categorical("transformer_model", ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/LaBSE"])
    # transformer_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    transformer_model = "sentence-transformers/LaBSE"
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-3)
    loss_fn = "vic_reg"  # trial.suggest_categorical("loss_fn", ["vic_reg", "info_nce"])
    distance_fn = "euclidean"  # trial.suggest_categorical("distance_fn", ["cosine", "euclidean"])
    lambda_ = 25  # trial.suggest_int("lambda_", 1, 40)
    mu = 14  # trial.suggest_int("mu", 1, 40)
    nu = 15  # trial.suggest_int("nu", 1, 40)
    temperature = 0.06  # trial.suggest_loguniform("temperature", 1e-5, 1)
    max_mention_length = 350  # trial.suggest_int("max_mention_length", 32, 512)
    add_linear_projector = "no"  # trial.suggest_categorical("add_linear_projector", ["yes", "no"])
    final_embedding_size = 0  # trial.suggest_int("final_embedding_size", low=100, high=1000, step=100)
    projector_number_layers = 0  # trial.suggest_int("projector_number_layers", low=2, high=5)
    hidden_layer_size: int = 0  # trial.suggest_int("hidden_layer_size", low=100, high=1000, step=100)
    pooling_strategy = "mean_masked_pooling"  # trial.suggest_categorical("pooling_strategy", ["pooler_output", "mean_masked_pooling"])
    num_polyencoder_query_vectors = trial.suggest_int("num_polyencoder_query_vectors", low=10, high=1000)

    config = ConfigSchema(
        transformer_model=transformer_model,
        default_learning_rate=learning_rate,
        use_vic_reg_loss=bool(loss_fn == "vic_reg"),
        metric_function=distance_fn,
        lambda_=lambda_,
        mu=mu,
        nu=nu,
        temperature=temperature,
        max_mention_length=max_mention_length,
        add_linear_projection=add_linear_projector == "yes",
        final_embedding_size=final_embedding_size,
        projector_number_layers=projector_number_layers,
        hidden_layer_size=hidden_layer_size,
        pooling_strategy=pooling_strategy,
        num_polyencoder_query_vectors=num_polyencoder_query_vectors
    )

    return train_fn(args.copy(), config, trial)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", choices=["triplet", "bi-encoder", "cross-encoder", "poly-encoder"], required=True,
                        help="The model to train for this task. At the moment the two choices are between a triplet loss network and a bi-encoder.",)

    args = vars(parser.parse_args())

    args["evaluate"] = True
    args["orgs_to_embed"] = Path("data") / "cleaned_all_orgs_with_linkedin.json"
    args["special_token_strategy"] = None

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=5), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(partial(objective, args=args.copy()), n_trials=15, gc_after_trial=True)

    print(study.best_params)
    print(study.best_value)
