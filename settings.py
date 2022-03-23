from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigSchema:
    # Huggingface
    transformer_model: str = "sentence-transformers/LaBSE"

    # Model
    default_learning_rate: float = 5e-5
    use_vic_reg_loss: bool = False
    metric_function: str = "cosine"  # "euclidean" / "cosine"
    lambda_: int = 25
    mu: int = 14
    nu: int = 15
    max_mention_length: int = 350
    num_polyencoder_query_vectors: int = 16
    add_linear_projection: bool = False
    final_embedding_size: int = 200
    projector_number_layers: int = 2
    hidden_layer_size: int = 1000
    temperature: float = 0.06

    # Lightning Data Module settings
    default_data_folder: Path = Path("data/")
    default_batch_size: int = 40
    # default_batch_size: int = 4
    validation_frac: float = 0.2
    dataloader_workers: int = 10

    pooling_strategy: str = "mean_masked_pooling"

    # Training defaults
    gpus: int = -1
    max_epochs: int = 50
