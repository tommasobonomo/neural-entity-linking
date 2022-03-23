from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from settings import ConfigSchema
from src.datamodules import BiEncoderBatch, CrossEncoderBatch, Batch

EncoderOutput = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

config = ConfigSchema()


class BaseEncoder(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate: float,
        pooling_strategy: Optional[str],
    ):
        super().__init__()

        assert pooling_strategy in ["pooler_output", "mean_pooling", "mean_masked_pooling"], \
            "Pooling strategy must be chosen between `pooler_output`, `mean_pooling` and `mean_masked_pooling`"

        self.pooling_strategy = pooling_strategy
        self.learning_rate = learning_rate

        # Company embeddings
        self.epoch_company_embeds: List[torch.Tensor] = []
        self.epoch_company_ids: List[torch.Tensor] = []
        self.val_change_item_embeds: List[torch.Tensor] = []
        self.val_company_ids: List[torch.Tensor] = []
        self.val_change_item_attention_masks: List[torch.Tensor] = []

        self.save_hyperparameters()

    def _transformers_output_to_pooled(
        self,
        output: BaseModelOutputWithPoolingAndCrossAttentions,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pooling_strategy == "pooler_output":
            return output.pooler_output
        elif (
            self.pooling_strategy == "mean_pooling"
            or (attention_mask is None and self.pooling_strategy == "mean_masked_pooling")
        ):
            return torch.mean(output.last_hidden_state, dim=-1)
        elif attention_mask is not None and self.pooling_strategy == "mean_masked_pooling":
            last_layer = output.last_hidden_state
            attention_mask = attention_mask.unsqueeze(-1)
            non_masked_mean = torch.sum(attention_mask * last_layer, dim=1) / attention_mask.sum(dim=1)
            return non_masked_mean
        else:
            raise RuntimeError("`self.pooling_strategy` should be one of `pooler_output`, `mean_pooling` and `mean_masked_pooling`")

    def configure_optimizers(self):
        # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.hparams.learning_rate, adamw_mode=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=0)
        return optimizer
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss/dataloader_idx_0"
        #     }
        # }

    @abstractmethod
    def step(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, batch: Batch) -> EncoderOutput:  # type: ignore
        raise NotImplementedError()

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, batch_nb) -> Optional[torch.Tensor]:  # type: ignore
        with torch.no_grad():
            if batch_nb == 0:
                # First dataloader, just compute loss
                loss = self.step(batch)
                self.log("val_loss", loss)

                # And save projected_change_item
                _, projected_change_item = self.forward(batch)
                self.val_change_item_embeds.append(projected_change_item.cpu())
                self.val_company_ids.append(batch[0].cpu())

                return loss
            else:
                # Second dataloader, compute company embedding and save
                projected_company, _ = self.forward(batch)
                self.epoch_company_embeds.append(projected_company.cpu())
                self.epoch_company_ids.append(batch[0].cpu())

    def on_validation_epoch_end(self, outputs: torch.Tensor = None):
        with torch.no_grad():
            company_embeds = torch.cat(self.epoch_company_embeds)
            company_ids = torch.cat(self.epoch_company_ids)
            correct = torch.tensor(0)
            total = torch.tensor(0)
            true_ranks: List[torch.Tensor] = []
            for change_item_batch, true_org_ids in zip(self.val_change_item_embeds, self.val_company_ids):
                if self.use_vic_reg_loss:
                    distances = torch.cdist(change_item_batch.float(), company_embeds.float(), p=2)
                    ranks = torch.argsort(distances, dim=1, descending=False)
                else:
                    similarities = change_item_batch @ company_embeds.t()
                    ranks = torch.argsort(similarities, dim=1, descending=True)
                predicted_orgs = company_ids[ranks].squeeze()
                true_ranks += torch.argmax((predicted_orgs == true_org_ids).to(torch.int), dim=1)
                correct += torch.sum(predicted_orgs[:, 0] == true_org_ids)
                total += change_item_batch.size(0)

            self.log("val_accuracy", correct / total)
            self.log("val_mean_true_rank", torch.sum(torch.stack(true_ranks)) / len(true_ranks))

            self.epoch_company_embeds = []
            self.epoch_company_ids = []
            self.val_change_item_embeds = []
            self.val_company_ids = []

    def predict_step(self, batch: Batch, batch_idx: int) -> EncoderOutput:  # type: ignore
        return self.forward(batch)

    def on_predict_batch_end(self, predictions: EncoderOutput, batch: Batch, batch_idx: int, dataloader_idx: int):  # type: ignore
        if isinstance(predictions, tuple):
            (prediction.cpu() for prediction in predictions)
        else:
            predictions.cpu()


class BiEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_model: str,
        learning_rate: float,
        use_vic_reg_loss: bool,
        metric_function: str,
        lambda_: int,
        mu: int,
        nu: int,
        pooling_strategy: Optional[str],
        add_linear_projection: bool = config.add_linear_projection,
        projection_dimension: int = config.final_embedding_size,
        projector_number_layers: int = config.projector_number_layers,
        projector_hidden_layers_size: int = config.hidden_layer_size,
        new_vocabulary_size: Optional[int] = None,
        temperature: float = config.temperature
    ):
        super().__init__(learning_rate, pooling_strategy)

        # Add transformer encoders
        self.change_item_transformer = AutoModel.from_pretrained(transformer_model)
        self.company_transformer = AutoModel.from_pretrained(transformer_model)
        if new_vocabulary_size:
            self.change_item_transformer.resize_token_embeddings(new_vocabulary_size)
            self.company_transformer.resize_token_embeddings(new_vocabulary_size)

        # Check metric function, can only be one of two
        assert metric_function in ["cosine", "euclidean"], "Only metric functions supported are `cosine` and `euclidean`"
        self.metric_function = metric_function

        # VICReg loss hyperparams
        self.use_vic_reg_loss = use_vic_reg_loss
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.temperature = temperature

        self.add_linear_projection = add_linear_projection
        if add_linear_projection:
            assert projector_number_layers >= 2, "There must be at least two layers in the Projector"
            # Add linear projection layers
            self.change_item_projector = torch.nn.Sequential(
                ProjectorBlock(self.change_item_transformer.config.hidden_size, projector_hidden_layers_size),
                *[ProjectorBlock(projector_hidden_layers_size, projector_hidden_layers_size) for i in range(projector_number_layers - 2)],
                ProjectorBlock(projector_hidden_layers_size, projection_dimension)
            )
            self.company_projector = torch.nn.Sequential(
                ProjectorBlock(self.company_transformer.config.hidden_size, projector_hidden_layers_size),
                *[ProjectorBlock(projector_hidden_layers_size, projector_hidden_layers_size) for i in range(projector_number_layers - 2)],
                ProjectorBlock(projector_hidden_layers_size, projection_dimension)
            )

        # Company embeddings
        self.epoch_company_embeds: List[torch.Tensor] = []
        self.epoch_company_ids: List[torch.Tensor] = []
        self.val_change_item_embeds: List[torch.Tensor] = []
        self.val_company_ids: List[torch.Tensor] = []

        self.save_hyperparameters()

    def info_nce_loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u_n, v_n = u.norm(p=2, dim=1, keepdim=True), v.norm(p=2, dim=1, keepdim=True)
        eps = 1e-8
        u_norm = u / torch.clamp(u_n, min=eps)
        v_norm = v / torch.clamp(v_n, min=eps)
        cosine_similarity = u_norm @ v_norm.T
        if self.temperature:
            cosine_similarity /= self.temperature

        # InfoNCE
        loss = F.cross_entropy(cosine_similarity, torch.arange(cosine_similarity.size(0), dtype=torch.long, device=cosine_similarity.device))

        return loss

    def vic_reg_loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, embedding_dim = u.size()

        # Invariance loss (or alignment loss)
        sim_loss = F.mse_loss(u, v)

        # Variance loss
        std_u = torch.sqrt(u.var(dim=0) + 1e-04)
        std_v = torch.sqrt(v.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_u)) + torch.mean(F.relu(1 - std_v))

        # Covariance loss
        z_u = u - u.mean(dim=0)
        z_v = v - v.mean(dim=0)
        cov_z_u = ((z_u.T @ z_u) / (batch_size - 1)).fill_diagonal_(0)
        cov_z_v = ((z_v.T @ z_v) / (batch_size - 1)).fill_diagonal_(0)
        cov_loss = (cov_z_u.pow_(2).sum() / embedding_dim) + (cov_z_v.pow_(2).sum() / embedding_dim)

        loss = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * cov_loss
        return loss

    def step(self, batch: BiEncoderBatch) -> torch.Tensor:

        projected_company, projected_change_item = self.forward(batch)

        if projected_company is None or projected_change_item is None:
            raise RuntimeError("Cannot have `None` projection during training")
        else:
            if self.use_vic_reg_loss:
                loss = self.vic_reg_loss(projected_company, projected_change_item)
            else:
                loss = self.info_nce_loss(projected_company, projected_change_item)

            return loss

    def forward(self, batch: BiEncoderBatch) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:  # type:ignore
        _, company, change_item = batch

        if company:
            transformer_output_company = self.company_transformer(**company)
            pooled_embedding_company = self._transformers_output_to_pooled(transformer_output_company, company.get("attention_mask"))
            projected_company = self.company_projector(pooled_embedding_company) if self.add_linear_projection else pooled_embedding_company
        else:
            projected_company = None

        if change_item:
            transformer_output_change_item = self.change_item_transformer(**change_item)
            pooled_embedding_change_item = self._transformers_output_to_pooled(transformer_output_change_item, change_item.get("attention_mask"))
            projected_change_item = self.change_item_projector(
                pooled_embedding_change_item) if self.add_linear_projection else pooled_embedding_change_item
        else:
            projected_change_item = None

        return projected_company, projected_change_item

    def embed_company(self, company_batch: BatchEncoding) -> Optional[torch.Tensor]:
        with torch.no_grad():
            return self.forward((None, company_batch, None))[0]

    def embed_change_item(self, change_item_batch: BatchEncoding) -> Optional[torch.Tensor]:
        with torch.no_grad():
            return self.forward((None, None, change_item_batch))[1]


class PolyEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_model: str,
        learning_rate: float,
        use_vic_reg_loss: bool,
        metric_function: str,
        lambda_: int,
        mu: int,
        nu: int,
        temperature: float,
        pooling_strategy: Optional[str],
        add_linear_projection: bool = config.add_linear_projection,
        projection_dimension: int = config.final_embedding_size,
        projector_number_layers: int = config.projector_number_layers,
        projector_hidden_layers_size: int = config.hidden_layer_size,
        new_vocabulary_size: Optional[int] = None,
        number_query_vectors: int = config.num_polyencoder_query_vectors,
        scale_attention: bool = False,
        masked_attention: bool = True,
    ):
        super().__init__(learning_rate, pooling_strategy)
        self.change_item_transformer = AutoModel.from_pretrained(transformer_model)
        self.company_transformer = AutoModel.from_pretrained(transformer_model)
        if new_vocabulary_size:
            self.change_item_transformer.resize_token_embeddings(new_vocabulary_size)
            self.company_transformer.resize_token_embeddings(new_vocabulary_size)

        self.scale_attention = scale_attention
        self.masked_attention = masked_attention
        self.query_vectors = torch.nn.Parameter(torch.zeros(number_query_vectors, self.change_item_transformer.config.hidden_size))
        torch.nn.init.xavier_uniform_(self.query_vectors)

        # Check metric function, can only be one of two
        assert metric_function in ["cosine", "euclidean"], "Only metric functions supported are `cosine` and `euclidean`"
        self.metric_function = metric_function

        # VICReg loss hyperparams
        self.use_vic_reg_loss = use_vic_reg_loss
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.temperature = temperature

        self.add_linear_projection = add_linear_projection
        if add_linear_projection:
            assert projector_number_layers >= 2, "There must be at least two layers in the Projector"
            # Add linear projection layers
            self.change_item_projector = torch.nn.Sequential(
                ProjectorBlock(self.change_item_transformer.config.hidden_size, projector_hidden_layers_size),
                *[ProjectorBlock(projector_hidden_layers_size, projector_hidden_layers_size) for i in range(projector_number_layers - 2)],
                ProjectorBlock(projector_hidden_layers_size, projection_dimension)
            )
            self.company_projector = torch.nn.Sequential(
                ProjectorBlock(self.company_transformer.config.hidden_size, projector_hidden_layers_size),
                *[ProjectorBlock(projector_hidden_layers_size, projector_hidden_layers_size) for i in range(projector_number_layers - 2)],
                ProjectorBlock(projector_hidden_layers_size, projection_dimension)
            )

        self.save_hyperparameters()

    def _softmax_scale(self, alphas: torch.Tensor, dimension: int, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scale_attention:
            # Following Vaswani et al. (2017) (https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
            # If `key_dim` (d_k in the paper) is large, the dot product risks being too large and therefore the gradient out of the
            # softmax might be too small to learn properly.
            key_dim = torch.tensor(dimension)
            alphas = alphas / torch.sqrt(key_dim)

        if self.masked_attention and attention_mask is not None:
            alphas = alphas.masked_fill((1 - attention_mask.unsqueeze(1)).bool(), float("-inf"))

        softmaxed_alphas = F.softmax(alphas, dim=-1)
        return torch.nan_to_num(softmaxed_alphas, 0)

    def attention_block(
        self, query_vectors: torch.Tensor, last_hidden_layer: torch.Tensor, company_embedding: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # M: number of query vectors, B: batch size, S: sentence length in tokens, D: token embedding size
        # query_vectors: (M, D)
        # last_hidden_layer: (B, S, D)
        # company_embedding: (B, D)

        if company_embedding is None:
            raise RuntimeError("Cannot have `None` company embedding when calculating PolyEncoder output")

        if self.masked_attention:
            assert attention_mask is not None, "Attention mask must be defined if masked attention is required"
            last_hidden_layer = last_hidden_layer * attention_mask.unsqueeze(-1)

        # First attention block
        alphas1 = torch.einsum("md, bsd -> bms", query_vectors, last_hidden_layer)  # (B, M, S)
        alphas1 = self._softmax_scale(alphas1, last_hidden_layer.size(-1), attention_mask)  # Element-wise operation, with softmax on last axis S (sentence)
        queried_last_layer = torch.einsum("bms, bsd -> bmd", alphas1, last_hidden_layer)  # (B, M, D)

        # Second attention block
        alphas2 = torch.einsum("...d, bmd -> ...bm", company_embedding, queried_last_layer)  # (B, M)
        alphas2 = self._softmax_scale(alphas2, queried_last_layer.size(-1))  # Element-wise operation, with softmax on last axis M (query vectors)
        final_change_item_embedding = torch.einsum("...bm, bmd -> ...bd", alphas2, queried_last_layer)  # (B, D)

        return final_change_item_embedding

    def info_nce_loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u_n, v_n = u.norm(p=2, dim=1, keepdim=True), v.norm(p=2, dim=1, keepdim=True)
        eps = 1e-8
        u_norm = u / torch.clamp(u_n, min=eps)
        v_norm = v / torch.clamp(v_n, min=eps)
        cosine_similarity = u_norm @ v_norm.T
        if self.temperature:
            cosine_similarity /= self.temperature

        # InfoNCE
        loss = F.cross_entropy(cosine_similarity, torch.arange(cosine_similarity.size(0), dtype=torch.long, device=cosine_similarity.device))

        return loss

    def vic_reg_loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, embedding_dim = u.size()

        # Invariance loss (or alignment loss)
        sim_loss = F.mse_loss(u, v)

        # Variance loss
        std_u = torch.sqrt(u.var(dim=0) + 1e-04)
        std_v = torch.sqrt(v.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_u)) + torch.mean(F.relu(1 - std_v))

        # Covariance loss
        z_u = u - u.mean(dim=0)
        z_v = v - v.mean(dim=0)
        cov_z_u = ((z_u.T @ z_u) / (batch_size - 1)).fill_diagonal_(0)
        cov_z_v = ((z_v.T @ z_v) / (batch_size - 1)).fill_diagonal_(0)
        cov_loss = (cov_z_u.pow_(2).sum() / embedding_dim) + (cov_z_v.pow_(2).sum() / embedding_dim)

        loss = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * cov_loss
        return loss

    def step(self, batch: BiEncoderBatch, validation: bool = False) -> torch.Tensor:
        # Sizes:
        # B: batch size; D: hidden embedding size

        # (B, D), (B, B, D)
        projected_company, projected_change_item = self(batch)

        # We know the correct company that was used for the change item projection, so we can take it on the diagonal
        # (B, D)
        projected_change_item = torch.diagonal(projected_change_item).T

        if projected_company is None or projected_change_item is None:
            raise RuntimeError("Cannot have `None` projection during training")
        else:
            if self.use_vic_reg_loss:
                loss = self.vic_reg_loss(projected_company, projected_change_item)
            else:
                loss = self.info_nce_loss(projected_company, projected_change_item)

            return loss

    def forward(self, batch: BiEncoderBatch) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:  # type:ignore
        _, company, change_item = batch

        if company is not None:
            transformer_output_company = self.company_transformer(**company)
            pooled_embedding_company = self._transformers_output_to_pooled(transformer_output_company, company.get("attention_mask"))
            projected_company: Optional[torch.Tensor] = (
                self.company_projector(pooled_embedding_company) if self.add_linear_projection else pooled_embedding_company
            )
        else:
            projected_company = None

        if change_item is not None:
            transformer_output_change_item = self.change_item_transformer(**change_item).last_hidden_state
            attentioned_change_item: Optional[torch.Tensor] = self.attention_block(
                self.query_vectors, transformer_output_change_item, projected_company, change_item.get("attention_mask")
            )
        else:
            attentioned_change_item = None

        return projected_company, attentioned_change_item

    def embed_company(self, company_batch: BatchEncoding) -> Optional[torch.Tensor]:
        with torch.no_grad():
            return self.forward((None, company_batch, None))[0]

    def company_embedding_idx_for_change_item(
        self, transformer_output_change_item: torch.Tensor, company_embeddings: torch.Tensor, attention_mask: torch.Tensor, metric: str = "euclidean_distance", cdist_on_gpu: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We expect to receive company_embeddings with shape (N, D), and a change item with batch B
        with torch.no_grad():
            # Move embeddings to full precision for distance calculation
            # if company_embeddings.dtype == torch.half:
            #     company_embeddings = company_embeddings.to(torch.float)
            # self.float()

            # transformer_output_change_item = self.change_item_transformer(**change_item_batch).last_hidden_state  # (B, S, D)
            attentioned_change_items_per_company_embedding = self.attention_block(
                self.query_vectors, transformer_output_change_item, company_embeddings.cuda(), attention_mask
            )  # (N, B, D)

            if metric == "euclidean_distance":
                # Using euclidean distance as score between embeddings. Will take ascending argsort as we are using a distance.
                company_embeddings = company_embeddings.unsqueeze(1).float()  # (N, 1, D)
                attentioned_change_items_per_company_embedding = attentioned_change_items_per_company_embedding.float()
                if cdist_on_gpu:
                    scores = torch.cdist(attentioned_change_items_per_company_embedding, company_embeddings,
                                         p=2, compute_mode="donot_use_mm_for_euclid_dist").squeeze().T  # (B, N)
                else:
                    scores = torch.cdist(attentioned_change_items_per_company_embedding.cpu(), company_embeddings.cpu(), p=2).squeeze().T  # (B, N)
                indices = torch.argsort(scores, dim=1, descending=False)  # (B, N)
            else:
                # Using dot product as score between embeddings. Dot product is a similarity measure, so will take argmax.
                scores = torch.einsum("nbd, nd -> bn", attentioned_change_items_per_company_embedding.float().cpu(), company_embeddings.float().cpu())  # (B, N)
                indices = torch.argsort(scores, dim=1, descending=True)  # (B, N)

            return indices, scores

    def validation_step(self, batch, batch_idx, batch_nb) -> Optional[torch.Tensor]:  # type: ignore
        with torch.no_grad():
            if batch_nb == 0:
                # First dataloader, just compute loss
                loss = self.step(batch)
                self.log("val_loss", loss)

                # And save raw change item
                company_ids, _, raw_change_item = batch
                last_hidden_state = self.change_item_transformer(**raw_change_item).last_hidden_state
                self.val_change_item_embeds.append(last_hidden_state)
                self.val_company_ids.append(company_ids)
                # Avoid weird bug for attention mask saving
                if batch_idx == 0:
                    self.val_change_item_attention_masks = []
                self.val_change_item_attention_masks.append(raw_change_item.get("attention_mask"))

                pass
            else:
                # Second dataloader, compute company embedding and save
                projected_company, _ = self.forward(batch)
                self.epoch_company_embeds.append(projected_company.cpu())
                self.epoch_company_ids.append(batch[0].cpu())

    def on_validation_epoch_end(self, outputs: torch.Tensor = None):
        with torch.no_grad():
            company_embeds = torch.cat(self.epoch_company_embeds)
            company_ids = torch.cat(self.epoch_company_ids)
            correct = torch.tensor(0)
            total = torch.tensor(0)
            true_ranks: List[torch.Tensor] = []
            for change_item_batch, true_org_ids, attention_mask in zip(self.val_change_item_embeds, self.val_company_ids, self.val_change_item_attention_masks):
                # Calculate the ranks now
                metric = "euclidean_distance" if self.use_vic_reg_loss else "cosine_similarity"
                ranks, _ = self.company_embedding_idx_for_change_item(change_item_batch, company_embeds, attention_mask, metric)

                predicted_orgs = company_ids[ranks].squeeze()
                true_ranks += torch.argmax((predicted_orgs == true_org_ids.cpu()).to(torch.int), dim=1)
                correct += torch.sum(predicted_orgs[:, 0] == true_org_ids.cpu())
                total += ranks.size(0)

            self.log("val_accuracy", correct / total)
            self.log("val_mean_true_rank", torch.sum(torch.stack(true_ranks)) / len(true_ranks))

            self.epoch_company_embeds = []
            self.epoch_company_ids = []
            self.val_change_item_embeds = []
            self.val_company_ids = []


class CrossEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_model: str = config.transformer_model,
        learning_rate: float = config.default_learning_rate,
        pooling_strategy: Optional[str] = "pooler_output",
        use_bias: bool = False,
        new_vocabulary_size: Optional[int] = None,
    ):
        super().__init__(learning_rate, pooling_strategy)

        self.transformer = AutoModel.from_pretrained(transformer_model)
        if new_vocabulary_size:
            self.transformer.resize_token_embeddings(new_vocabulary_size)

        self.score_projection = torch.nn.Linear(
            in_features=self.transformer.config.hidden_size, out_features=1, bias=use_bias
        )

        self.save_hyperparameters()

    def batch_contrastive_loss_fn(self, scores: torch.Tensor) -> torch.Tensor:
        # Shape of scores: (batch_size, 1)
        # Cross entropy loss of one element vs all other in the batch
        # The loss for each item of the batch follows from Equation 4 in Wu et al, 2020 (https://aclanthology.org/2020.emnlp-main.519.pdf)

        # If defined, scale scores by temperature (helps with very small numerical values in scores)
        if self.temperature:
            scores /= self.temperature

        # Get log softmax across batch of scores
        log_softmaxed_scores = F.log_softmax(scores, dim=0)

        # Next calculate loss for each element i of batch
        # loss_i = -sum(((1 if i == j else 0) * log_softmaxed_scores_j) for j in batch_idx)
        # loss_i = -log_softmaxed_scores_i
        # This is actually equivalent to the formulation by Wu et al (2020) in the paper
        # loss_i = -score(m_i, e_i) + log sum(exp(score(m_j, e_j)) for j in batch_indices)
        loss_per_item = -log_softmaxed_scores

        # Reduced loss through mean
        loss = torch.mean(loss_per_item)
        return loss

    def step(self, batch: CrossEncoderBatch) -> torch.Tensor:
        scores = self(batch)

        # Compute loss
        loss = self.batch_contrastive_loss_fn(scores)

        return loss

    def forward(self, batch: CrossEncoderBatch) -> torch.Tensor:  # type: ignore
        _, concatenated_tokenized = batch

        #  Pass through transformer
        transformer_out = self.transformer(**concatenated_tokenized)

        # Pool according to predefined strategy
        pooled_embeddings = self._transformers_output_to_pooled(transformer_out, concatenated_tokenized["attention_mask"])

        # Pass through scoring layer
        scores = self.score_projection(pooled_embeddings)

        return scores


class Transformer(torch.nn.Module):
    def __init__(self, transformer_model: str = "sentence-transformers/LaBSE", frozen: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(transformer_model).requires_grad_(not frozen)

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        return self.model(**batch).last_hidden_state


class Embedder(torch.nn.Module):
    def __init__(self, transformer: Optional[AutoModel] = None, frozen_transformer: bool = True):
        super().__init__()

        # Set transformer model
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = Transformer(config.transformer_model, frozen_transformer)

        # Flatten sequences
        self.flattener = torch.nn.Flatten()

        # Single FF layer
        self.feed_forward_1 = torch.nn.Linear(
            self.transformer.model.config.hidden_size * self.transformer.model.config.max_position_embeddings,
            config.final_embedding_size
        )

        # Set FF layers
        # ff_hidden_layer_size = config.hidden_layer_size
        # self.feed_forward_1 = torch.nn.Linear(
        #     self.transformer.model.config.hidden_size * self.transformer.model.config.max_position_embeddings,
        #     ff_hidden_layer_size
        # )
        # self.feed_forward_2 = torch.nn.Linear(
        #     ff_hidden_layer_size,
        #     config.final_embedding_size
        # )

        # Set dropout
        # self.dropout = torch.nn.Dropout(p=config.dropout_rate)

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        x = self.transformer(batch)
        x = self.flattener(x)
        x = self.feed_forward_1(x)
        # x = self.feed_forward_2(x)
        # x = self.dropout(x)
        return x


class ProjectorBlock(torch.nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.linear = torch.nn.Linear(in_dims, out_dims)
        self.batch_norm = torch.nn.BatchNorm1d(out_dims)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class NeuralMatcher(pl.LightningModule):
    def __init__(self, learning_rate: float = config.default_learning_rate, frozen_transformer: bool = True):
        super().__init__()

        self.learning_rate = learning_rate

        if frozen_transformer:
            transformer = Transformer(transformer_model=config.transformer_model, frozen=True)
            self.company_embedder = Embedder(transformer=transformer)
            self.change_item_embedder = Embedder(transformer=transformer)
        else:
            self.company_embedder = Embedder(frozen_transformer=False)
            self.change_item_embedder = Embedder(frozen_transformer=False)

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)

    def triplet_loss(self, company: torch.Tensor, change_item: torch.Tensor, org_ids: torch.Tensor) -> torch.Tensor:

        norm_company = F.normalize(company, p=2)
        norm_change_item = F.normalize(change_item, p=2)

        similarities = norm_company @ norm_change_item.t()

        mask = torch.stack([(org_ids == idx).squeeze() for idx in org_ids])
        positive_similarities = similarities.masked_fill(~mask, 1)
        negative_similarities = similarities.masked_fill(mask, -1)

        hardest_positive = torch.min(positive_similarities, dim=-1)
        hardest_negative = torch.max(negative_similarities, dim=-1)

        loss = F.softplus(-hardest_positive.values + hardest_negative.values)

        return torch.mean(loss)

    def step(self, batch: Batch) -> torch.Tensor:
        org_ids, company, change_item = batch

        # Pass through embedding pipeline
        company = self.company_embedder(company)
        change_item = self.change_item_embedder(change_item)

        # Get loss
        loss = self.triplet_loss(company, change_item, org_ids)

        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def embed_company(self, company_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            company = self.company_embedder(company_batch)
            company = F.normalize(company, p=2)
            return company

    def embed_change_item(self, change_item_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            change_item = self.change_item_embedder(change_item_batch)
            change_item = F.normalize(change_item, p=2)
            return change_item

    def forward(self, change_item_batch: torch.Tensor) -> torch.Tensor:  # type:ignore
        return self.embed_change_item(change_item_batch)
