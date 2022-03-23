import json
import os
import re
from functools import partial
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BatchEncoding

from settings import ConfigSchema

BiEncoderBatch = Tuple[torch.Tensor, BatchEncoding, BatchEncoding]
CrossEncoderBatch = Tuple[torch.Tensor, BatchEncoding]
Batch = Union[BiEncoderBatch, CrossEncoderBatch]

config = ConfigSchema()


class ModelType(Enum):
    bi_encoder = 0
    cross_encoder = 1


class DatasetStage(Enum):
    train = 0
    val = 1
    test = 2


class EmptyBatchEncoding(BatchEncoding):
    def __init__(self, length: int):
        super().__init__(data={
            "input_ids": torch.zeros(length, dtype=torch.int64),
            "attention_mask": torch.zeros(length, dtype=torch.int64),
            "token_type_ids": torch.zeros(length, dtype=torch.int64)
        })


class Matches(Dataset):
    """
    An iterable dataset that returns already batched tokenized data, specifically the `org_id`, the tokenized company string and
    the tokenized change item string
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_type: ModelType = ModelType.bi_encoder,
        stage: DatasetStage = DatasetStage.train,
        data_folder: Path = Path("data/"),
        max_context_length: int = config.max_mention_length,
        special_token_strategy: Optional[str] = None,
        data_path: Optional[Path] = None,
        change_item_minimum_token_length: int = 20
    ) -> None:
        super().__init__()

        # Checks
        if data_path is None:
            if stage == DatasetStage.train:
                file_path = data_folder / "train.json"
            elif stage == DatasetStage.val:
                file_path = data_folder / "val.json"
            else:
                # stage == DatasetStage.test
                file_path = data_folder / "test.json"
            assert os.path.exists(file_path), f"{data_folder} must be a valid path and must contain either `train.json` or `test.json`"
        else:
            file_path = data_path
            assert os.path.exists(file_path), f"{file_path} must be a valid `.json` file."

        assert special_token_strategy in [None, "humeau", "wu"], "Special token should be one of `humeau`, `wu` or None."

        self.model_type = model_type
        self.special_token_strategy = special_token_strategy
        self.stage = stage
        self.file_path = file_path
        self.change_item_minimum_length = change_item_minimum_token_length

        # Init and saving attributes
        with open(file_path) as f:
            self.raw_data = json.load(f)

        self.tokenizer = tokenizer
        self.tokenize_fn = partial(self.tokenizer, padding="max_length", truncation=True, return_tensors="pt", verbose=False)

        # Calculate max length of tokenizer input
        # Sometimes a very high value is the default, so this should be changed to the first value of the family of tokenizers used
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if self.tokenizer.model_max_length < 1e5
            else list(self.tokenizer.max_model_input_sizes.values())[0]
        )
        self.tokenizer.model_max_length = self.tokenizer_max_length

        self.max_context_length = max_context_length

    def _tokenize_window_around_mention(self, element: dict) -> BatchEncoding:
        """
        Function that tokenizes the given organization element by finding the first mention of the organization's base name,
        tokenizing the left and right windows (with uniformly random length) and adding special tokens for the mention.
        """

        # Random window length
        left_window = torch.randint(low=1, high=self.max_context_length - 5, size=(1,))

        # Find mention and build encodings according to special token strategy
        match = re.search(element["org_base_name"], element["change_item_str"])
        if match:
            start_mention, end_mention = match.span()
            # Left context
            left_context_encoding = self.tokenizer.encode(element["change_item_str"][:start_mention], add_special_tokens=False, verbose=False)
            # Mention
            if self.special_token_strategy == "wu":
                # Tokenize according to Wu et al, 2020 (https://aclanthology.org/2020.emnlp-main.519.pdf)
                # `[CLS] context_left [M_S] mention [M_E] context_right [SEP]` in BERT-like,
                # `<s> context_left [M_S] mention [M_E] context_right </s>` in RoBERTa-like
                mention_string = f"[M_S] {element['change_item_str'][start_mention:end_mention]} [M_E]"
            else:
                # self.special_token_strategy == "humeau" falls into normal tokenization
                # Tokenize normally according to Humeau et al, 2020 (https://openreview.net/pdf?id=SkxgnnNFvH)
                # `[CLS] Entity name description [SEP]` in BERT-like
                # `<s> Entity name description </s>` in RoBERTa-like
                mention_string = element["change_item_str"][start_mention:end_mention]
            mention_encoding = self.tokenizer.encode(mention_string, add_special_tokens=False, verbose=False)
            # Right context
            right_context_encoding = self.tokenizer.encode(element["change_item_str"][end_mention:], add_special_tokens=False, verbose=False)

            # Trim encoded sentences to respect max_context_length
            trimmed_left_context = [self.tokenizer.cls_token_id] + left_context_encoding[-left_window:]  # type: ignore
            right_window = self.max_context_length - len(trimmed_left_context) - len(mention_encoding) - 1
            trimmed_right_context = right_context_encoding[:right_window] + [self.tokenizer.sep_token_id]
            token_ids = trimmed_left_context + mention_encoding + trimmed_right_context

            tensor_ids = torch.zeros((1, self.tokenizer_max_length), dtype=torch.int64)
            tensor_ids[0, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.int64)[:self.tokenizer_max_length]
            attention_mask = (tensor_ids != 0).to(dtype=torch.int64)

            return BatchEncoding({
                "input_ids": tensor_ids, "attention_mask": attention_mask, "token_type_ids": torch.zeros_like(tensor_ids)
            })
        else:
            return EmptyBatchEncoding(self.tokenizer_max_length)

    def concatenate_encoded(self, company_encoding: BatchEncoding, change_item_encoding: BatchEncoding) -> BatchEncoding:
        num_tokens_company = torch.argmin(company_encoding.attention_mask) - 1
        num_tokens_company = self.tokenizer_max_length if num_tokens_company == -1 else num_tokens_company
        if num_tokens_company + self.change_item_minimum_length < self.tokenizer_max_length:
            # We have enough space to include change item encoding while keeping the whole company
            company_limit = num_tokens_company
        else:
            # We must trim company in order to have a minimum amount of change item
            company_limit = num_tokens_company - self.change_item_minimum_length

        remaining_tokens_for_change_item = company_encoding.input_ids.size(0) - company_limit
        concatenated_encoding = BatchEncoding(data={
            key: torch.cat([
                company_encoding[key][:company_limit], change_item_encoding[key][1:remaining_tokens_for_change_item + 1]  # type: ignore
            ])
            for key in company_encoding.keys()
        })

        if concatenated_encoding["input_ids"][-1] != self.tokenizer.sep_token_id:
            concatenated_encoding["input_ids"][-1] = self.tokenizer.sep_token_id

        return concatenated_encoding

    def __getitem__(self, index: int) -> Batch:
        element = self.raw_data[index]

        if element.get("org_str"):
            if self.special_token_strategy == "wu":
                # Tokenize according to Wu et al, 2020 (https://aclanthology.org/2020.emnlp-main.519.pdf)
                # `[CLS] Entity name [ENT] description [SEP]` in BERT-like,
                # `<s> Entity name [ENT] description </s>` in RoBERTa-like
                company_string = f"{self.tokenizer.cls_token} {element['org_name']} [ENT] {element['org_str']} {self.tokenizer.sep_token}"
            else:
                # self.special_token_strategy == "humeau" falls into normal tokenization
                # Tokenize normally according to Humeau et al, 2020 (https://openreview.net/pdf?id=SkxgnnNFvH)
                # `[CLS] Entity name description [SEP]` in BERT-like
                # `<s> Entity name description </s>` in RoBERTa-like
                company_string = f"{self.tokenizer.cls_token} {element['org_name']} {element['org_str']} {self.tokenizer.sep_token}"

            company_tensor = self.tokenize_fn(company_string, add_special_tokens=False)
            company_encoding = BatchEncoding(dict((key, torch.squeeze(tensor)) for key, tensor in company_tensor.items()))

            # Add sep token at the end if it was truncated
            if company_encoding["input_ids"][-1] != self.tokenizer.sep_token_id:
                company_encoding["input_ids"][-1] = self.tokenizer.sep_token_id
        else:
            company_encoding = EmptyBatchEncoding(self.tokenizer_max_length)

        if element.get("change_item_str"):
            if (self.stage == DatasetStage.train or self.stage == DatasetStage.val) and self.max_context_length > 0 and element.get("org_base_name"):
                change_item_tensor = self._tokenize_window_around_mention(element)
            else:
                change_item_tensor = self.tokenize_fn(element["change_item_str"])
            change_item_encoding = BatchEncoding(dict((key, torch.squeeze(tensor)) for key, tensor in change_item_tensor.items()))
        else:
            change_item_encoding = EmptyBatchEncoding(self.tokenizer_max_length)

        tensor_org_id = torch.unsqueeze(torch.tensor(element["org_id"]) if element.get("org_id") else torch.tensor(0), 0)

        if self.model_type == ModelType.cross_encoder:
            concatenated_encoding = self.concatenate_encoded(company_encoding, change_item_encoding)
            return (
                tensor_org_id,
                concatenated_encoding
            )
        else:
            return (
                tensor_org_id,
                company_encoding,
                change_item_encoding
            )

    def __len__(self) -> int:
        return len(self.raw_data)


class MatchesDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = config.default_batch_size,
        model_type: ModelType = ModelType.bi_encoder,
        special_token_strategy: Optional[str] = None,
        data_path: Optional[Path] = None,
        data_folder: Path = config.default_data_folder,
        max_context_length: int = config.max_mention_length
    ):
        super().__init__()

        # Checks
        assert isinstance(batch_size, int), "Batch size must be integral"
        assert model_type in [ModelType.bi_encoder, ModelType.cross_encoder], \
            f"Supported `model_type`s are `{ModelType.bi_encoder}` and `{ModelType.cross_encoder}`"

        self.batch_size = batch_size
        self.model_type = model_type
        self.special_token_strategy = special_token_strategy
        self.data_folder = data_folder
        self.data_path = data_path
        self.max_context_length = max_context_length

        self.tokenizer = AutoTokenizer.from_pretrained(config.transformer_model)
        if special_token_strategy == "wu" and model_type in [ModelType.bi_encoder, ModelType.cross_encoder]:
            # Extra special tokens required by Wu et al, 2020 (https://aclanthology.org/2020.emnlp-main.519.pdf)
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[M_S]", "[M_E]", "[ENT]"]})

        self.all_orgs_data_path = Path("data") / "cleaned_all_orgs_with_linkedin.json"

    def setup(self, stage: Optional[str] = None):
        if self.data_path is None:
            if stage == "fit" or stage is None:
                self.train_dataset = Matches(
                    self.tokenizer, model_type=self.model_type, stage=DatasetStage.train, data_folder=self.data_folder, special_token_strategy=self.special_token_strategy, max_context_length=self.max_context_length
                )
                self.val_dataset = Matches(
                    self.tokenizer, model_type=self.model_type, stage=DatasetStage.val, data_folder=self.data_folder, special_token_strategy=self.special_token_strategy, max_context_length=self.max_context_length
                )
                self.all_orgs_dataset = Matches(
                    self.tokenizer, model_type=self.model_type, data_path=self.all_orgs_data_path, special_token_strategy=self.special_token_strategy, max_context_length=self.max_context_length
                )

            if stage == "test" or stage is None:
                self.test_dataset = Matches(
                    self.tokenizer, model_type=self.model_type, stage=DatasetStage.test, data_folder=self.data_folder, special_token_strategy=self.special_token_strategy, max_context_length=self.max_context_length
                )
        else:
            self.dataset = Matches(self.tokenizer, data_path=self.data_path, model_type=self.model_type, max_context_length=self.max_context_length)

    def train_dataloader(self):
        try:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=config.dataloader_workers
            )
        except AttributeError:
            raise RuntimeError("The `train_dataloader` method is only available when the data module was set up with `stage=fit` or `stage=None`")

    def val_dataloader(self):
        try:
            return [
                DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=config.dataloader_workers
                ),
                DataLoader(
                    self.all_orgs_dataset,
                    batch_size=self.batch_size,
                    num_workers=config.dataloader_workers
                )
            ]

        except AttributeError:
            raise RuntimeError("The `val_dataloader` method is only available when the data module was set up with `stage=fit` or `stage=None`")

    def test_dataloader(self):
        try:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=config.dataloader_workers
            )
        except AttributeError:
            raise RuntimeError("The `test_dataloader` method is only available when the data module was set up with `stage=test` or `stage=None`")

    def dataloader(self):
        try:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=config.dataloader_workers
            )
        except AttributeError:
            raise RuntimeError("The `dataloader` method is only available if this class was instantiated with a `data_path`")
