import gc
from typing import Literal, Sequence, TypeVar, Optional
from loguru import logger

import numpy as np
import pandas as pd
import polars as pl
from torch.nn import functional as F

from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

pl_df = TypeVar("pl_df", pl.DataFrame, pl.LazyFrame)


def multi_join(
    df_list: Sequence[pl_df],
    on: Sequence[str | None] | str | None,
    how: pl._typing.JoinStrategy = "inner",
) -> pl_df:
    """
    Join the list of pl.DataFrame or pl.LazyFrame of length N
    using respective keys of length N-1 as `on`
    with a given `how` method, common for all dataframes
    """

    if not isinstance(on, Sequence):
        on = [on] * len(df_list)

    while len(df_list) > 1:
        df_list[0] = df_list[0].join(df_list[1], on=on[0], how=how)
        del df_list[1], on[1]
        gc.collect()

    return df_list[0]


def sample_it(
    s: pl.Series,
    n_samples: int | float | None,
    sample_mode: Literal["exact", "approximate"] = "exact",
) -> pl.Series:
    """
    Custom pl.LazyFrame.sample implementation using shuffle
    or binomial sampling techinques
    """
    if isinstance(n_samples, float):
        n_samples = int(s.len() * n_samples)
    elif n_samples is None:
        n_samples = s.len()

    if sample_mode == "exact":
        values = np.random.permutation(
            np.hstack([np.ones(n_samples), np.zeros(s.len() - n_samples)])
        )
    elif sample_mode == "approximate":
        values = np.random.binomial(1, n_samples / s.len(), s.len())

    return pl.Series(
        values=values,
        dtype=pl.Boolean,
    )


def df_self_product(
    dataset: pd.DataFrame | pl.DataFrame,
    partition_col: str,
    fields: Sequence[str] | str | None = None,
    n_samples: int | float | None = None,
    sample_mode: Literal["exact", "approximate"] = "approximate",
) -> pl.DataFrame:
    """
    Dataframe self cross product of different columns with sampling if necessary
    """

    if fields is None:
        fields = dataset.columns

    dataset = pl.DataFrame(dataset).partition_by(
        partition_col, as_dict=True, include_key=False
    )
    dataset = [
        dataset[key].select(pl.all().name.suffix(f"_{key[0]}")).lazy()
        for key in dataset
    ]
    dataset = multi_join(dataset, on=fields, how="cross")

    # pl.LazyFrame has no efficient method of sampling,
    # the block below is a placeholder for the future
    if n_samples is not None:
        dataset = (
            dataset.with_columns(
                sample=pl.first().map_batches(
                    lambda x: sample_it(
                        x, n_samples=n_samples, sample_mode=sample_mode
                    )
                )
            )
            .filter(pl.col("sample"))
            .drop("sample")
        )

    return dataset.collect(streaming=True)


def prepare_reward_dataset(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    truncation: bool = True,
) -> Dataset:

    token_kwargs = dict(
        truncation=truncation,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    new_examples = dict()

    for texts in examples.keys():
        logger.info(f"Starting tokenizing `{texts}`")
        tokenized = tokenizer(text=examples[texts], **token_kwargs)
        tokenized = {k + "_" + texts: v for k, v in tokenized.items()}
        new_examples.update(tokenized)
    dataset = Dataset.from_dict(new_examples)

    dataset.set_format(type="torch")
    return dataset


def create_reward_dataset(
    source: str,
    tokenizer: PreTrainedTokenizer,
    n_samples: int | float = 25000,
    test_size: int | float = 0.2,
    dataset_path: str | None = None,
) -> Dataset:
    dataset = load_dataset(source, split="train").to_pandas()
    dataset = (
        df_self_product(dataset, partition_col="label")
        .sample(n_samples)
        .rename({"text_0": "chosen", "text_1": "rejected"})
    )
    dataset = prepare_reward_dataset(
        dataset.to_dict(as_series=False), tokenizer=tokenizer
    )
    dataset = dataset.train_test_split(test_size=test_size)
    if dataset_path is not None:
        dataset.save_to_disk(dataset_path)
    return dataset


def prepare_warp_dataset(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizer,
    model: Optional[PreTrainedModel] = None,
    max_length: int = 15,
    truncation: bool = True,
) -> Dataset:

    new_examples = dict()

    logger.info("Starting tokenizing `text`")
    tokenized = tokenizer(
        examples["text"],
        truncation=truncation,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    new_examples.update(tokenized)
    new_examples.update({"text": examples["text"]})

    # potentially might be easier to preload rewards
    # logger.info("Starting reward estimation of `text`")
    # new_examples["reward"] = F.softmax(
    #     model(
    #         input_ids=new_examples["input_ids"],
    #         attention_mask=new_examples["attention_mask"],
    #     ).logits
    # )[:, 1]

    dataset = Dataset.from_dict(new_examples)
    dataset.set_format(type="torch")
    return dataset


def create_warp_dataset(
    source: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 15,
    truncation: bool = True,
    dataset_path: str | None = None,
) -> Dataset:
    dataset = load_dataset(source, split="train").to_pandas()
    dataset = prepare_warp_dataset(
        pl.DataFrame(dataset).to_dict(as_series=False),
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=truncation,
    )
    if dataset_path is not None:
        dataset.save_to_disk(dataset_path)
    return dataset
