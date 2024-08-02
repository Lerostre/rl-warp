import gc
from typing import Sequence, Literal, Tuple, List, Dict

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm


def multi_join(
    df_list: Sequence[pl.DataFrame | pl.LazyFrame],
    on: Sequence[str | None] | str | None,
    how: pl._typing.JoinStrategy = "inner",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Join the list of pl.DataFrame of length N
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
    s: pl.Series, n_samples: int | float | None, sample_mode: Literal["exact", "approximate"] = "exact"
) -> pl.Series:
    """
    Custom pl.LazyFrame.sample implementation using shuffle or binomial sampling techinques
    """
    if isinstance(n_samples, float):
        n_samples = int(s.len() * n_samples)
    elif n_samples is None:
        n_samples = s.len()

    if sample_mode == "exact":
        values = np.random.permutation(np.hstack([np.ones(n_samples), np.zeros(s.len() - n_samples)]))
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

    dataset = pl.DataFrame(dataset).partition_by(partition_col, as_dict=True, include_key=False)
    dataset = [dataset[key].select(pl.all().name.suffix(f"_{key[0]}")).lazy() for key in dataset]
    dataset = multi_join(dataset, on=fields, how="cross")

    # pl.LazyFrame has no efficient method of sampling, thie block below is a placeholder for the future
    if n_samples is not None:
        dataset = (
            dataset.with_columns(
                sample=pl.first().map_batches(lambda x: sample_it(x, n_samples=n_samples, sample_mode=sample_mode))
            )
            .filter(pl.col("sample"))
            .drop("sample")
        )

    return dataset.collect(streaming=True)


def prepare_reward_dataset(examples: Tuple[List], tokenizer) -> Dict:
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples
