import os
from pathlib import Path
from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from constants import CONFIG_DIR, DATASET_DIR, MODEL_DIR
from utils.data import df_self_product, prepare_reward_dataset


@hydra.main(
    config_path=CONFIG_DIR, config_name="reward_config", version_base="1.2"
)
def train_reward(cfg: DictConfig) -> None:

    logger.info("Parsing reward config")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    dataset_path = Path(DATASET_DIR, cfg.dataset.name)
    model_path = Path(MODEL_DIR, cfg.model.name)

    peft_config = LoraConfig(**cfg.peft)
    reward_config = RewardConfig(**cfg.trainer)

    logger.info(f"Loading model and tokenizer for `{cfg.model_name}`")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, **cfg.tokenizer_args
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, **cfg.model_args
    ).to(cfg.device)

    if not os.path.exists(dataset_path) or cfg.dataset.rewrite:

        logger.info(f"Creating `{cfg.dataset.name}` reward dataset")
        dataset = load_dataset(cfg.dataset.source, split=["train"]).to_pandas()
        dataset = (
            df_self_product(dataset, partition_col="label")
            .sample(cfg.dataset.n_samples)
            .rename({"text_0": "chosen", "text_1": "rejected"})
        )
        dataset = prepare_reward_dataset(
            dataset.to_dict(as_series=False), tokenizer=tokenizer, verbose=True
        )
        dataset = dataset.train_test_split(test_size=cfg.dataset.test_size)
        dataset.save_to_disk(dataset_path)
    else:
        logger.info(f"Loading {dataset_path}")
        dataset = load_from_disk(dataset_path)

    logger.info(f"Starting RewardTrainer")
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )
    trainer.train()

    logger.info(f"Saving model to `{model_path}`")
    trainer.save_model(model_path)


# poetry run scripts train_reward --config-name reward_config
