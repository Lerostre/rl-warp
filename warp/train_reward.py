import hydra
import os
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from constants import DATASET_DIR, MODEL_DIR
from utils.data import create_reward_dataset
from utils.misc import seed_everything


@hydra.main(
    config_path="configs", config_name="reward_config", version_base="1.2"
)
def train_reward(cfg: DictConfig) -> None:

    # define configs and paths
    logger.info("Parsing reward config")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(cfg.seed)

    dataset_path = Path(DATASET_DIR, cfg.dataset.name)
    model_path = Path(MODEL_DIR, cfg.model.name)
    tokenizer_path = Path(MODEL_DIR, cfg.tokenizer.source)
    orig_model_path = Path(MODEL_DIR, cfg.model.source)

    # load trainer configs
    peft_config = LoraConfig(**OmegaConf.to_container(cfg.peft))
    reward_config = RewardConfig(**OmegaConf.to_container(cfg.trainer))

    # load tokenizer and model
    if not os.path.exists(tokenizer_path):
        tokenizer_path = cfg.tokenizer.source
    if not os.path.exists(orig_model_path):
        orig_model_path = cfg.model.source
    logger.info(f"Loading tokenizer from `{tokenizer_path}`")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        **cfg.tokenizer.args,
    )
    logger.info(f"Loading model from `{orig_model_path}`")
    model = AutoModelForSequenceClassification.from_pretrained(
        orig_model_path, **cfg.model.args
    ).to(cfg.model.device)

    # prepare dataset
    if not os.path.exists(dataset_path) or cfg.dataset.rewrite:
        logger.info(f"Creating `{cfg.dataset.name}` dataset")
        dataset = create_reward_dataset(
            source=cfg.dataset.source,
            tokenizer=tokenizer,
            n_samples=cfg.dataset.n_samples,
            test_size=cfg.dataset.test_size,
            dataset_path=dataset_path,
        )
    else:
        logger.info(f"Loading {dataset_path}")
        dataset = load_from_disk(dataset_path)

    # launch trainer
    logger.info(f"Starting RewardTrainer")
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # peft_config=peft_config,
    )
    trainer.train()

    logger.info(f"Saving model to `{model_path}`")
    trainer.save_model(model_path)


train_reward()

# poetry run python warp/train_reward.py --config-name reward_config
