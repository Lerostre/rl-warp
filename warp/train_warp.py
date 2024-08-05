import hydra
import os
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from constants import DATASET_DIR, MODEL_DIR
from structs.warp_trainer import WARPTrainer
from utils.data import create_warp_dataset
from utils.misc import seed_everything

from peft import get_peft_model, LoraConfig


@hydra.main(
    config_path="configs", config_name="warp_config", version_base="1.2"
)
def train_warp(cfg: DictConfig) -> None:

    # define configs and paths
    logger.info("Parsing WARP config")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(cfg.seed)

    dataset_path = Path(DATASET_DIR, cfg.dataset.name)
    sft_model_path = Path(MODEL_DIR, cfg.sft_model.name)
    sft_tokenizer_path = Path(MODEL_DIR, cfg.sft_tokenizer.source)
    if not os.path.exists(sft_tokenizer_path):
        sft_tokenizer_path = cfg.sft_tokenizer.source
    if not os.path.exists(sft_model_path):
        sft_model_path = cfg.sft_model.source
    reward_tokenizer_path = Path(MODEL_DIR, cfg.reward_tokenizer.source)
    reward_model_path = Path(MODEL_DIR, cfg.reward_model.name)

    # load tokenizers and models
    logger.info(f"Loading SFT tokenizer from `{sft_tokenizer_path}`")
    sft_tokenizer = AutoTokenizer.from_pretrained(
        sft_tokenizer_path,
        **cfg.sft_tokenizer.args,
    )
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
    logger.info(f"Loading SFT model from `{sft_model_path}`")
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, **cfg.sft_model.args
    ).to(cfg.device)
    logger.info(f"Loading Reward tokenizer from `{reward_tokenizer_path}`")
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_tokenizer_path,
        **cfg.reward_tokenizer.args,
    )
    logger.info(f"Loading Reward model from `{reward_model_path}`")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        **cfg.reward_model.args,
    ).to(cfg.device)

    # prepare dataset and dataloader
    if not os.path.exists(dataset_path) or cfg.dataset.rewrite:
        logger.info(f"Creating `{cfg.dataset.name}` dataset")
        dataset = create_warp_dataset(
            source=cfg.dataset.source,
            tokenizer=sft_tokenizer,
            max_length=cfg.dataset.max_length,
            dataset_path=dataset_path,
        )
    else:
        logger.info(f"Loading {dataset_path}")
        dataset = load_from_disk(dataset_path)
    loader = DataLoader(dataset, shuffle=True, **cfg.dataloader)

    # launch trainer
    logger.info(f"Starting WARPTrainer")
    # wandb.init(
    #     config=OmegaConf.to_container(cfg),
    #     project=cfg.project,
    #     name=str(cfg.run_name),
    # )

    lora_config = LoraConfig(OmegaConf.to_container(cfg.peft))
    get_peft_model(sft_model, lora_config)

    trainer = WARPTrainer(
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        sft_tokenizer=sft_tokenizer,
        sft_model=sft_model,
        optimizer=hydra.utils.get_class(cfg.optimizer.cls),
        optimizer_kwargs=OmegaConf.to_container(cfg.optimizer.kwargs),
        scheduler=hydra.utils.get_class(cfg.scheduler.cls),
        scheduler_kwargs=OmegaConf.to_container(cfg.scheduler.kwargs),
        loader=loader,
        device=cfg.device,
        **cfg.trainer,
    )
    trainer.training_iterations()

    logger.info(f"Saving model to `{cfg.run_name}`")
    trainer.sft_model.save_pretrained(Path(MODEL_DIR), cfg.run_name+"/")


train_warp()

# poetry run python warp/train_warp.py --config-name warp_config
