from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from omegaconf import OmegaConf

import os

peft_params = OmegaConf.load('../scripts/configs/peft_reward.yaml')
peft_config = LoraConfig(**peft_params)

reward_trainer_params = OmegaConf.load('../scripts/configs/config_reward_trainer.yaml')
reward_config = RewardConfig(**reward_trainer_params)

trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    tokenizer=reward_tokenizer,
    train_dataset=reward_dataset["train"],
    eval_dataset=reward_dataset["test"]
)

trainer.train()