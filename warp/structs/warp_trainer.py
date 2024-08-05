import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from tqdm.auto import tqdm, trange
from typing import Dict, Optional, Iterable
from utils.train import slerp, policy
from torch.utils.data import DataLoader

from collections import defaultdict
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

import copy


class TorchBase(nn.Module):
    """
    Base class for other pl models. Accepts any torch.optim optimizer
    and compatible schedulers. Does not support pl.Tuner usage!
    Args:
        optimizer: any torch optimizer
        optimizer_kwargs: optimizer parameters, such as lr
        scheduler: any scheduler compatible with torch optimizer
        scheduler_kwargs: scheduler parameters, such as warmup, num_training_steps etc
        save_hparams: flag for hyperparameters logging
    """

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = optim.SGD,
        optimizer_kwargs: Optional[dict[str, object]] = dict(lr=0.1),
        scheduler: Optional[object] = None,
        scheduler_kwargs: Optional[dict[str, object]] = dict(),
        device: str = "cpu",
    ):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.device = device


class WARPTrainer(TorchBase):
    """
    Encoder-decoder architecture adapted from transformers.
    Accepts images as encoder input and class labels as decoder input.
    Args:
        config: BertConfig from transformers
        input_size: input_size for image embedding projection
    """

    def __init__(
        self,
        sft_model: AutoModelForCausalLM,
        sft_tokenizer: AutoTokenizer,
        reward_model: AutoModelForSequenceClassification,
        reward_tokenizer: AutoTokenizer,
        optimizer: Optional[torch.optim.Optimizer] = optim.SGD,
        optimizer_kwargs: Optional[dict[str, object]] = dict(lr=0.1),
        scheduler: Optional[object] = None,
        scheduler_kwargs: Optional[dict[str, object]] = dict(),
        num_iterations: Optional[int] = 2,
        num_runs: Optional[int] = 2,
        num_steps: Optional[int] = 100,
        mu: Optional[float] = 0.01,
        lamb: Optional[float] = 0.05,
        eta: Optional[float] = 0.5,
        loader: Optional[DataLoader] = None,
        beta: Optional[float] = 0.1,
        device: str = "cuda",
        temperature: Optional[float] = 0.9,
        min_new_tokens: Optional[int] = 15,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            device=device,
        )
        self.sft_model = sft_model
        self.sft_tokenizer = sft_tokenizer
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.num_iterations = num_iterations
        self.num_steps = num_steps
        self.num_runs = num_runs
        self.num_runs = num_runs
        self.mu = mu
        self.lamb = lamb
        self.eta = eta
        self.loader = loader
        self.beta = beta
        self.temperature = temperature
        self.min_new_tokens = min_new_tokens

    def training_step(self):

        # make copies of sft_model
        theta = copy.deepcopy(self.sft_model)
        theta_ema = copy.deepcopy(self.sft_model)
        # start tokenizer and scheduler
        optimizer = self.optimizer(
            params=theta.parameters(), **self.optimizer_kwargs
        )
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        loss_history = defaultdict(list)

        # for t in T
        for t in trange(self.num_steps, desc="Going through steps"):

            optimizer.zero_grad()

            batch = next(iter(self.loader))
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # generate y ~ \pi_theta(| x)
            completion = theta.generate(
                input_ids,
                attention_mask=attention_mask,
                temperature=self.temperature,
                # min_new_tokens=self.min_new_tokens,
                pad_token_id=self.sft_tokenizer.eos_token_id,
            )
            len_generated = completion.shape[1] - input_ids.shape[1]

            # get policies, reward and compute KL reg
            logps_theta = policy(theta, completion, None, len_generated)
            logps_ema = policy(theta_ema, completion, None, len_generated)
            reward = self.get_reward(input_ids, completion)
            kl_div = logps_theta - logps_ema

            # theta update
            reward_beta = reward - self.beta * kl_div
            loss = (reward_beta * logps_theta).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # theta_ema update
            with torch.no_grad():
                for param, param_ema in zip(
                    theta.parameters(), theta_ema.parameters()
                ):
                    if param_ema.requires_grad:
                        param_ema.copy_(
                            (1 - self.mu) * param_ema + self.mu * param
                        )

            # log results
            for key, value in zip(
                ["logps_theta", "logps_ema", "reward", "reward_beta", "loss"],
                [logps_theta, logps_ema, reward, reward_beta, loss],
            ):
                loss_history[key].append(value.detach().cpu())

        for key in loss_history:
            loss_history[key] = torch.stack(loss_history[key]).mean()
        print(loss_history)

        loss_history["prompt_sample"] = self.sft_tokenizer.batch_decode(
            input_ids
        )
        loss_history["completion_sample"] = self.sft_tokenizer.batch_decode(
            completion
        )
        print(loss_history["completion_sample"])
        # wandb.log(loss_history)
        return theta

    def training_run(self):

        # for m in M gather thetas
        thetas = []
        for m in trange(self.num_runs, desc="Going through runs"):
            thetas.append(self.training_step())
        theta_params = [theta.parameters() for theta in thetas]

        # slerp merge
        theta_slerp = copy.deepcopy(self.sft_model)
        with torch.no_grad():
            for param in zip(theta_slerp.parameters(), *theta_params):
                if param[0].requires_grad:
                    param[0].copy_(
                        slerp(param[0], param[1:], 1 / self.num_runs).detach()
                    )

        # theta_sft update
        with torch.no_grad():
            for param, param_slerp in zip(
                self.sft_model.parameters(), theta_slerp.parameters()
            ):
                if param.requires_grad:
                    param.copy_((1 - self.eta) * param + self.eta * param_slerp)

    def training_iterations(self):

        # for i in I
        self.weights = []
        for i in trange(self.num_iterations, desc="Going through iterations"):
            self.training_run()
            self.weights.append(copy.deepcopy(self.sft_model))

        # store for pareto front
        return self.weights

    def get_reward(self, prompt: torch.Tensor, response: torch.Tensor):
        return (
            -torch.stack(
                [
                    self.reward_model(
                        **self.reward_tokenizer(
                            self.sft_tokenizer.batch_decode(input),
                            padding=True,
                            return_tensors="pt",
                        )
                    ).logits.detach()
                    for input in [prompt, response]
                ]
            )
            .mean(dim=2)
            .softmax(0)[1, :]
        )
