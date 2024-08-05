import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from typing import Optional, Sequence


def policy(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    len_generated: Optional[int] = None,
) -> torch.Tensor:
    logits = model(input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    policy = -F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, reduction="none"
    )
    if len_generated is not None:
        policy = policy[:, -len_generated:]

    return policy.sum(-1)


def slerp(
    theta_init: torch.Tensor, thetas: Sequence[torch.Tensor], lamb: float
) -> torch.Tensor:

    # avoid 1D mismatch
    thetas = list(thetas)
    if len(theta_init.shape) == 1:
        thetas = [theta.unsqueeze(0) for theta in thetas]

    for i in range(len(thetas) - 1):

        delta_1 = thetas[i] - theta_init
        delta_2 = thetas[i + 1] - theta_init

        omega = (
            F.cosine_similarity(delta_1, delta_2)
            .clamp(-1, 1)  # values close to 1 or -1 produce nans
            .acos()
            .unsqueeze(-1)
        )

        thetas[i + 1] = (
            torch.sin((1 - lamb) * omega) / torch.sin(omega) * delta_1
            + (torch.sin(lamb * omega) / torch.sin(omega)) * delta_2
        )

    if thetas[-1].shape[0] == 1:
        thetas[-1] = thetas[-1].squeeze(0)
    return theta_init + thetas[-1]
