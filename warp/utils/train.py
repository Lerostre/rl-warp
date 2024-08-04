import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from typing import Optional, Sequence


def policy(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    len_generated: Optional[int] = None,
):

    logits = model(input_ids).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    policy = -F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, reduction="none"
    )
    if len_generated is not None:
        policy = policy[:, -len_generated:]

    return policy.sum(-1)


def slerp(
    theta: torch.Tensor, thetas: Sequence[torch.Tensor], lamb: float
) -> torch.Tensor:

    thetas = list(thetas)
    if len(thetas[0].shape) == 1:
        thetas = [theta.unsqueeze(0) for theta in thetas]

    for i in range(len(thetas) - 1):

        delta_1 = thetas[i] - theta
        delta_2 = thetas[i + 1] - theta

        omega = (
            torch.einsum(
                "ij, ij -> i",
                delta_1 / delta_1.norm(p=2, dim=1, keepdim=True),
                delta_2 / delta_2.norm(p=2, dim=1, keepdim=True),
            )
            .unsqueeze(-1)
            .arccos()
        )

        thetas[i + 1] = (
            torch.sin((1 - lamb) * omega) / torch.sin(omega)
        ) * delta_1 + (torch.sin(lamb * omega) / torch.sin(omega)) * delta_2

    return theta + thetas[-1]
