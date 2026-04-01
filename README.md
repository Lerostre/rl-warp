## Report <a name="report"></a>

This report covers the implementation of the WARP method from the paper [WARP: On the Benefits of Weight Averaged Rewarded Policies](https://arxiv.org/pdf/2406.16768). The project explores how to make a model generate texts under a specific condition and how to train it with WARP. The task is split into three parts: reward model training, WARP implementation, and hyperparameter analysis. All three parts are documented in the notebooks, although most of the logic is wrapped into scripts located in the `/warp` directory.

I did not manage to fully debug everything by the deadline. Missing deadlines is obviously not acceptable in production work. Still, I pushed the project as far as I could and tried to finish at least a meaningful subset of the work. In the end, the paper itself raises a number of questions, and a significant amount of time went into resolving those; in a real production task, that overhead would likely be much smaller.

### How to run

All project dependencies are listed in `poetry.lock`, so the environment can be recreated manually from there. The most important scripts are also duplicated in the notebooks; look for cells starting with `!poetry run`, as they launch the full training pipeline from scratch. A simpler option is to clone the repository, install `poetry`, and then install the dependencies with the following commands:

```bash
git clone https://github.com/Lerostre/rl-warp.git
cd rl-warp
pip install poetry
poetry lock --no-update
poetry install
```

Running the project in Colab requires a few extra steps. An example is available in `/notebooks`:

```bash
!git clone https://github.com/Lerostre/rl-warp.git
%cd rl-warp
!git switch master
```

```bash
!pip install -q poetry
!poetry config virtualenvs.in-project true
!poetry lock --no-update
!poetry install
```

### Level 1. Reward modelling

The first part of the task was to train a reward model. Strictly speaking, the assignment did not specify how to do this, but since the broader setup is RL-based, I decided to approach it from that angle. The dataset had to be constructed from `chosen-rejected` pairs, for example:

| chosen | rejected |
| --- | --- |
| this movie is awful | I've never seen anything better in my life |

The next step is to train a model in an RLHF-style setup to decide which text in the pair is better. The problem is that such a model does not directly learn how to assign a scalar reward to a single text; as far as I understand, it still needs a reference.

This matters because in the paper reward is written as \(r(x, y)\), where \(x\) is a prompt and \(y\) is its continuation. This can be interpreted in two different ways. One interpretation is that reward is assigned to the full prompt-continuation pair, in which case a standard classification model such as `distilbert-imdb` would be sufficient. The other interpretation is that reward is relative, comparing the original prompt and its continuation; in that case, `RewardTrainer` is a more natural choice, and this is also the interpretation used in the paper.

For estimating relative reward, I reused the function from `RewardTrainer`:

```python
# Stack accepted against rejected, mean over logits
# and softmax to get preferences between accepted and rejected to sum to 1
logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T
```

In the end, `RewardTrainer` does train, but it often becomes overly confident, producing very large probabilities. One additional observation is that reward in the paper is negative.

### Level 2. WARP

This part was significantly more difficult. WARP depends on several components, and any of them could contain mistakes:

1. **SLERP**. This part seems mostly correct. Possible failure points are an incorrect implementation for the multidimensional case or for the matrix-valued case, but overall it looks reasonably sound.

2. **Policy**. This part is conceptually less clear. I interpreted policy as the probability of the entire generated sentence given a prompt. The values I obtained were similar to what I had seen inside the trainer, so I am relatively confident here, though not completely certain. I computed it through cross-entropy, since this has to be evaluated for the model directly and cannot be recovered from generation alone.

3. **LIMA and other updates**. This part was implemented somewhat heuristically, but the manual updates appear to behave correctly. I verified that the parameters change and that they change by the intended amount.

4. **KL divergence**. This is where I have the strongest doubts. First, the regularization term used in the paper does not really look like a divergence. I checked other sources on this form of regularization, and the same expression appears there as well. The issue is that KL divergence is unbounded, so the parameter \(\beta\) does not necessarily make it easy to control. Second, it seems to become negative if I swap the arguments, which strongly suggests that something in my implementation or interpretation is wrong. This is the part where I believe I am most likely mistaken. In practice, to avoid letting this dominate the behavior, I set \(\beta = 0\) everywhere.

5. **Reward**. The reward model itself is also questionable. RLHF seems to provide reward only in comparison to another sample, unless I misunderstood the setup. The fact that the paper uses negative reward also makes the formulation harder to interpret. It is also unclear whether reward is bounded. The paper seems to suggest that it is; with a softmax-based formulation I can enforce that, but I am not sure whether that is actually the correct thing to do. This could easily be another source of error.

One more important deviation from the paper is that I used only 50 examples instead of 100, simply because I ran out of time. Most of the time went into debugging, but I still was not able to identify the critical error. In the end, WARP does train and the rewards move in the desired direction, but the generated texts are often nonsensical. For example, they sometimes repeat a single word or the same phrase over and over, such as “I was very disappointed.” That may be expected behavior, since it looks like an easy way for the model to maximize the desired negative reward.

### Level 3. Interpretation

I then tried to reproduce at least one of the plots from the paper, and I chose the parameter \(\eta\). The interesting part is that, similarly to DPO with different divergences, there appears to be a trade-off between KL and reward, which suggests that the method can in principle be tuned quite precisely for a specific objective.

This part was not successful either. The most plausible explanation is simply the limited number of experiments: I cannot claim that the estimates are robust, because both the number of examples and the number of trained models were too small.
