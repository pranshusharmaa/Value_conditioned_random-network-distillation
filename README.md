# Value-Conditioned Random Network Distillation (VC-RND)

A value-aware extension of [Random Network Distillation](https://github.com/openai/random-network-distillation). The intrinsic reward is scaled by deviation from a running value baseline so curiosity stays where the value signal is informative and shrinks where it isn't.

**Paper:** [`docs/CSC415_Project_Proposal.pdf`](docs/CSC415_Project_Proposal.pdf)
**Authors:** Hassan Ghouri, Pranshu Sharma — CSC415, University of Toronto Mississauga.

> **About this repo.** This is a fork of [openai/random-network-distillation](https://github.com/openai/random-network-distillation). OpenAI's original TF1+MPI codebase remains at the repo root. Our modifications are in `ppo_agent_vcrnd_final.py` (the modified PPO agent) and the `minigrid/`, `notebooks/`, and `docs/` folders.

---

## TL;DR

Standard RND rewards novelty regardless of task value. Following VCSE (Kim et al., 2023), we condition the intrinsic bonus on a value deviation $\Delta V_t = |V(s_t) - \bar V_t|$, where $\bar V_t$ is an EMA of $V(s_t)$. The final reward is

$$r_t^{\text{VC-RND}} = r_t^{\text{RND}} \cdot g(V(s_t))$$

with two scaling variants tested:

- **Inverse linear:** $g(V(s_t)) = 1 / (1 + \lambda \Delta V_t)$
- **Sigmoid gating:** $g(V(s_t)) = \sigma(-\lambda \Delta V_t)$

Evaluated on three DeepMind Control Suite tasks (Cartpole Swingup Sparse, Walker Stand, Cheetah Run) and two MiniGrid tasks (FourRooms, MultiRoom-N6) against PPO, PPO+RND, and PPO+RND+Replay (Sovrano 2019).

---

## Headline results

### DeepMind Control Suite — final mean episodic return

| Environment | PPO | PPO+RND | PPO+RND+Replay | **VC-RND Linear** | **VC-RND Sigmoid** |
|---|---:|---:|---:|---:|---:|
| Cartpole Swingup Sparse | 0.00 | 2.02 | 1.53 | **2.51** | — |
| Walker Stand | 274.99 | 338.57 | **369.88** | 366.02 | 339.17 |
| Cheetah Run | **182.12** | 49.62 | 93.21 | 93.81 | 94.99 |

Walker Stand is the clearest win — VC-RND Linear nearly matches the much heavier replay baseline. Cheetah is a case where curiosity hurts in general, but VC-RND hurts *less* than vanilla RND. Cartpole has reward signal so sparse that value-conditioning has nothing to act on.

### MiniGrid FourRooms (1M steps)

| Algorithm | Last | Last-10% Avg |
|---|---:|---:|
| PPO | **0.339** | **0.320 ± 0.051** |
| PPO+RND | 0.056 | 0.029 ± 0.015 |
| PPO+RND+Replay | 0.018 | 0.020 ± 0.006 |
| VC-RND Linear | 0.018 | 0.030 ± 0.012 |
| VC-RND Sigmoid | 0.023 | 0.036 ± 0.019 |

MiniGrid's partial pixel observations make RND prediction error noisy; all RND variants underperform plain PPO. VC-RND still produces measurably tighter intrinsic-reward distributions than vanilla RND (see paper Table 4).

---

## Repository layout

```
.                                # OpenAI's original RND code lives at the root
├── ppo_agent.py                 # OpenAI's original (preserved for diff)
├── ppo_agent_vcrnd_final.py     # OUR modified version with VC-RND scaling
├── policies/                    # OpenAI's policies module
├── atari_wrappers.py            # OpenAI's
├── mpi_util.py                  # OpenAI's
├── ...                          # rest of OpenAI's TF1 + MPI codebase
│
├── minigrid/                    # OUR self-contained PyTorch implementation
│   └── run_minigrid_vcrnd.py
│
├── notebooks/                   # OUR experiment runs + plots
│   └── VC_RND_Experiments.ipynb
│
├── docs/                        # paper + supplementary material
│   └── CSC415_Project_Proposal.pdf
│
└── README.md                    # this file
```

The TF1 + MPI codebase at the root is OpenAI's original RND implementation. Our `ppo_agent_vcrnd_final.py` is a drop-in replacement for their `ppo_agent.py` with the VC-RND scaling added. The PyTorch implementation in `minigrid/` is a clean self-contained reimplementation used for the final experiments — that's the recommended entry point.

---

## Method details

### Reward normalization

Because RND prediction error drifts during training, we normalize using EMA statistics with $\alpha = 0.01$:

$$\mu_t = (1-\alpha)\mu_{t-1} + \alpha \tilde r_t^\text{RND}, \quad \sigma_t^2 = (1-\alpha)\sigma_{t-1}^2 + \alpha(\tilde r_t^\text{RND} - \mu_t)^2, \quad r_t^\text{RND} = \frac{\tilde r_t^\text{RND}}{\sigma_t + \epsilon}$$

### Value baseline

The running value mean tracks $V(s_t)$ via EMA with $\beta$ as a tunable parameter:

$$\bar V_t = (1-\beta)\bar V_{t-1} + \beta V(s_t)$$

### Why two scaling variants

- **Inverse linear** is the smoother of the two — it never hits zero, so curiosity is dampened but not killed even when $\Delta V$ is large. Generally the more stable option in our experiments.
- **Sigmoid gating** is more aggressive — once $\Delta V$ exceeds the inflection point, intrinsic reward collapses sharply. Useful in dense-reward environments (Cheetah) but can over-suppress curiosity in sparse settings (MultiRoom).

The DMC ablations show inverse linear wins in 2 of 3 environments. The sigmoid variant produced the smallest Walker explained-variance instability (0.764 vs 0.820 for linear), suggesting it stabilizes the critic at the cost of sometimes-too-strong curiosity dampening.

---

## Running the experiments

### MiniGrid (PyTorch — recommended)

```bash
cd minigrid
pip install torch numpy gymnasium minigrid opencv-python-headless scikit-learn

# all six conditions on FourRooms
python run_minigrid_vcrnd.py --condition all --env MiniGrid-FourRooms-v0

# single condition on MultiRoom
python run_minigrid_vcrnd.py --condition ppo_vcrnd --env MiniGrid-MultiRoom-N6-v0
```

Conditions:

| Flag | Description |
|---|---|
| `ppo` | Vanilla PPO, no intrinsic reward |
| `ppo_rnd` | PPO + standard RND |
| `ppo_rnd_replay` | PPO + RND + replay buffer (Sovrano 2019) |
| `ppo_vcrnd` | PPO + VC-RND, inverse linear scaling (Eq. 11) |
| `ppo_vcrnd_sigmoid` | PPO + VC-RND, sigmoid gating (Eq. 12) |
| `ppo_vcse` | PPO + VCSE-v2 baseline w/ frozen encoder + cached kNN |

### Atari / DMC (TF1 — using OpenAI's original setup)

To run our modified PPO agent in OpenAI's original codebase, swap in `ppo_agent_vcrnd_final.py` for `ppo_agent.py` and follow OpenAI's setup:

```bash
# follow OpenAI's original install instructions
# (mpi4py, tensorflow 1.x, baselines, etc.)
mv ppo_agent.py ppo_agent_original.py
cp ppo_agent_vcrnd_final.py ppo_agent.py
mpiexec -np 32 python run_atari.py --env_id MontezumaRevengeNoFrameskip-v4
```

Note: TF1 environment setup is non-trivial in 2026. The PyTorch implementation in `minigrid/` is the recommended path unless you specifically need Atari results.

### Logged metrics

- Mean episodic return
- Time-to-first-reward
- Sample efficiency (`ev_ext`)
- Intrinsic reward distribution (raw, normalized, final)
- Correlation between $r_\text{int}$ and $V(s_t)$
- VC scale mean, $\Delta V$ mean, running value mean

---

## Key findings

1. **Value-conditioning is environment-dependent.** It helps most in environments with structured but variable extrinsic reward (Walker Stand). It does little when reward is too sparse to inform $V$ (Cartpole, MiniGrid FourRooms) or when curiosity itself is harmful (Cheetah, where PPO alone wins anyway).

2. **Inverse linear scaling generalizes better than sigmoid.** Smoother dampening preserves more useful exploration than the sharp gating of the sigmoid form.

3. **Even when return doesn't improve, exploration becomes more controlled.** MiniGrid MultiRoom shows VC-RND collapsing the intrinsic-reward distribution from $\sim$6.0 spread to $\sim$0.5 spread — agents are exploring more deliberately, just not yet finding the goal.

4. **MiniGrid's partial-observation pixel input is a weak setting for RND-based methods overall.** All RND variants underperform plain PPO. This isn't a VC-RND specific issue; the prediction error is noisy because the encoder can't distinguish meaningful novelty from observation noise.

---

## Limitations (also see Section 7 of the paper)

- **3 seeds per configuration.** Statistical robustness is limited; small differences (e.g. Cartpole margins) may reflect seed variance.
- **No systematic hyperparameter sweep.** $\lambda$ (scaling rate), $\beta$ (value EMA), and the intrinsic coefficient interact non-trivially. FourRooms needed coefficient 0.02; MultiRoom needed 1.0.
- **Limited environment coverage.** No Atari, no procedurally-generated tasks (ProcGen), no high-dimensional visual control.
- **Single global $\bar V_t$.** Assumes unimodal value distribution — may break in environments with multiple disjoint reward regions.
- **Fixed scaling forms.** Both $g$ variants are chosen a priori; a learned scaling function is left for future work.

---

## Citation

```bibtex
@misc{ghouri2026vcrnd,
  title  = {Value-Conditioned Intrinsic Scaling (RND + VCSE)},
  author = {Ghouri, Hassan and Sharma, Pranshu},
  year   = {2026},
  note   = {CSC415 Project, University of Toronto Mississauga}
}
```

## References

Full reference list in `docs/CSC415_Project_Proposal.pdf`. Key works:

- Burda et al., *Exploration by Random Network Distillation*, arXiv:1810.12894 (2018) — the original RND paper, code in this repo.
- Kim et al., *Accelerating RL with Value-Conditional State Entropy Exploration* (VCSE), arXiv:2305.19476 (2023).
- Sovrano, *Combining Experience Replay with Exploration by RND*, arXiv:1905.07579 (2019).
- Pathak et al., *Curiosity-Driven Exploration by Self-Supervised Prediction* (ICM), arXiv:1705.05363 (2017).

---

## Team contributions

**Pranshu Sharma:** Literature review, Limitations & Future Work, MiniGrid experiments and analysis.
**Hassan Ghouri:** Abstract, Introduction, Methodologies, DMC experiments and analysis.

---

## License

MIT — see [LICENSE](LICENSE). The TF1 codebase at the root is OpenAI's original RND implementation, also MIT-licensed; see https://github.com/openai/random-network-distillation.
