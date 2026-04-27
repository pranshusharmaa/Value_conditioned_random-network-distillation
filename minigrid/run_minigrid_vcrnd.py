#!/usr/bin/env python3
"""
run_minigrid_vcrnd.py  (PyTorch — Colab-ready)
-----------------------------------------------
Runs all six experimental conditions on MiniGrid:

  ppo               -- PPO only (no intrinsic reward)
  ppo_rnd           -- PPO + RND baseline
  ppo_rnd_replay    -- PPO + RND + Replay Buffer (Sovrano 2019)
  ppo_vcrnd         -- PPO + VC-RND, inverse scaling (Eq. 11)
  ppo_vcrnd_sigmoid -- PPO + VC-RND, sigmoid scaling (Eq. 12, fixed 2x)
  ppo_vcse          -- PPO + VCSE-v2 baseline with frozen encoder + cached kNN index

Evaluation metrics logged (Section 3.4 of proposal):
  - Mean episodic return
  - Time-to-first-reward
  - Sample efficiency (ev_ext)
  - Intrinsic reward distribution (raw, norm, final)
  - Correlation between r_int and V(s_t)
  - VC scale mean, delta_v mean, running value mean

Usage:
  python run_minigrid_vcrnd.py --condition all --env MiniGrid-FourRooms-v0
  python run_minigrid_vcrnd.py --condition ppo_vcrnd --env MiniGrid-MultiRoom-N6-v0

Install:
  pip install minigrid gymnasium opencv-python-headless torch numpy
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    NearestNeighbors = None
    SKLEARN_AVAILABLE = False

try:
    import minigrid  # noqa: F401 — registers MiniGrid envs with gym
except ImportError:
    raise ImportError("Run:  pip install minigrid")


# ══════════════════════════════════════════════════════════════════
#  Condition configuration
# ══════════════════════════════════════════════════════════════════

CONDITION_OVERRIDES = {
    "ppo": dict(
        int_coeff=0.0,
        intrinsic_type="none",
        use_replay_buffer=False,
        use_vc_rnd=False,
    ),
    "ppo_rnd": dict(
        intrinsic_type="rnd",
        use_replay_buffer=False,
        use_vc_rnd=False,
    ),
    "ppo_rnd_replay": dict(
        intrinsic_type="rnd",
        use_replay_buffer=True,
        use_vc_rnd=False,
    ),
    "ppo_vcrnd": dict(
        intrinsic_type="rnd",
        use_replay_buffer=False,
        use_vc_rnd=True,
    ),
    "ppo_vcrnd_sigmoid": dict(
        intrinsic_type="rnd",
        use_replay_buffer=False,
        use_vc_rnd=True,
    ),
    "ppo_vcse": dict(
        intrinsic_type="vcse",
        use_replay_buffer=False,
        use_vc_rnd=False,
    ),
}


# ══════════════════════════════════════════════════════════════════
#  MiniGrid environment wrapper + VecEnv
# ══════════════════════════════════════════════════════════════════

class MiniGridWrapper(gym.Wrapper):
    """Converts MiniGrid dict-obs to a resized uint8 RGB image and
    accumulates episode reward properly."""

    def __init__(self, env, obs_size: int = 56):
        super().__init__(env)
        self.obs_size = obs_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size, obs_size, 3), dtype=np.uint8,
        )
        self._ep_rew = 0.0
        self._ep_len = 0

    def _process_obs(self, obs):
        img = obs["image"] if isinstance(obs, dict) else obs
        if img.shape[0] != self.obs_size or img.shape[1] != self.obs_size:
            img = cv2.resize(img, (self.obs_size, self.obs_size),
                             interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def reset(self, **kwargs):
        self._ep_rew = 0.0
        self._ep_len = 0
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self._process_obs(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:          # gym ≥0.26
            obs, rew, term, trunc, info = result
            done = term or trunc
        else:                          # gym <0.26
            obs, rew, done, info = result

        self._ep_rew += float(rew)
        self._ep_len += 1

        if done:
            info["episode"] = {
                "r": self._ep_rew,
                "l": self._ep_len,
            }
        return self._process_obs(obs), float(rew), done, info


class MiniGridVecEnv:
    """Minimal synchronous vectorised environment for MiniGrid."""

    def __init__(self, env_fns):
        self.envs     = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space      = self.envs[0].action_space

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs_l, rew_l, done_l, info_l = [], [], [], []
        for i, (env, ac) in enumerate(zip(self.envs, actions)):
            obs, rew, done, info = env.step(int(ac))
            if done:
                obs = env.reset()
            obs_l.append(obs); rew_l.append(rew)
            done_l.append(done); info_l.append(info)
        return (
            np.stack(obs_l),
            np.array(rew_l, np.float32),
            np.array(done_l, bool),
            info_l,
        )

    def close(self):
        for e in self.envs:
            e.close()


def make_minigrid_venv(env_id, num_env, seed, max_episode_steps, obs_size):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, render_mode="rgb_array")
            if max_episode_steps > 0:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

            env = RGBImgPartialObsWrapper(env, tile_size=8)
            env = ImgObsWrapper(env)
            env = RGBResizeWrapper(env, obs_size=obs_size)

            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return env
        return _thunk
    return MiniGridVecEnv([make_env(i) for i in range(num_env)])


class RGBResizeWrapper(gym.Wrapper):
    def __init__(self, env, obs_size: int = 56):
        super().__init__(env)
        self.obs_size = obs_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size, obs_size, 3), dtype=np.uint8
        )
        self._ep_rew = 0.0
        self._ep_len = 0

    def _process_obs(self, obs):
        img = obs
        if img.shape[0] != self.obs_size or img.shape[1] != self.obs_size:
            img = cv2.resize(img, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def reset(self, **kwargs):
        self._ep_rew = 0.0
        self._ep_len = 0
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self._process_obs(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, rew, term, trunc, info = result
            done = term or trunc
        else:
            obs, rew, done, info = result

        self._ep_rew += float(rew)
        self._ep_len += 1

        if done:
            info["episode"] = {"r": self._ep_rew, "l": self._ep_len}

        return self._process_obs(obs), float(rew), done, info

# ══════════════════════════════════════════════════════════════════
#  Networks  (IMPALA CNN for image observations, discrete actions)
# ══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Pre-activation residual block: ReLU → Conv → ReLU → Conv + skip."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class IMPALABlock(nn.Module):
    """One IMPALA conv block: Conv → MaxPool → 2x ResidualBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class IMPALACNN(nn.Module):
    """IMPALA-style residual CNN trunk (Espeholt et al., 2018).
    Much deeper than NatureCNN — 15 conv layers vs 3 — with residual
    connections for stable gradient flow. Standard choice for MiniGrid
    and Procgen benchmarks.

    Input: (B, H, W, C) uint8 or float, channels-last.
    Output: (B, out_dim) feature vector."""

    def __init__(self, obs_shape, out_dim=256):
        super().__init__()
        c = obs_shape[2]   # channels-last from env
        self.blocks = nn.Sequential(
            IMPALABlock(c, 16),
            IMPALABlock(16, 32),
            IMPALABlock(32, 32),
        )
        # Compute flat size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, c, obs_shape[0], obs_shape[1])
            flat_size = self.blocks(dummy).numel()
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # (B, H, W, C) uint8/float → (B, C, H, W) float [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.dim() == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)
        return self.head(self.blocks(x))


class ActorCriticCNN(nn.Module):
    """IMPALA CNN + LSTM actor-critic with dual value heads for discrete actions.
    The LSTM provides temporal memory so the agent can remember visited rooms
    under partial observability."""

    def __init__(self, obs_shape, n_actions, hidden=256, lstm_size=256):
        super().__init__()
        self.trunk     = IMPALACNN(obs_shape, out_dim=hidden)
        self.lstm_size = lstm_size
        self.lstm      = nn.LSTM(hidden, lstm_size, batch_first=True)
        self.policy    = nn.Linear(lstm_size, n_actions)
        self.vf_int    = nn.Linear(lstm_size, 1)
        self.vf_ext    = nn.Linear(lstm_size, 1)
        # Orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        # LSTM init
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def initial_state(self, batch_size, device='cpu'):
        """Return zero (h, c) for LSTM."""
        return (torch.zeros(1, batch_size, self.lstm_size, device=device),
                torch.zeros(1, batch_size, self.lstm_size, device=device))

    def forward(self, obs, hx, masks=None):
        """Forward pass through CNN + LSTM.
        obs:   (B, T, H, W, C) for sequences or (B, H, W, C) for single step
        hx:    tuple of (h, c) each (1, B, lstm_size)
        masks: (B, T) done masks — 0.0 where episode ended, 1.0 otherwise
        Returns: dist, vf_int, vf_ext, new_hx
        """
        single_step = (obs.dim() == 4)
        if single_step:
            obs = obs.unsqueeze(1)  # (B, 1, H, W, C)

        B, T = obs.shape[0], obs.shape[1]

        # CNN: flatten batch and time, then reshape back
        cnn_out = self.trunk(obs.reshape(B * T, *obs.shape[2:]))  # (B*T, hidden)
        cnn_out = cnn_out.reshape(B, T, -1)                       # (B, T, hidden)

        # Zero out LSTM state where episodes ended (between steps)
        if masks is not None and T > 1:
            lstm_out_list = []
            for t in range(T):
                # Reset hidden state at episode boundaries
                mask_t = masks[:, t].unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
                hx = (hx[0] * mask_t, hx[1] * mask_t)
                out_t, hx = self.lstm(cnn_out[:, t:t+1, :], hx)
                lstm_out_list.append(out_t)
            lstm_out = torch.cat(lstm_out_list, dim=1)  # (B, T, lstm_size)
        else:
            lstm_out, hx = self.lstm(cnn_out, hx)       # (B, T, lstm_size)

        lstm_flat = lstm_out.reshape(B * T, -1)
        dist   = Categorical(logits=self.policy(lstm_flat))
        vf_i   = self.vf_int(lstm_flat).squeeze(-1)
        vf_e   = self.vf_ext(lstm_flat).squeeze(-1)

        if single_step:
            # Squeeze time dim back out
            return dist, vf_i, vf_e, hx
        else:
            return dist, vf_i.reshape(B, T), vf_e.reshape(B, T), hx

    @torch.no_grad()
    def act(self, obs, hx):
        """Single-step action selection during rollout."""
        dist, vi, ve, hx_new = self(obs, hx)
        ac   = dist.sample()
        logp = dist.log_prob(ac)
        return ac, vi, ve, logp, hx_new


class RNDNetworkCNN(nn.Module):
    """CNN-based RND: fixed random target + trainable predictor."""

    def __init__(self, obs_shape, rnd_out_dim=512, hidden=256):
        super().__init__()
        self.target    = IMPALACNN(obs_shape, out_dim=rnd_out_dim)
        self.predictor = nn.Sequential(
            IMPALACNN(obs_shape, out_dim=hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, rnd_out_dim),
        )
        for p in self.target.parameters():
            p.requires_grad_(False)

    def intrinsic_reward(self, obs_norm):
        with torch.no_grad():
            tgt = self.target(obs_norm)
        return ((self.predictor(obs_norm) - tgt) ** 2).mean(dim=-1)

    def predictor_loss(self, obs_norm):
        return ((self.predictor(obs_norm) - self.target(obs_norm).detach()) ** 2).mean()




class VCSEModule:
    """VCSE-style kNN entropy baseline with a frozen encoder and cached index.

    Key design choices for long runs:
      - features come from a separate frozen random encoder, not the live policy trunk
      - memory/index are rebuilt once per PPO rollout, not every env step
      - feature/value normalisation stats are cached between rebuilds

    Novelty is computed in the joint space
      [normalized_feature, value_scale * normalized_value]
    using mean distance to the k nearest neighbours in memory.
    """

    def __init__(self, feature_dim, k=10, buffer_capacity=50000,
                 value_scale=0.1, eps=1e-6, algorithm="auto"):
        self.feature_dim = feature_dim
        self.k = k
        self.buffer_capacity = buffer_capacity
        self.value_scale = value_scale
        self.eps = eps
        self.algorithm = algorithm

        self.features = deque(maxlen=buffer_capacity)
        self.values = deque(maxlen=buffer_capacity)

        self.feat_std = np.ones((1, feature_dim), dtype=np.float32)
        self.val_std = np.float32(1.0)
        self.memory_joint = None
        self.nn_index = None
        self.index_ready = False

    def __len__(self):
        return len(self.features)

    def add_batch(self, feats, values):
        feats = np.asarray(feats, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        for f, v in zip(feats, values):
            self.features.append(f.astype(np.float32, copy=True))
            self.values.append(np.float32(v))

    def update_stats_and_index(self):
        if len(self.features) < max(2, self.k):
            self.feat_std = np.ones((1, self.feature_dim), dtype=np.float32)
            self.val_std = np.float32(1.0)
            self.memory_joint = None
            self.nn_index = None
            self.index_ready = False
            return

        mem_feats = np.asarray(self.features, dtype=np.float32)
        mem_vals = np.asarray(self.values, dtype=np.float32).reshape(-1)

        self.feat_std = mem_feats.std(axis=0, keepdims=True).astype(np.float32) + self.eps
        self.val_std = np.float32(mem_vals.std() + self.eps)

        mem_feats_n = mem_feats / self.feat_std
        mem_vals_n = mem_vals / self.val_std
        self.memory_joint = np.concatenate(
            [mem_feats_n, self.value_scale * mem_vals_n[:, None]], axis=1
        ).astype(np.float32)

        if SKLEARN_AVAILABLE and self.memory_joint.shape[0] >= self.k:
            self.nn_index = NearestNeighbors(
                n_neighbors=min(self.k, self.memory_joint.shape[0]),
                algorithm=self.algorithm,
                metric="euclidean",
            )
            self.nn_index.fit(self.memory_joint)
        else:
            self.nn_index = None
        self.index_ready = True

    def compute_reward(self, feats, values):
        feats = np.asarray(feats, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        n = feats.shape[0]

        if not self.index_ready or self.memory_joint is None:
            return np.ones(n, dtype=np.float32)

        query = np.concatenate(
            [feats / self.feat_std, self.value_scale * (values / self.val_std)[:, None]],
            axis=1,
        ).astype(np.float32)

        if self.nn_index is not None:
            dists, _ = self.nn_index.kneighbors(
                query, n_neighbors=min(self.k, self.memory_joint.shape[0]), return_distance=True
            )
            return dists.mean(axis=1).astype(np.float32)

        # Fallback exact search when sklearn is unavailable
        dists = np.sqrt(
            np.sum((query[:, None, :] - self.memory_joint[None, :, :]) ** 2, axis=-1) + self.eps
        )
        k_eff = min(self.k, dists.shape[1])
        knn = np.partition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
        return knn.mean(axis=1).astype(np.float32)

# ══════════════════════════════════════════════════════════════════
#  Running statistics
# ══════════════════════════════════════════════════════════════════

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean  = np.zeros(shape, np.float64)
        self.var   = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape) if self.mean.shape else x.ravel()
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d          = bm - self.mean
        tot        = self.count + bc
        self.mean  = self.mean + d * bc / tot
        self.var   = (self.var * self.count + bv * bc + d**2 * self.count * bc / tot) / tot
        self.count = tot


# ══════════════════════════════════════════════════════════════════
#  Replay buffer
# ══════════════════════════════════════════════════════════════════

class RolloutReplayBuffer:
    """Circular buffer of past rollout observations — only the RND
    predictor is updated from replay (Sovrano 2019)."""

    def __init__(self, capacity, seg_size, obs_shape):
        self.capacity  = capacity
        self.seg_size  = seg_size
        self.obs_shape = obs_shape
        self._buf  = np.zeros((capacity, seg_size, *obs_shape), dtype=np.uint8)
        self._ptr  = 0
        self._full = False

    @property
    def n_segments(self):
        return self.capacity if self._full else self._ptr

    @property
    def n_transitions(self):
        return self.n_segments * self.seg_size

    def push(self, obs_seg):
        self._buf[self._ptr] = obs_seg.reshape(self.seg_size, *self.obs_shape)
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0:
            self._full = True

    def sample(self, n):
        flat = self._buf[:self.n_segments].reshape(self.n_transitions, *self.obs_shape)
        idxs = np.random.randint(0, self.n_transitions, size=n)
        return flat[idxs]


# ══════════════════════════════════════════════════════════════════
#  VC-RND scaling  (Eq. 9-13 from proposal)
# ══════════════════════════════════════════════════════════════════

def apply_value_conditioning(rews_int_norm, vpreds_ext, running_value_mean,
                             vc_rnd_beta, vc_rnd_lambda, vc_rnd_mode, vc_rnd_min_scale):
    values       = vpreds_ext.astype(np.float32)
    values_tm    = values.T                        # [nsteps, nenvs]
    running_mean = float(running_value_mean)
    trace_tm     = np.zeros_like(values_tm)

    for t in range(values_tm.shape[0]):
        running_mean = (
            (1.0 - vc_rnd_beta) * running_mean
            + vc_rnd_beta * float(np.mean(values_tm[t]))
        )
        trace_tm[t] = running_mean

    delta_v = np.abs(values - trace_tm.T)

    if vc_rnd_mode == "inverse":
        scale = 1.0 / (1.0 + vc_rnd_lambda * delta_v)
    elif vc_rnd_mode == "sigmoid":
        scale = 2.0 / (1.0 + np.exp(vc_rnd_lambda * delta_v))
    else:
        raise ValueError(f"Unknown vc_rnd_mode: {vc_rnd_mode}")

    if vc_rnd_min_scale > 0.0:
        scale = np.maximum(scale, vc_rnd_min_scale)

    return (
        rews_int_norm * scale,
        scale.astype(np.float32),
        delta_v.astype(np.float32),
        float(running_mean),
    )


# ══════════════════════════════════════════════════════════════════
#  GAE
# ══════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, last_values, dones, gamma, lam, use_dones=True):
    nenvs, nsteps = rewards.shape
    advantages    = np.zeros_like(rewards)
    lastgae       = np.zeros(nenvs, np.float32)
    for t in reversed(range(nsteps)):
        nv    = values[:, t + 1] if t + 1 < nsteps else last_values
        nnd   = (1.0 - dones[:, t].astype(np.float32)) if use_dones else 1.0
        delta = rewards[:, t] + gamma * nv * nnd - values[:, t]
        advantages[:, t] = lastgae = delta + gamma * lam * nnd * lastgae
    return advantages, advantages + values


# ══════════════════════════════════════════════════════════════════
#  Training loop for one condition
# ══════════════════════════════════════════════════════════════════


def train_condition(args, condition: str, overrides: dict, log_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        intrinsic_type="rnd",
        use_replay_buffer=False,
        use_vc_rnd=bool(args.vc_rnd),
    )
    cfg.update(overrides)

    # Scaling mode is determined by condition name, not CLI flag,
    # so --condition all always runs both variants correctly.
    vc_rnd_mode = "sigmoid" if condition == "ppo_vcrnd_sigmoid" else "inverse"

    print(f"\n{'='*60}")
    print(f"  Condition        : {condition}")
    print(f"  Env              : {args.env}")
    print(f"  intrinsic_type   : {cfg['intrinsic_type']}")
    print(f"  use_replay_buffer: {cfg['use_replay_buffer']}")
    print(f"  use_vc_rnd       : {cfg['use_vc_rnd']}")
    print(f"  vc_rnd_mode      : {vc_rnd_mode if cfg['use_vc_rnd'] else 'n/a'}")
    print(f"  int_coeff        : {cfg['int_coeff']}")
    print(f"  device           : {device}")
    print(f"  log_dir          : {log_dir}")
    print(f"{'='*60}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Environment ──────────────────────────────────────
    venv = make_minigrid_venv(
        env_id=args.env,
        num_env=args.num_env,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        obs_size=args.obs_size,
    )
    obs_shape = venv.observation_space.shape   # (H, W, 3)
    n_actions = venv.action_space.n
    nenvs = args.num_env
    nsteps = args.nsteps
    eps = 1e-8

    # ── Networks ─────────────────────────────────────────
    ac = ActorCriticCNN(
        obs_shape, n_actions, hidden=args.hidden_size, lstm_size=args.hidden_size
    ).to(device)

    rnd = None
    vcse = None
    vcse_encoder = None
    if cfg["intrinsic_type"] == "rnd":
        rnd = RNDNetworkCNN(
            obs_shape, rnd_out_dim=args.rnd_out_dim, hidden=args.hidden_size
        ).to(device)
    elif cfg["intrinsic_type"] == "vcse":
        vcse = VCSEModule(
            feature_dim=args.hidden_size,
            k=args.vcse_k,
            buffer_capacity=args.vcse_buffer_capacity,
            value_scale=args.vcse_value_scale,
            eps=args.vcse_eps,
            algorithm=args.vcse_knn_algorithm,
        )
        vcse_encoder = IMPALACNN(obs_shape, out_dim=args.hidden_size).to(device)
        for p in vcse_encoder.parameters():
            p.requires_grad_(False)
        vcse_encoder.eval()

    opt_params = list(ac.parameters())
    if rnd is not None:
        opt_params += list(rnd.predictor.parameters())
    optimizer = optim.Adam(opt_params, lr=args.lr, eps=1e-5)

    pred_optimizer = optim.Adam(
        list(rnd.predictor.parameters()), lr=args.lr, eps=1e-5
    ) if (rnd is not None and cfg["use_replay_buffer"]) else None

    # ── Running stats ────────────────────────────────────
    obs_rms = RunningMeanStd(shape=obs_shape)
    int_rms = RunningMeanStd(shape=())
    vc_running_value_mean = 0.0

    # ── Replay buffer ────────────────────────────────────
    replay_buf = RolloutReplayBuffer(
        capacity=args.replay_buffer_capacity,
        seg_size=nenvs * nsteps,
        obs_shape=obs_shape,
    ) if cfg["use_replay_buffer"] else None

    # ── Logging ──────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "progress.csv")
    ep_rew_buf = deque(maxlen=100)
    ep_len_buf = deque(maxlen=100)
    t_start = time.time()
    first_reward_step = -1

    with open(log_path, "w") as f:
        f.write(
            "timestep,mean_ep_rew,mean_ep_len,"
            "mean_int_rew_raw,mean_int_rew_norm,mean_int_rew_final,"
            "vc_scale_mean,vc_delta_v_mean,vc_running_value_mean,"
            "replay_buf_size,first_reward_step,corr_int_v,"
            "pg_loss,vf_loss,rnd_loss,replay_rnd_loss,entropy,"
            "ev_int,ev_ext,tps\n"
        )

    # ── Rollout buffers ──────────────────────────────────
    buf_obs = np.zeros((nenvs, nsteps, *obs_shape), np.uint8)
    buf_acs = np.zeros((nenvs, nsteps), np.int64)
    buf_logps = np.zeros((nenvs, nsteps), np.float32)
    buf_rews_ext = np.zeros((nenvs, nsteps), np.float32)
    buf_rews_int = np.zeros((nenvs, nsteps), np.float32)
    buf_vint = np.zeros((nenvs, nsteps), np.float32)
    buf_vext = np.zeros((nenvs, nsteps), np.float32)
    buf_dones = np.zeros((nenvs, nsteps), bool)
    buf_masks = np.ones((nenvs, nsteps), np.float32)  # mask for current obs
    buf_vcse_feats = np.zeros((nenvs, nsteps, args.hidden_size), np.float32) if cfg["intrinsic_type"] == "vcse" else None

    obs = venv.reset()
    lstm_hx = ac.initial_state(nenvs, device=device)
    prev_masks = np.ones(nenvs, dtype=np.float32)

    # ── Warm up obs normalisation ────────────────────────
    print(f"[{condition}] Warming up observation normalisation ...")
    warmup_obs = [obs]
    for _ in range(50):
        rand_acs = np.random.randint(0, n_actions, size=nenvs)
        obs_next, _, _, _ = venv.step(rand_acs)
        warmup_obs.append(obs_next)
    obs_rms.update(np.concatenate(warmup_obs, axis=0).astype(np.float64))
    obs = venv.reset()
    lstm_hx = ac.initial_state(nenvs, device=device)
    prev_masks = np.ones(nenvs, dtype=np.float32)
    print(f"[{condition}] Warm-up complete.")

    total_steps = 0
    update_count = 0

    while total_steps < args.num_timesteps:

        # Save LSTM state at start of rollout for training
        init_lstm_hx = (lstm_hx[0].detach().clone(), lstm_hx[1].detach().clone())

        # ── Collect rollout ──────────────────────────────
        for t in range(nsteps):
            buf_masks[:, t] = prev_masks

            obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=device)
            with torch.no_grad():
                acs_t, vint_t, vext_t, logp_t, lstm_hx = ac.act(obs_t, lstm_hx)

                if cfg["intrinsic_type"] == "rnd":
                    obs_f = obs.astype(np.float64)
                    obs_norm = (
                        (obs_f - obs_rms.mean) / (np.sqrt(obs_rms.var) + eps)
                    ).astype(np.float32)
                    int_rew_t = rnd.intrinsic_reward(
                        torch.as_tensor(obs_norm, device=device)
                    ).cpu().numpy()

                elif cfg["intrinsic_type"] == "vcse":
                    feat_t = vcse_encoder(obs_t).cpu().numpy()
                    int_rew_t = vcse.compute_reward(
                        feat_t,
                        vext_t.cpu().numpy(),
                    )

                else:
                    feat_t = None
                    int_rew_t = np.zeros(nenvs, np.float32)

            acs_np = acs_t.cpu().numpy()
            next_obs, rews_ext, dones, infos = venv.step(acs_np)

            buf_obs[:, t] = obs
            buf_acs[:, t] = acs_np
            buf_logps[:, t] = logp_t.cpu().numpy()
            buf_rews_ext[:, t] = rews_ext
            buf_rews_int[:, t] = int_rew_t
            buf_vint[:, t] = vint_t.cpu().numpy()
            buf_vext[:, t] = vext_t.cpu().numpy()
            if buf_vcse_feats is not None:
                buf_vcse_feats[:, t] = feat_t
            buf_dones[:, t] = dones

            # Reset LSTM hidden state for envs that finished
            for i, done in enumerate(dones):
                if done:
                    lstm_hx[0][:, i, :] = 0.0
                    lstm_hx[1][:, i, :] = 0.0

            for info in infos:
                if "episode" in info:
                    ep_rew_buf.append(info["episode"]["r"])
                    ep_len_buf.append(info["episode"]["l"])

            prev_masks = 1.0 - dones.astype(np.float32)
            obs = next_obs
            total_steps += nenvs

        # ── Bootstrap ────────────────────────────────────
        obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=device)
        with torch.no_grad():
            _, last_vint, last_vext, _ = ac(obs_t, lstm_hx)
        last_vint = last_vint.cpu().numpy()
        last_vext = last_vext.cpu().numpy()

        # ── Update obs RMS ───────────────────────────────
        obs_rms.update(buf_obs.reshape(-1, *obs_shape).astype(np.float64))

        # ── Push to replay buffer ────────────────────────
        if replay_buf is not None:
            replay_buf.push(buf_obs)

        # ── Update VCSE memory/index once per rollout ───
        if vcse is not None:
            vcse.add_batch(
                buf_vcse_feats.reshape(-1, args.hidden_size),
                buf_vext.reshape(-1),
            )
            vcse.update_stats_and_index()

        # ── Normalise intrinsic rewards ──────────────────
        if cfg["intrinsic_type"] != "none":
            int_rms.update(buf_rews_int.ravel())
            rews_int_norm = buf_rews_int / (np.sqrt(int_rms.var) + eps)
        else:
            rews_int_norm = np.zeros_like(buf_rews_int)

        # ── VC-RND scaling (RND only) ────────────────────
        if cfg["intrinsic_type"] == "rnd" and cfg["use_vc_rnd"]:
            rews_int_final, vc_scale, vc_delta_v, vc_running_value_mean = apply_value_conditioning(
                rews_int_norm, buf_vext, vc_running_value_mean,
                vc_rnd_beta=args.vc_rnd_beta,
                vc_rnd_lambda=args.vc_rnd_lambda,
                vc_rnd_mode=vc_rnd_mode,
                vc_rnd_min_scale=args.vc_rnd_min_scale,
            )
        else:
            rews_int_final = rews_int_norm
            vc_scale = np.ones_like(rews_int_norm)
            vc_delta_v = np.zeros_like(rews_int_norm)
            if cfg["intrinsic_type"] != "rnd":
                vc_running_value_mean = 0.0

        # ── GAE ──────────────────────────────────────────
        adv_int, ret_int = compute_gae(
            rews_int_final, buf_vint, last_vint,
            buf_dones, args.gamma, args.lam, use_dones=True
        )
        adv_ext, ret_ext = compute_gae(
            buf_rews_ext, buf_vext, last_vext,
            buf_dones, args.gamma_ext, args.lam, use_dones=True
        )
        advantages = cfg["int_coeff"] * adv_int + cfg["ext_coeff"] * adv_ext
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        # ── Normalised obs for RND predictor (flat, no LSTM needed) ──
        flat_obs = buf_obs.reshape(-1, *obs_shape)
        flat_obs_f = flat_obs.astype(np.float64)
        flat_obs_norm = (
            (flat_obs_f - obs_rms.mean) / (np.sqrt(obs_rms.var) + eps)
        ).astype(np.float32)

        # ── PPO + intrinsic on-policy update (sequence-based for LSTM) ──
        envs_per_mb = max(1, nenvs // args.nminibatches)
        env_inds = np.arange(nenvs)
        pg_losses, vf_losses, rnd_losses, entropies = [], [], [], []

        t_obs = torch.as_tensor(buf_obs, device=device, dtype=torch.uint8)
        t_acs = torch.as_tensor(buf_acs, device=device, dtype=torch.long)
        t_logps = torch.as_tensor(buf_logps, device=device)
        t_adv = torch.as_tensor(advantages, device=device)
        t_ret_int = torch.as_tensor(ret_int, device=device)
        t_ret_ext = torch.as_tensor(ret_ext, device=device)
        t_masks = torch.as_tensor(buf_masks, device=device)
        t_obs_norm = torch.as_tensor(
            flat_obs_norm.reshape(nenvs, nsteps, *obs_shape), device=device
        )

        for _ in range(args.nepochs):
            np.random.shuffle(env_inds)
            for start in range(0, nenvs, envs_per_mb):
                mb_env = env_inds[start:start + envs_per_mb]

                mb_obs = t_obs[mb_env]
                mb_acs = t_acs[mb_env]
                mb_logps = t_logps[mb_env]
                mb_adv = t_adv[mb_env]
                mb_ret_int = t_ret_int[mb_env]
                mb_ret_ext = t_ret_ext[mb_env]
                mb_masks = t_masks[mb_env]
                mb_obs_n = t_obs_norm[mb_env]

                mb_hx = (
                    init_lstm_hx[0][:, mb_env, :].detach(),
                    init_lstm_hx[1][:, mb_env, :].detach(),
                )

                dist, vpi, vpe, _ = ac(mb_obs, mb_hx, masks=mb_masks)

                new_logp = dist.log_prob(mb_acs.reshape(-1))
                ent = dist.entropy().mean()
                old_logp = mb_logps.reshape(-1)
                ratio = torch.exp(new_logp - old_logp)
                flat_adv = mb_adv.reshape(-1)

                pg_loss = torch.maximum(
                    -flat_adv * ratio,
                    -flat_adv * ratio.clamp(1 - args.cliprange, 1 + args.cliprange)
                ).mean()
                vf_loss = (
                    0.5 * ((vpi - mb_ret_int) ** 2).mean()
                    + 0.5 * ((vpe - mb_ret_ext) ** 2).mean()
                )

                rnd_loss = rnd.predictor_loss(
                    mb_obs_n.reshape(-1, *obs_shape)
                ) if rnd is not None else torch.tensor(0.0, device=device)

                loss = pg_loss - args.ent_coef * ent + vf_loss + rnd_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                rnd_losses.append(rnd_loss.item() if rnd is not None else 0.0)
                entropies.append(ent.item())

        # ── Replay buffer predictor update ───────────────
        replay_rnd_loss = 0.0
        if replay_buf is not None and rnd is not None and replay_buf.n_transitions > 0:
            n_replay = args.replay_buffer_replay_ratio * nenvs * nsteps
            replay_obs_np = replay_buf.sample(n_replay)
            replay_obs_f = replay_obs_np.astype(np.float64)
            replay_obs_norm = (
                (replay_obs_f - obs_rms.mean) / (np.sqrt(obs_rms.var) + eps)
            ).astype(np.float32)
            replay_obs_t = torch.as_tensor(replay_obs_norm, device=device)
            replay_loss = rnd.predictor_loss(replay_obs_t)
            pred_optimizer.zero_grad()
            replay_loss.backward()
            nn.utils.clip_grad_norm_(list(rnd.predictor.parameters()), args.max_grad_norm)
            pred_optimizer.step()
            replay_rnd_loss = replay_loss.item()

        # ── Evaluation metrics ───────────────────────────
        if first_reward_step == -1 and len(ep_rew_buf) > 0 and max(ep_rew_buf) > 0:
            first_reward_step = total_steps

        if cfg["intrinsic_type"] != "none":
            int_flat = buf_rews_int.ravel()
            vext_flat = buf_vext.ravel()
            corr_int_v = (
                float(np.corrcoef(int_flat, vext_flat)[0, 1])
                if int_flat.std() > 1e-8 and vext_flat.std() > 1e-8
                else 0.0
            )
        else:
            corr_int_v = 0.0

        def ev(pred, actual):
            var_y = np.var(actual)
            return 1 - np.var(actual - pred) / (var_y + eps) if var_y > eps else 0.0

        ev_int = ev(buf_vint.ravel(), ret_int.ravel())
        ev_ext = ev(buf_vext.ravel(), ret_ext.ravel())

        update_count += 1
        tps = (nenvs * nsteps * update_count) / max(time.time() - t_start, 1e-6)

        # ── Write CSV row ────────────────────────────────
        with open(log_path, "a") as f:
            f.write(
                f"{total_steps},"
                f"{np.mean(ep_rew_buf) if ep_rew_buf else 0.0:.4f},"
                f"{np.mean(ep_len_buf) if ep_len_buf else 0.0:.1f},"
                f"{buf_rews_int.mean():.6f},"
                f"{rews_int_norm.mean():.6f},"
                f"{rews_int_final.mean():.6f},"
                f"{np.mean(vc_scale):.4f},"
                f"{np.mean(vc_delta_v):.4f},"
                f"{vc_running_value_mean:.4f},"
                f"{replay_buf.n_transitions if replay_buf else 0},"
                f"{first_reward_step},"
                f"{corr_int_v:.4f},"
                f"{np.mean(pg_losses):.6f},"
                f"{np.mean(vf_losses):.6f},"
                f"{np.mean(rnd_losses):.6f},"
                f"{replay_rnd_loss:.6f},"
                f"{np.mean(entropies):.6f},"
                f"{ev_int:.4f},{ev_ext:.4f},{tps:.1f}\n"
            )

        if update_count % args.log_interval == 0:
            print(
                f"[{condition}][{total_steps:>10d}] "
                f"ep_rew={np.mean(ep_rew_buf) if ep_rew_buf else 0.0:.3f}  "
                f"int_final={rews_int_final.mean():.4f}  "
                f"vc_scale={np.mean(vc_scale):.3f}  "
                f"corr={corr_int_v:.3f}  "
                f"ev_ext={ev_ext:.3f}  "
                f"tps={tps:.0f}"
            )

    venv.close()
    print(f"[{condition}] Done. Logs: {log_path}")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="PPO + intrinsic-reward baselines on MiniGrid (PyTorch)")

    parser.add_argument("--env",           type=str, default="MiniGrid-FourRooms-v0")
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--num_env",       type=int, default=8)
    parser.add_argument("--num_timesteps", type=int, default=int(2e6))
    parser.add_argument("--condition",     type=str, default="ppo_vcrnd",
                        choices=list(CONDITION_OVERRIDES.keys()) + ["all"])
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--obs_size",      type=int, default=56)

    # PPO
    parser.add_argument("--nsteps",        type=int,   default=128)
    parser.add_argument("--nepochs",       type=int,   default=4)
    parser.add_argument("--nminibatches",  type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--cliprange",     type=float, default=0.1)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--gamma_ext",     type=float, default=0.99)
    parser.add_argument("--lam",           type=float, default=0.95)
    parser.add_argument("--ent_coef",      type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--int_coeff",     type=float, default=0.05)
    parser.add_argument("--ext_coeff",     type=float, default=1.0)
    parser.add_argument("--policy",        type=str,   default="impala_lstm", choices=["impala_lstm"])

    # Network
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--rnd_out_dim", type=int, default=512)

    # VCSE
    parser.add_argument("--vcse_k",               type=int,   default=10)
    parser.add_argument("--vcse_buffer_capacity", type=int,   default=50000)
    parser.add_argument("--vcse_value_scale",     type=float, default=0.1)
    parser.add_argument("--vcse_eps",             type=float, default=1e-6)
    parser.add_argument("--vcse_knn_algorithm",   type=str,   default="auto",
                        choices=["auto", "ball_tree", "kd_tree", "brute"])

    # VC-RND
    parser.add_argument("--vc_rnd",           type=int,   default=1)
    parser.add_argument("--vc_rnd_mode",      type=str,   default="inverse",
                        choices=["inverse", "sigmoid"])
    parser.add_argument("--vc_rnd_lambda",    type=float, default=1.0)
    parser.add_argument("--vc_rnd_beta",      type=float, default=0.01)
    parser.add_argument("--vc_rnd_min_scale", type=float, default=0.0)

    # Replay buffer
    parser.add_argument("--replay_buffer_capacity",     type=int, default=100)
    parser.add_argument("--replay_buffer_replay_ratio", type=int, default=4)

    # Logging
    parser.add_argument("--log_dir",      type=str, default="logs/minigrid_vcrnd")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--tag",          type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    conditions_to_run = (
        list(CONDITION_OVERRIDES.keys()) if args.condition == "all"
        else [args.condition]
    )

    for condition in conditions_to_run:
        tag     = f"{args.tag}_{condition}" if args.tag else condition
        log_dir = os.path.join(args.log_dir, tag)
        train_condition(args, condition, CONDITION_OVERRIDES[condition], log_dir)
