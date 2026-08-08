from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from utils import adj_from_edges, topo_order


@dataclass
class TrueModel:
    lag_graphs: list[np.ndarray]
    inst_graphs: list[np.ndarray]
    mixture_weights: np.ndarray
    bias0: list[np.ndarray]
    self_coef0: list[np.ndarray]
    lag_coef0: list[np.ndarray]
    inst_coef0: list[np.ndarray]
    bias1: list[np.ndarray]
    self_coef1: list[np.ndarray]
    lag_coef1: list[np.ndarray]
    inst_coef1: list[np.ndarray]
    noise_sd0: list[np.ndarray]
    noise_sd1: list[np.ndarray]
    intervention_conditions: np.ndarray


@dataclass
class ObservationModel:
    linear: np.ndarray
    nonlinear_in: np.ndarray
    nonlinear_out: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class SimulationData:
    true_model: TrueModel
    observation_model: ObservationModel
    Z_true_np: np.ndarray
    Y_np: np.ndarray
    I_np: np.ndarray
    regime_labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    Y: torch.Tensor
    I_torch: torch.Tensor
    Z_test_true_np: np.ndarray
    Y_test_np: np.ndarray
    I_test_np: np.ndarray
    test_regime_labels: np.ndarray
    device: torch.device


def build_true_model(cfg: Any) -> TrueModel:
    lag_graphs = [
        adj_from_edges([(0, 1), (0, 2), (1, 2)], cfg.A),
        adj_from_edges([(2, 1), (2, 0), (1, 0)], cfg.A),
    ]
    inst_graphs = [
        adj_from_edges([(0, 1), (1, 2)], cfg.A),
        adj_from_edges([(2, 1), (1, 0)], cfg.A),
    ]

    bias0 = [
        np.array([0.45, -0.40, 0.30], np.float32),
        np.array([-0.45, 0.40, -0.30], np.float32),
    ]
    self_coef0 = [
        np.array([0.55, 0.45, 0.50], np.float32),
        np.array([0.48, 0.52, 0.42], np.float32),
    ]
    lag_coef0 = [np.zeros((cfg.A, cfg.A), np.float32) for _ in range(cfg.K_DATA)]
    inst_coef0 = [np.zeros((cfg.A, cfg.A), np.float32) for _ in range(cfg.K_DATA)]
    lag_coef0[0][0, 1] = 0.80
    lag_coef0[0][0, 2] = -0.60
    lag_coef0[0][1, 2] = 0.70
    inst_coef0[0][0, 1] = 0.65
    inst_coef0[0][1, 2] = -0.55
    lag_coef0[1][2, 1] = -0.75
    lag_coef0[1][2, 0] = 0.65
    lag_coef0[1][1, 0] = -0.70
    inst_coef0[1][2, 1] = 0.60
    inst_coef0[1][1, 0] = 0.55

    # Soft intervention: preserve the same parents but change their coefficients.
    bias_shift = [
        np.array([-0.75, 0.80, -0.65], np.float32),
        np.array([0.75, -0.80, 0.65], np.float32),
    ]
    bias1 = [bias0[k] + bias_shift[k] for k in range(cfg.K_DATA)]
    self_coef1 = [0.65 * self_coef0[k] for k in range(cfg.K_DATA)]
    lag_coef1 = [0.65 * lag_coef0[k] for k in range(cfg.K_DATA)]
    inst_coef1 = [0.65 * inst_coef0[k] for k in range(cfg.K_DATA)]

    noise_sd0 = [
        np.array([0.20, 0.20, 0.20], np.float32),
        np.array([0.22, 0.19, 0.21], np.float32),
    ]
    noise_multiplier = np.array([1.25, 0.80, 1.15], np.float32)[: cfg.A]
    noise_sd1 = [noise_sd0[k] * noise_multiplier for k in range(cfg.K_DATA)]

    return TrueModel(
        lag_graphs=lag_graphs,
        inst_graphs=inst_graphs,
        mixture_weights=np.array([0.55, 0.45], dtype=np.float64),
        bias0=bias0,
        self_coef0=self_coef0,
        lag_coef0=lag_coef0,
        inst_coef0=inst_coef0,
        bias1=bias1,
        self_coef1=self_coef1,
        lag_coef1=lag_coef1,
        inst_coef1=inst_coef1,
        noise_sd0=noise_sd0,
        noise_sd1=noise_sd1,
        intervention_conditions=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        )[:, : cfg.A],
    )


def structural_mean(
    previous: np.ndarray,
    current: np.ndarray,
    child: int,
    lag_graph: np.ndarray,
    inst_graph: np.ndarray,
    bias: np.ndarray,
    self_coef: np.ndarray,
    lag_coef: np.ndarray,
    inst_coef: np.ndarray,
) -> float:
    mean = bias[child] + self_coef[child] * previous[child]
    mean += np.sum(
        lag_graph[:, child]
        * lag_coef[:, child]
        * np.tanh(previous)
    )
    mean += np.sum(
        inst_graph[:, child]
        * inst_coef[:, child]
        * np.tanh(current)
    )
    return float(mean)


def simulate_latents(
    n: int,
    length: int,
    rng: np.random.Generator,
    cfg: Any,
    true_model: TrueModel,
    mixture_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = true_model.mixture_weights if mixture_weights is None else mixture_weights
    labels = rng.choice(cfg.K_DATA, size=n, p=weights)
    intervention_index = rng.integers(0, len(true_model.intervention_conditions), size=n)
    interventions = true_model.intervention_conditions[intervention_index].astype(np.float32)
    latents = np.zeros((n, length, cfg.A), np.float32)
    latents[:, 0] = rng.normal(size=(n, cfg.A)).astype(np.float32)

    for trajectory in range(n):
        regime = int(labels[trajectory])
        lag_graph = true_model.lag_graphs[regime]
        inst_graph = true_model.inst_graphs[regime]
        order = topo_order(inst_graph)
        for time_index in range(1, length):
            previous = latents[trajectory, time_index - 1]
            current = np.zeros(cfg.A, np.float32)
            for child in order:
                intervened = interventions[trajectory, child] > 0.5
                if intervened:
                    mean = structural_mean(
                        previous,
                        current,
                        child,
                        lag_graph,
                        inst_graph,
                        true_model.bias1[regime],
                        true_model.self_coef1[regime],
                        true_model.lag_coef1[regime],
                        true_model.inst_coef1[regime],
                    )
                    sd = true_model.noise_sd1[regime][child]
                else:
                    mean = structural_mean(
                        previous,
                        current,
                        child,
                        lag_graph,
                        inst_graph,
                        true_model.bias0[regime],
                        true_model.self_coef0[regime],
                        true_model.lag_coef0[regime],
                        true_model.inst_coef0[regime],
                    )
                    sd = true_model.noise_sd0[regime][child]
                current[child] = rng.normal(mean, sd)
            latents[trajectory, time_index] = current
    return latents, interventions, labels


def build_observation_model(
    cfg: Any,
    latents: np.ndarray,
) -> tuple[ObservationModel, np.ndarray]:
    rng = np.random.default_rng(cfg.SEED + 200)
    linear = rng.normal(size=(cfg.A, cfg.P)).astype(np.float32)
    left, _, right_t = np.linalg.svd(linear, full_matrices=False)
    linear = (2.0 * (left @ right_t)).astype(np.float32)
    nonlinear_in = rng.normal(0, 0.8, size=(cfg.A, 24)).astype(np.float32)
    nonlinear_out = rng.normal(
        0, 1 / np.sqrt(24), size=(24, cfg.P)
    ).astype(np.float32)
    raw = (
        latents @ linear
        + cfg.OBS_NONLINEAR_STRENGTH
        * (np.tanh(latents @ nonlinear_in) @ nonlinear_out)
        + rng.normal(
            0,
            cfg.OBS_NOISE_SD,
            size=(cfg.N, cfg.TRAIN_L, cfg.P),
        ).astype(np.float32)
    )
    mean = raw.reshape(-1, cfg.P).mean(0, keepdims=True)
    std = raw.reshape(-1, cfg.P).std(0, keepdims=True) + 1e-6
    standardized = ((raw - mean) / std).astype(np.float32)
    return ObservationModel(
        linear, nonlinear_in, nonlinear_out, mean, std
    ), standardized


def observe_latents(
    latents: np.ndarray,
    rng: np.random.Generator,
    cfg: Any,
    observation_model: ObservationModel,
) -> np.ndarray:
    raw = (
        latents @ observation_model.linear
        + cfg.OBS_NONLINEAR_STRENGTH
        * (
            np.tanh(latents @ observation_model.nonlinear_in)
            @ observation_model.nonlinear_out
        )
        + rng.normal(
            0,
            cfg.OBS_NOISE_SD,
            size=(*latents.shape[:-1], cfg.P),
        ).astype(np.float32)
    )
    return (
        (raw - observation_model.mean) / observation_model.std
    ).astype(np.float32)


def simulate_experiment_data(
    cfg: Any,
    device: torch.device,
) -> SimulationData:
    true_model = build_true_model(cfg)

    rng_train = np.random.default_rng(cfg.SEED)
    Z_true_np, I_np, regime_labels = simulate_latents(
        cfg.N, cfg.TRAIN_L, rng_train, cfg, true_model
    )
    observation_model, Y_np = build_observation_model(cfg, Z_true_np)
    rng_split = np.random.default_rng(cfg.SEED + 11)
    indices = np.arange(cfg.N)
    rng_split.shuffle(indices)
    n_val = int(round(cfg.VAL_FRAC * cfg.N))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    rng_test = np.random.default_rng(cfg.SEED + 555)
    Z_test_true_np, I_test_np, test_regime_labels = simulate_latents(
        cfg.N_TEST, cfg.TEST_L, rng_test, cfg, true_model
    )
    Y_test_np = observe_latents(
        Z_test_true_np, rng_test, cfg, observation_model
    )
    return SimulationData(
        true_model=true_model,
        observation_model=observation_model,
        Z_true_np=Z_true_np,
        Y_np=Y_np,
        I_np=I_np,
        regime_labels=regime_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        Y=torch.tensor(Y_np, dtype=torch.float32, device=device),
        I_torch=torch.tensor(I_np, dtype=torch.float32, device=device),
        Z_test_true_np=Z_test_true_np,
        Y_test_np=Y_test_np,
        I_test_np=I_test_np,
        test_regime_labels=test_regime_labels,
        device=device,
    )
