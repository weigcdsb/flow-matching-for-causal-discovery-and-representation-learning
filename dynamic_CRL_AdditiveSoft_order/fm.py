from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
import torch

from models import RecordingContext, ResidualDecoder, TemporalCoupledFM
from particles import GraphSpace, coefficient_dimension, features_for_graph_np, _stabilize_mechanism
from sim import SimulationData
from utils import is_dag, pad_y_list, topo_order


@dataclass
class FMMetadata:
    e_lag: int
    e_inst: int
    coefficient_dim: int
    h_child_dim: int
    h_dim: int
    h_scale_vector: np.ndarray

    @classmethod
    def build(cls, cfg: Any, graph_space: GraphSpace) -> "FMMetadata":
        coefficient_dim = coefficient_dimension(cfg)
        h_child_dim = 2 * coefficient_dim + 2
        child_scale = np.r_[
            np.full(coefficient_dim, cfg.H_COEF_SCALE),
            np.full(coefficient_dim, cfg.H_COEF_SCALE),
            cfg.H_LOGVAR_SCALE,
            cfg.H_LOGVAR_SCALE,
        ].astype(np.float32)
        return cls(
            e_lag=len(graph_space.lag_edge_list),
            e_inst=len(graph_space.inst_pair_list),
            coefficient_dim=coefficient_dim,
            h_child_dim=h_child_dim,
            h_dim=cfg.A * h_child_dim,
            h_scale_vector=np.tile(child_scale, cfg.A).astype(np.float32),
        )


def graph_to_lag_labels(
    graph: np.ndarray,
    graph_space: GraphSpace,
) -> np.ndarray:
    return np.array(
        [
            int(graph[parent, child] > 0.5)
            for parent, child in graph_space.lag_edge_list
        ],
        dtype=np.int64,
    )


def lag_labels_to_graph(
    labels: np.ndarray,
    graph_space: GraphSpace,
    cfg: Any,
) -> np.ndarray:
    graph = np.zeros((cfg.A, cfg.A), np.float32)
    for state, (parent, child) in zip(
        labels, graph_space.lag_edge_list
    ):
        graph[parent, child] = float(int(state) == 1)
    return graph


def graph_to_inst_labels(
    graph: np.ndarray,
    graph_space: GraphSpace,
) -> np.ndarray:
    labels = []
    for left, right in graph_space.inst_pair_list:
        labels.append(
            1
            if graph[left, right] > 0.5
            else (2 if graph[right, left] > 0.5 else 0)
        )
    return np.asarray(labels, dtype=np.int64)


def inst_labels_to_graph(
    labels: np.ndarray,
    graph_space: GraphSpace,
    cfg: Any,
) -> np.ndarray:
    graph = np.zeros((cfg.A, cfg.A), np.float32)
    for state, (left, right) in zip(
        labels, graph_space.inst_pair_list
    ):
        state = int(state)
        if state == 1:
            graph[left, right] = 1.0
        elif state == 2:
            graph[right, left] = 1.0
    return graph


def pack_H_raw(mechanisms: list[dict]) -> np.ndarray:
    output: list[float] = []
    for mechanism in mechanisms:
        log_obs_var = np.log(mechanism["obs_var"])
        output.extend(mechanism["alpha0"].tolist())
        output.extend(mechanism["delta_alpha"].tolist())
        output.extend(
            [
                log_obs_var,
                np.log(mechanism["int_var"]) - log_obs_var,
            ]
        )
    return np.asarray(output, np.float32)


def pack_H_scaled(
    mechanisms: list[dict],
    metadata: FMMetadata,
) -> np.ndarray:
    return pack_H_raw(mechanisms) / metadata.h_scale_vector


def unpack_H_scaled(
    scaled: np.ndarray,
    cfg: Any,
    metadata: FMMetadata,
) -> list[dict]:
    raw = np.asarray(scaled) * metadata.h_scale_vector
    output: list[dict] = []
    position = 0
    for _ in range(cfg.A):
        alpha0 = raw[
            position : position + metadata.coefficient_dim
        ].copy()
        position += metadata.coefficient_dim
        delta_alpha = raw[
            position : position + metadata.coefficient_dim
        ].copy()
        position += metadata.coefficient_dim
        log_obs_var = float(raw[position])
        delta_log_var = float(raw[position + 1])
        position += 2
        alpha1 = alpha0 + delta_alpha
        output.append(
            _stabilize_mechanism(
                alpha0,
                alpha1,
                float(np.exp(log_obs_var)),
                float(np.exp(log_obs_var + delta_log_var)),
                cfg,
            )
        )
    return output


def H_target_mask(
    lag_graph: np.ndarray,
    inst_graph: np.ndarray,
    cfg: Any,
    metadata: FMMetadata,
) -> np.ndarray:
    child_masks = []
    for child in range(cfg.A):
        coefficient_mask = np.r_[
            np.ones(2, np.float32),
            lag_graph[:, child].astype(np.float32),
            inst_graph[:, child].astype(np.float32),
        ]
        child_masks.append(
            np.r_[coefficient_mask, coefficient_mask, 1.0, 1.0]
        )
    output = np.concatenate(child_masks).astype(np.float32)
    if output.shape != (metadata.h_dim,):
        raise ValueError("Unexpected H mask dimension")
    return output

def _single_step_features(
    previous: np.ndarray,
    current: np.ndarray,
    lag_graph: np.ndarray,
    inst_graph: np.ndarray,
    child: int,
    cfg: Any,
) -> np.ndarray:
    """Fixed-size feature construction for one FM simulation step."""
    if previous.shape != (cfg.A,) or current.shape != (cfg.A,):
        raise ValueError("Unexpected latent state shape during FM simulation")
    if lag_graph.shape != (cfg.A, cfg.A) or inst_graph.shape != (cfg.A, cfg.A):
        raise ValueError("Unexpected graph shape during FM simulation")

    features = np.empty(2 + 2 * cfg.A, dtype=np.float64)
    features[0] = 1.0
    features[1] = float(previous[child])
    for parent in range(cfg.A):
        features[2 + parent] = (
            math.tanh(float(previous[parent])) * float(lag_graph[parent, child])
        )
        features[2 + cfg.A + parent] = (
            math.tanh(float(current[parent])) * float(inst_graph[parent, child])
        )
    return features


def simulate_latent_from_component(
    state: dict[str, np.ndarray],
    pair_index: int,
    mechanisms: list[dict],
    intervention: np.ndarray,
    length: int,
    rng: np.random.Generator,
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
) -> np.ndarray:
    lag_graph, inst_graph = graph_space.graph_pairs[int(pair_index)]
    order = topo_order(inst_graph)
    latent = np.zeros((length, cfg.A), np.float32)
    pool = state["Z"][data.train_idx, 0]
    latent[0] = pool[rng.integers(0, len(pool))]
    for time_index in range(1, length):
        previous = latent[time_index - 1]
        current = np.zeros(cfg.A, np.float32)
        for child in order:
            mechanism = mechanisms[child]
            features = _single_step_features(
                previous, current, lag_graph, inst_graph, child, cfg
            )
            alpha = mechanism["alpha0"] + intervention[
                child
            ] * mechanism["delta_alpha"]
            mean = float(alpha @ features)
            variance = float(
                mechanism["int_var"]
                if intervention[child] > 0.5
                else mechanism["obs_var"]
            )
            current[child] = float(
                rng.normal(mean, math.sqrt(max(variance, 1e-8)))
            )
            if not np.isfinite(current[child]):
                raise FloatingPointError("Non-finite latent value during FM simulation")
        latent[time_index] = current
    if not np.isfinite(latent).all():
        raise FloatingPointError("Non-finite latent trajectory during FM simulation")
    return latent


def latent_to_observation(
    decoder: ResidualDecoder,
    state: dict[str, np.ndarray],
    latent: np.ndarray,
    rng: np.random.Generator,
    cfg: Any,
    data: SimulationData,
) -> np.ndarray:
    latent_raw = latent * state["zs"] + state["zm"]
    decoder.eval()
    with torch.no_grad():
        observation = decoder(
            torch.tensor(
                latent_raw, dtype=torch.float32, device=data.device
            )
        ).cpu().numpy()
    decoder.train()
    observation = observation + rng.normal(
        0,
        cfg.OBS_NOISE_SD,
        size=observation.shape,
    ).astype(np.float32)
    if not np.isfinite(observation).all():
        raise FloatingPointError("Non-finite observation during FM simulation")
    return observation.astype(np.float32)


def make_fm_simulation_batch(
    decoder: ResidualDecoder,
    state: dict[str, np.ndarray],
    pair_indices: np.ndarray,
    mechanisms: list[list[dict]],
    mixture_weights: np.ndarray,
    outer: int,
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    metadata: FMMetadata,
) -> dict:
    rng = np.random.default_rng(cfg.SEED + 5000 + outer)
    recordings: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    lag_targets: list[np.ndarray] = []
    inst_targets: list[np.ndarray] = []
    h_targets: list[np.ndarray] = []
    h_masks: list[np.ndarray] = []
    weights: list[float] = []
    intervention_conditions = np.unique(
        data.I_np[data.train_idx], axis=0
    ).astype(np.float32)
    for component in range(len(pair_indices)):
        lag_graph, inst_graph = graph_space.graph_pairs[
            int(pair_indices[component])
        ]
        for intervention in intervention_conditions:
            for _ in range(
                cfg.FM_SIM_REPS_PER_COMPONENT_INTERVENTION
            ):
                length = cfg.FM_SIM_LENGTHS[
                    rng.integers(0, len(cfg.FM_SIM_LENGTHS))
                ]
                latent = simulate_latent_from_component(
                    state,
                    pair_indices[component],
                    mechanisms[component],
                    intervention,
                    length,
                    rng,
                    cfg,
                    data,
                    graph_space,
                )
                observation = latent_to_observation(
                    decoder, state, latent, rng, cfg, data
                )
                recordings.append(observation)
                interventions.append(intervention.copy())
                lag_targets.append(
                    graph_to_lag_labels(lag_graph, graph_space)
                )
                inst_targets.append(
                    graph_to_inst_labels(inst_graph, graph_space)
                )
                h_targets.append(
                    pack_H_scaled(mechanisms[component], metadata)
                )
                h_masks.append(
                    H_target_mask(
                        lag_graph, inst_graph, cfg, metadata
                    )
                )
                weights.append(float(mixture_weights[component]))
    y_padded, valid_mask = pad_y_list(recordings, cfg.P, data.device)
    return {
        "ypad": y_padded,
        "valid_mask": valid_mask,
        "I": torch.tensor(
            np.stack(interventions),
            dtype=torch.float32,
            device=data.device,
        ),
        "g_lag": torch.tensor(
            np.stack(lag_targets), dtype=torch.long, device=data.device
        ),
        "g_inst": torch.tensor(
            np.stack(inst_targets), dtype=torch.long, device=data.device
        ),
        "h": torch.tensor(
            np.stack(h_targets),
            dtype=torch.float32,
            device=data.device,
        ),
        "h_mask": torch.tensor(
            np.stack(h_masks),
            dtype=torch.float32,
            device=data.device,
        ),
        "weight": torch.tensor(
            np.asarray(weights),
            dtype=torch.float32,
            device=data.device,
        ),
        "lengths": np.asarray(
            [len(recording) for recording in recordings]
        ),
    }


def fm_update_from_batch(
    context_net: RecordingContext,
    fm: TemporalCoupledFM,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    cfg: Any,
    metadata: FMMetadata,
    device: torch.device,
) -> np.ndarray:
    total_batch = batch["ypad"].shape[0]
    logs = []
    for _ in range(cfg.FM_UPDATES_PER_OUTER):
        batch_size = min(cfg.FM_BATCH, total_batch)
        indices = torch.randint(
            0, total_batch, (batch_size,), device=device
        )
        y_padded = batch["ypad"][indices]
        valid_mask = batch["valid_mask"][indices]
        intervention = batch["I"][indices]
        target_lag = batch["g_lag"][indices]
        target_inst = batch["g_inst"][indices]
        target_h = batch["h"][indices]
        h_mask = batch["h_mask"][indices]
        weight = batch["weight"][indices]
        weight = weight / (weight.mean() + 1e-8)
        time = cfg.FM_GRAPH_EPS + (
            1 - 2 * cfg.FM_GRAPH_EPS
        ) * torch.rand(batch_size, 1, device=device)
        keep_lag = (
            torch.rand(batch_size, metadata.e_lag, device=device) < time
        )
        keep_inst = (
            torch.rand(batch_size, metadata.e_inst, device=device) < time
        )
        lag_state = torch.where(
            keep_lag, target_lag, torch.zeros_like(target_lag)
        )
        inst_state = torch.where(
            keep_inst, target_inst, torch.zeros_like(target_inst)
        )
        source_h = torch.randn_like(target_h)
        h_state = (1 - time) * source_h + time * target_h
        h_velocity_target = target_h - source_h

        with torch.no_grad():
            context = context_net(y_padded, valid_mask, intervention)
        lag_logits, inst_logits, h_velocity = fm(
            context, lag_state, inst_state, h_state, time
        )
        lag_loss_edge = torch.nn.functional.cross_entropy(
            lag_logits.reshape(-1, 2),
            target_lag.reshape(-1),
            reduction="none",
        ).reshape(batch_size, metadata.e_lag)
        inst_loss_edge = torch.nn.functional.cross_entropy(
            inst_logits.reshape(-1, 3),
            target_inst.reshape(-1),
            reduction="none",
        ).reshape(batch_size, metadata.e_inst)
        lag_loss_per = lag_loss_edge.mean(1)
        inst_loss_per = inst_loss_edge.mean(1)
        h_loss_per = (
            ((h_velocity - h_velocity_target) ** 2) * h_mask
        ).sum(1) / h_mask.sum(1).clamp_min(1.0)
        lag_loss = torch.mean(weight * lag_loss_per)
        inst_loss = torch.mean(weight * inst_loss_per)
        h_loss = torch.mean(weight * h_loss_per)
        total_loss = (
            cfg.FM_LAMBDA_LAG * lag_loss
            + cfg.FM_LAMBDA_INST * inst_loss
            + cfg.FM_LAMBDA_H * h_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 10.0)
        optimizer.step()
        logs.append(
            [
                float(total_loss.detach()),
                float(lag_loss.detach()),
                float(inst_loss.detach()),
                float(h_loss.detach()),
            ]
        )
    return np.mean(logs, axis=0)


def prepare_context_for_test(
    context_net: RecordingContext,
    observations: np.ndarray,
    interventions: np.ndarray,
    cfg: Any,
    device: torch.device,
) -> torch.Tensor:
    recordings = [
        observations[index] for index in range(len(observations))
    ]
    y_padded, valid_mask = pad_y_list(recordings, cfg.P, device)
    intervention_tensor = torch.tensor(
        interventions, dtype=torch.float32, device=device
    )
    context_net.eval()
    with torch.no_grad():
        return context_net(y_padded, valid_mask, intervention_tensor)


def sample_classes(probabilities: torch.Tensor) -> torch.Tensor:
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])


def defog_absorbing_rates(
    current: torch.Tensor,
    clean_sample: torch.Tensor,
    time: float,
    num_classes: int,
    eta: float = 0.0,
) -> torch.Tensor:
    """DeFoG R* plus optional general detailed-balance stochasticity."""
    source = torch.zeros_like(clean_sample)
    p_time = (1.0 - time) * torch.nn.functional.one_hot(
        source, num_classes
    ).float()
    p_time = p_time + time * torch.nn.functional.one_hot(
        clean_sample, num_classes
    ).float()
    derivative = (
        torch.nn.functional.one_hot(clean_sample, num_classes).float()
        - torch.nn.functional.one_hot(source, num_classes).float()
    )
    current_index = current[..., None]
    p_current = p_time.gather(-1, current_index).squeeze(-1)
    derivative_current = derivative.gather(
        -1, current_index
    ).squeeze(-1)
    support_size = (p_time > 0).sum(-1).float()
    denominator = support_size * p_current
    rates = torch.relu(
        derivative - derivative_current[..., None]
    ) / denominator[..., None]
    if eta != 0:
        rates = rates + float(eta) * p_time
    rates = torch.nan_to_num(
        rates, nan=0.0, posinf=0.0, neginf=0.0
    )
    rates = torch.where(
        (p_current > 0)[..., None], rates, torch.zeros_like(rates)
    )
    rates = torch.where(p_time > 0, rates, torch.zeros_like(rates))
    rates.scatter_(-1, current_index, 0.0)
    return rates


def euler_class_probabilities(
    current: torch.Tensor,
    rates: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    probabilities = rates * step_size
    off_diagonal_sum = probabilities.sum(-1, keepdim=True)
    stay = (1.0 - off_diagonal_sum).clamp_min(0.0)
    probabilities.scatter_(-1, current[..., None], stay)
    return probabilities / probabilities.sum(
        -1, keepdim=True
    ).clamp_min(1e-12)


def sample_inst_dag_step(
    current: torch.Tensor,
    probabilities: torch.Tensor,
    graph_space: GraphSpace,
    cfg: Any,
) -> torch.Tensor:
    """Sequential categorical update with cycle-forming states removed."""
    updated = current.clone()
    for edge_index_tensor in torch.randperm(
        len(graph_space.inst_pair_list), device=current.device
    ):
        edge_index = int(edge_index_tensor)
        weights = probabilities[edge_index].clone()
        for state in (1, 2):
            candidate = updated.clone()
            candidate[edge_index] = state
            graph = inst_labels_to_graph(
                candidate.cpu().numpy(), graph_space, cfg
            )
            if not is_dag(graph):
                weights[state] = 0.0
        if float(weights.sum()) <= 0:
            continue
        updated[edge_index] = torch.multinomial(weights, 1)[0]
    return updated


def sample_fm_given_context(
    fm: TemporalCoupledFM,
    context: torch.Tensor,
    cfg: Any,
    metadata: FMMetadata,
    graph_space: GraphSpace,
    device: torch.device,
    n_samples: int = 1,
    eta: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = cfg.FM_ETA if eta is None else eta
    fm.eval()
    lag_samples: list[np.ndarray] = []
    inst_samples: list[np.ndarray] = []
    h_samples: list[np.ndarray] = []
    step_size = 1.0 / cfg.FM_GEN_STEPS
    with torch.no_grad():
        for _ in range(n_samples):
            lag_state = torch.zeros(
                metadata.e_lag, dtype=torch.long, device=device
            )
            inst_state = torch.zeros(
                metadata.e_inst, dtype=torch.long, device=device
            )
            h_state = torch.randn(
                metadata.h_dim, dtype=torch.float32, device=device
            )
            for step in range(cfg.FM_GEN_STEPS):
                time_value = step / cfg.FM_GEN_STEPS + 1e-6
                time = torch.tensor(
                    [[time_value]], dtype=torch.float32, device=device
                )
                lag_logits, inst_logits, h_velocity = fm(
                    context[None, :],
                    lag_state[None, :],
                    inst_state[None, :],
                    h_state[None, :],
                    time,
                )
                predicted_lag = torch.softmax(lag_logits[0], dim=-1)
                predicted_inst = torch.softmax(inst_logits[0], dim=-1)
                h_state = h_state + step_size * h_velocity[0]
                if step == cfg.FM_GEN_STEPS - 1:
                    lag_state = sample_classes(predicted_lag)
                    inst_state = sample_inst_dag_step(
                        inst_state, predicted_inst, graph_space, cfg
                    )
                    continue
                clean_lag = sample_classes(predicted_lag)
                clean_inst = sample_classes(predicted_inst)
                lag_rates = defog_absorbing_rates(
                    lag_state, clean_lag, time_value, 2, eta
                )
                inst_rates = defog_absorbing_rates(
                    inst_state, clean_inst, time_value, 3, eta
                )
                lag_probabilities = euler_class_probabilities(
                    lag_state, lag_rates, step_size
                )
                inst_probabilities = euler_class_probabilities(
                    inst_state, inst_rates, step_size
                )
                lag_state = sample_classes(lag_probabilities)
                inst_state = sample_inst_dag_step(
                    inst_state, inst_probabilities, graph_space, cfg
                )
            lag_samples.append(lag_state.cpu().numpy())
            inst_samples.append(inst_state.cpu().numpy())
            h_samples.append(h_state.cpu().numpy())
    fm.train()
    return (
        np.stack(lag_samples),
        np.stack(inst_samples),
        np.stack(h_samples),
    )


def sample_H_given_fixed_graph(
    fm: TemporalCoupledFM,
    context: torch.Tensor,
    lag_target: np.ndarray,
    inst_target: np.ndarray,
    cfg: Any,
    metadata: FMMetadata,
    graph_space: GraphSpace,
    device: torch.device,
    n_samples: int = 100,
) -> np.ndarray:
    fm.eval()
    target_lag = torch.tensor(
        graph_to_lag_labels(lag_target, graph_space),
        dtype=torch.long,
        device=device,
    )
    target_inst = torch.tensor(
        graph_to_inst_labels(inst_target, graph_space),
        dtype=torch.long,
        device=device,
    )
    samples: list[np.ndarray] = []
    step_size = 1.0 / cfg.FM_GEN_STEPS
    with torch.no_grad():
        for _ in range(n_samples):
            h_state = torch.randn(
                metadata.h_dim, dtype=torch.float32, device=device
            )
            arrival_lag = torch.rand(metadata.e_lag, device=device)
            arrival_inst = torch.rand(metadata.e_inst, device=device)
            for step in range(cfg.FM_GEN_STEPS):
                time_value = (step + 0.5) / cfg.FM_GEN_STEPS
                lag_state = torch.where(
                    arrival_lag <= time_value,
                    target_lag,
                    torch.zeros_like(target_lag),
                )
                inst_state = torch.where(
                    arrival_inst <= time_value,
                    target_inst,
                    torch.zeros_like(target_inst),
                )
                time = torch.tensor(
                    [[time_value]], dtype=torch.float32, device=device
                )
                _, _, h_velocity = fm(
                    context[None, :],
                    lag_state[None, :],
                    inst_state[None, :],
                    h_state[None, :],
                    time,
                )
                h_state = h_state + step_size * h_velocity[0]
            samples.append(h_state.cpu().numpy())
    fm.train()
    return np.stack(samples)
