import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from models import CausalEncoder, ResidualDecoder
from sim import SimulationData
from utils import is_dag


@dataclass
class GraphSpace:
    lag_edge_list: list[tuple[int, int]]
    inst_pair_list: list[tuple[int, int]]
    lag_graphs: list[np.ndarray]
    inst_graphs: list[np.ndarray]
    graph_pairs: list[tuple[np.ndarray, np.ndarray]]
    pair_lookup: dict[tuple[bytes, bytes], int]

    @classmethod
    def build(cls, cfg: Any) -> "GraphSpace":
        lag_edges = [
            (parent, child)
            for parent in range(cfg.A)
            for child in range(cfg.A)
            if parent != child
        ]
        inst_pairs = [
            (left, right)
            for left in range(cfg.A)
            for right in range(left + 1, cfg.A)
        ]

        lag_graphs: list[np.ndarray] = []
        for bits in itertools.product([0, 1], repeat=len(lag_edges)):
            graph = np.zeros((cfg.A, cfg.A), np.float32)
            for bit, edge in zip(bits, lag_edges):
                graph[edge] = bit
            lag_graphs.append(graph)

        inst_graphs: list[np.ndarray] = []
        for states in itertools.product([0, 1, 2], repeat=len(inst_pairs)):
            graph = np.zeros((cfg.A, cfg.A), np.float32)
            for state, (left, right) in zip(states, inst_pairs):
                if state == 1:
                    graph[left, right] = 1.0
                elif state == 2:
                    graph[right, left] = 1.0
            if is_dag(graph):
                inst_graphs.append(graph)

        graph_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        pair_lookup: dict[tuple[bytes, bytes], int] = {}
        for lag_graph in lag_graphs:
            for inst_graph in inst_graphs:
                index = len(graph_pairs)
                graph_pairs.append((lag_graph.copy(), inst_graph.copy()))
                pair_lookup[
                    (
                        lag_graph.astype(np.int8).tobytes(),
                        inst_graph.astype(np.int8).tobytes(),
                    )
                ] = index

        return cls(
            lag_edge_list=lag_edges,
            inst_pair_list=inst_pairs,
            lag_graphs=lag_graphs,
            inst_graphs=inst_graphs,
            graph_pairs=graph_pairs,
            pair_lookup=pair_lookup,
        )

    def parent_tuple(self, graph: np.ndarray, child: int) -> tuple[int, ...]:
        return tuple(np.where(graph[:, child] > 0.5)[0].tolist())

    def pair_index(self, lag_graph: np.ndarray, inst_graph: np.ndarray) -> int:
        return self.pair_lookup[
            (
                lag_graph.astype(np.int8).tobytes(),
                inst_graph.astype(np.int8).tobytes(),
            )
        ]

    def true_pair_indices(self, data: SimulationData) -> list[int]:
        return [
            self.pair_index(
                data.true_model.lag_graphs[regime],
                data.true_model.inst_graphs[regime],
            )
            for regime in range(len(data.true_model.lag_graphs))
        ]

    def propose_local_pair(
        self,
        pair_index: int,
        rng: np.random.Generator,
    ) -> int:
        """Symmetric local proposal; invalid instantaneous moves are self-loops."""
        lag_graph, inst_graph = self.graph_pairs[int(pair_index)]

        if rng.random() < 0.5:
            parent, child = self.lag_edge_list[
                rng.integers(len(self.lag_edge_list))
            ]
            proposed_lag = lag_graph.copy()
            proposed_lag[parent, child] = 1.0 - proposed_lag[parent, child]
            return self.pair_index(proposed_lag, inst_graph)

        left, right = self.inst_pair_list[
            rng.integers(len(self.inst_pair_list))
        ]
        current = (
            1
            if inst_graph[left, right] > 0.5
            else (2 if inst_graph[right, left] > 0.5 else 0)
        )
        choices = [state for state in (0, 1, 2) if state != current]
        proposed_state = choices[rng.integers(2)]

        proposed_inst = inst_graph.copy()
        proposed_inst[left, right] = 0.0
        proposed_inst[right, left] = 0.0
        if proposed_state == 1:
            proposed_inst[left, right] = 1.0
        elif proposed_state == 2:
            proposed_inst[right, left] = 1.0

        if not is_dag(proposed_inst):
            return int(pair_index)
        return self.pair_index(lag_graph, proposed_inst)

    def local_mh_update(
        self,
        pair_index: int,
        scores: np.ndarray,
        rng: np.random.Generator,
        cfg: Any,
    ) -> tuple[int, int]:
        current = int(pair_index)
        accepted = 0
        temperature = max(float(cfg.TAU_GRAPH), 1e-12)
        for _ in range(cfg.GRAPH_MOVES_PER_OUTER):
            proposal = self.propose_local_pair(current, rng)
            if proposal == current:
                continue
            log_alpha = (scores[proposal] - scores[current]) / temperature
            if np.log(rng.random()) < min(0.0, log_alpha):
                current = proposal
                accepted += 1
        return current, accepted


def one_shot_intervention_warmstart(
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=cfg.D, random_state=cfg.SEED).fit(
        data.Y_np[data.train_idx].reshape(-1, cfg.P)
    )
    base_weight = pca.components_.copy()
    base_bias = -pca.mean_ @ base_weight.T
    latent = data.Y_np.reshape(-1, cfg.P) @ base_weight.T + base_bias
    latent = latent.reshape(cfg.N, cfg.TRAIN_L, cfg.D)

    train_flat = latent[data.train_idx].reshape(-1, cfg.D)
    mean = train_flat.mean(0)
    covariance = np.cov(train_flat - mean, rowvar=False) + 1e-6 * np.eye(cfg.D)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened = (latent - mean) @ whitening

    rows = []
    for target in range(cfg.A):
        active = data.train_idx[data.I_np[data.train_idx, target] > 0.5]
        current = whitened[active, 1:].reshape(-1, cfg.D)
        previous = whitened[active, :-1].reshape(-1, cfg.D)
        current -= current.mean(0)
        previous -= previous.mean(0)
        cross_covariance = current.T @ previous / max(len(current) - 1, 1)
        left, _, _ = np.linalg.svd(cross_covariance, full_matrices=True)
        rows.append(left[:, -1])
    chart = np.stack(rows)
    chart /= np.linalg.norm(chart, axis=1, keepdims=True) + 1e-12

    transform = whitening @ chart.T
    weight = (base_weight.T @ transform).T
    bias = (base_bias - mean) @ transform
    return weight, bias


def pca_warmstart(
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=cfg.D, random_state=cfg.SEED).fit(
        data.Y_np[data.train_idx].reshape(-1, cfg.P)
    )
    return pca.components_.copy(), -pca.mean_ @ pca.components_.T


def init_representation_models(
    cfg: Any,
    data: SimulationData,
) -> tuple[
    CausalEncoder,
    ResidualDecoder,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.MultiStepLR,
]:
    torch.manual_seed(cfg.SEED)
    if cfg.USE_INTERVENTION_WARMSTART:
        weight, bias = one_shot_intervention_warmstart(cfg, data)
    else:
        weight, bias = pca_warmstart(cfg, data)

    encoder = CausalEncoder(cfg, weight, bias).to(data.device)
    decoder = ResidualDecoder(cfg, encoder, data).to(data.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.LR_REP,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.LR_MILESTONES),
        gamma=cfg.LR_GAMMA,
    )
    return encoder, decoder, optimizer, scheduler


def encode_all_np(
    encoder: CausalEncoder,
    cfg: Any,
    data: SimulationData,
) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        latent_raw = encoder(data.Y.reshape(-1, cfg.P)).reshape(
            cfg.N, cfg.TRAIN_L, cfg.D
        )
        output = latent_raw.cpu().numpy().astype(np.float64)
    encoder.train()
    return output


def standardize_latents(
    latent_raw: np.ndarray,
    cfg: Any,
    data: SimulationData,
) -> dict[str, np.ndarray]:
    baseline = data.train_idx[data.I_np[data.train_idx].sum(1) == 0]
    if len(baseline) == 0:
        baseline = data.train_idx
    flat = latent_raw[baseline].reshape(-1, cfg.D)
    mean = flat.mean(0)
    std = flat.std(0) + 1e-8
    return {
        "Zraw": latent_raw,
        "zm": mean,
        "zs": std,
        "Z": (latent_raw - mean) / std,
    }


def weighted_ridge(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, float, np.ndarray]:
    weights = np.clip(weights, 1e-8, None)
    xtw = design.T * weights
    precision = xtw @ design + cfg.RIDGE * np.eye(design.shape[1])
    beta = np.linalg.solve(precision, xtw @ response)
    residual = response - design @ beta
    variance = float(
        np.clip(
            np.sum(weights * residual**2) / np.sum(weights),
            0.05**2,
            2.0**2,
        )
    )
    covariance = variance * np.linalg.inv(precision)
    return beta, variance, covariance


def fit_component_cache(
    latent: np.ndarray,
    gamma_column: np.ndarray,
    cfg: Any,
    data: SimulationData,
) -> dict:
    cache: dict = {}

    previous_train = latent[data.train_idx, :-1].reshape(-1, cfg.A)
    current_train = latent[data.train_idx, 1:].reshape(-1, cfg.A)
    intervention_train = np.repeat(
        data.I_np[data.train_idx], cfg.TRAIN_L - 1, axis=0
    )
    weight_train = np.repeat(gamma_column[data.train_idx], cfg.TRAIN_L - 1)

    previous_val = latent[data.val_idx, :-1].reshape(-1, cfg.A)
    current_val = latent[data.val_idx, 1:].reshape(-1, cfg.A)
    intervention_val = np.repeat(
        data.I_np[data.val_idx], cfg.TRAIN_L - 1, axis=0
    )
    weight_val = np.repeat(gamma_column[data.val_idx], cfg.TRAIN_L - 1)

    for child in range(cfg.A):
        other = [node for node in range(cfg.A) if node != child]
        lag_column = {node: 2 + index for index, node in enumerate(other)}
        inst_column = {
            node: 2 + len(other) + index for index, node in enumerate(other)
        }

        train_full = np.column_stack(
            [np.ones(len(previous_train)), previous_train[:, child]]
            + [np.tanh(previous_train[:, node]) for node in other]
            + [np.tanh(current_train[:, node]) for node in other]
        )
        val_full = np.column_stack(
            [np.ones(len(previous_val)), previous_val[:, child]]
            + [np.tanh(previous_val[:, node]) for node in other]
            + [np.tanh(current_val[:, node]) for node in other]
        )

        observational_train = intervention_train[:, child] < 0.5
        observational_val = intervention_val[:, child] < 0.5

        intervention_mask = ~observational_train
        intervention_response = current_train[intervention_mask, child]
        intervention_weight = weight_train[intervention_mask]
        intervention_mean = float(
            np.sum(intervention_weight * intervention_response)
            / (np.sum(intervention_weight) + 1e-12)
        )
        intervention_variance = float(
            np.clip(
                np.sum(
                    intervention_weight
                    * (intervention_response - intervention_mean) ** 2
                )
                / (np.sum(intervention_weight) + 1e-12),
                0.05**2,
                2.0**2,
            )
        )

        intervention_val_mask = ~observational_val
        intervention_val_response = current_val[intervention_val_mask, child]
        intervention_val_weight = weight_val[intervention_val_mask]
        intervention_log_likelihood = -0.5 * (
            np.log(2 * np.pi * intervention_variance)
            + (intervention_val_response - intervention_mean) ** 2
            / intervention_variance
        )

        response_train = current_train[observational_train, child]
        observational_weight_train = weight_train[observational_train]
        response_val = current_val[observational_val, child]
        observational_weight_val = weight_val[observational_val]

        for lag_bits in itertools.product([0, 1], repeat=len(other)):
            lag_parents = tuple(
                node for node, bit in zip(other, lag_bits) if bit
            )
            for inst_bits in itertools.product([0, 1], repeat=len(other)):
                inst_parents = tuple(
                    node for node, bit in zip(other, inst_bits) if bit
                )
                columns = (
                    [0, 1]
                    + [lag_column[node] for node in lag_parents]
                    + [inst_column[node] for node in inst_parents]
                )
                design_train = train_full[observational_train][:, columns]
                design_val = val_full[observational_val][:, columns]

                beta, variance, covariance = weighted_ridge(
                    design_train,
                    response_train,
                    observational_weight_train,
                    cfg,
                )
                log_likelihood = -0.5 * (
                    np.log(2 * np.pi * variance)
                    + (response_val - design_val @ beta) ** 2 / variance
                )
                total = float(
                    np.sum(observational_weight_val * log_likelihood)
                    + np.sum(intervention_val_weight * intervention_log_likelihood)
                )
                denominator = float(
                    np.sum(observational_weight_val)
                    + np.sum(intervention_val_weight)
                    + 1e-12
                )

                cache[(child, lag_parents, inst_parents)] = {
                    "avg_ll": total / denominator,
                    "beta": beta,
                    "beta_cov": covariance,
                    "obs_var": variance,
                    "int_mean": intervention_mean,
                    "int_var": intervention_variance,
                    "n_int": max(float(np.sum(intervention_weight)), 1.0),
                    "lag_parents": lag_parents,
                    "inst_parents": inst_parents,
                }
    return cache


def score_pairs(
    cache: dict,
    graph_space: GraphSpace,
    cfg: Any,
) -> np.ndarray:
    scores = np.empty(len(graph_space.graph_pairs))
    for index, (lag_graph, inst_graph) in enumerate(graph_space.graph_pairs):
        average_log_likelihood = np.mean(
            [
                cache[
                    (
                        child,
                        graph_space.parent_tuple(lag_graph, child),
                        graph_space.parent_tuple(inst_graph, child),
                    )
                ]["avg_ll"]
                for child in range(cfg.A)
            ]
        )
        scores[index] = (
            average_log_likelihood
            - cfg.LAMBDA_LAG_SCORE * lag_graph.sum()
            - cfg.LAMBDA_INST_SCORE * inst_graph.sum()
        )
    return scores


def H_for_pair(
    cache: dict,
    pair_index: int,
    graph_space: GraphSpace,
    cfg: Any,
) -> list[dict]:
    lag_graph, inst_graph = graph_space.graph_pairs[pair_index]
    return [
        deepcopy(
            cache[
                (
                    child,
                    graph_space.parent_tuple(lag_graph, child),
                    graph_space.parent_tuple(inst_graph, child),
                )
            ]
        )
        for child in range(cfg.A)
    ]


def canonical_dimension(cfg: Any) -> int:
    return 2 + 2 * cfg.A


def canonicalize(fit: dict, child: int, cfg: Any) -> np.ndarray:
    output = np.zeros(canonical_dimension(cfg))
    output[:2] = fit["beta"][:2]
    position = 2
    for parent in fit["lag_parents"]:
        output[2 + parent] = fit["beta"][position]
        position += 1
    for parent in fit["inst_parents"]:
        output[2 + cfg.A + parent] = fit["beta"][position]
        position += 1
    return output


def perturb_H(
    fit: dict,
    child: int,
    component: int,
    noise: dict[str, np.ndarray],
    cfg: Any,
) -> dict:
    canonical = canonicalize(fit, child, cfg)
    standard_deviation = np.zeros(canonical_dimension(cfg))
    diagonal = np.sqrt(np.clip(np.diag(fit["beta_cov"]), 0, None))
    standard_deviation[:2] = diagonal[:2]
    position = 2
    for parent in fit["lag_parents"]:
        standard_deviation[2 + parent] = diagonal[position]
        position += 1
    for parent in fit["inst_parents"]:
        standard_deviation[2 + cfg.A + parent] = diagonal[position]
        position += 1
    return {
        "beta_full": canonical
        + np.sqrt(cfg.TAU_H)
        * standard_deviation
        * noise["H"][component, child],
        "obs_var": float(
            np.clip(
                fit["obs_var"]
                * np.exp(cfg.TAU_LOGVAR * noise["lv"][component, child]),
                0.05**2,
                2.0**2,
            )
        ),
        "int_mean": float(
            fit["int_mean"]
            + np.sqrt(cfg.TAU_H * fit["int_var"] / fit["n_int"])
            * noise["im"][component, child]
        ),
        "int_var": float(
            np.clip(
                fit["int_var"]
                * np.exp(cfg.TAU_LOGVAR * noise["ilv"][component, child]),
                0.05**2,
                2.0**2,
            )
        ),
    }


def blend_H(old: Optional[dict], target: dict, cfg: Any) -> dict:
    if old is None:
        return deepcopy(target)
    return {
        "beta_full": (1 - cfg.ETA_H) * old["beta_full"]
        + cfg.ETA_H * target["beta_full"],
        "obs_var": float(
            np.exp(
                (1 - cfg.ETA_H) * np.log(old["obs_var"])
                + cfg.ETA_H * np.log(target["obs_var"])
            )
        ),
        "int_mean": float(
            (1 - cfg.ETA_H) * old["int_mean"]
            + cfg.ETA_H * target["int_mean"]
        ),
        "int_var": float(
            np.exp(
                (1 - cfg.ETA_H) * np.log(old["int_var"])
                + cfg.ETA_H * np.log(target["int_var"])
            )
        ),
    }


def trajectory_log_likelihood(
    latent: np.ndarray,
    pair_index: int,
    mechanisms: list[dict],
    graph_space: GraphSpace,
    cfg: Any,
    data: SimulationData,
) -> np.ndarray:
    lag_graph, inst_graph = graph_space.graph_pairs[int(pair_index)]
    output = np.zeros(cfg.N)
    previous, current = latent[:, :-1], latent[:, 1:]
    for child in range(cfg.A):
        mechanism = mechanisms[child]
        prediction = (
            mechanism["beta_full"][0]
            + mechanism["beta_full"][1] * previous[:, :, child]
        )
        for parent in range(cfg.A):
            prediction += (
                mechanism["beta_full"][2 + parent]
                * lag_graph[parent, child]
                * np.tanh(previous[:, :, parent])
            )
            prediction += (
                mechanism["beta_full"][2 + cfg.A + parent]
                * inst_graph[parent, child]
                * np.tanh(current[:, :, parent])
            )
        observational = data.I_np[:, child] < 0.5
        log_likelihood_observational = -0.5 * (
            np.log(2 * np.pi * mechanism["obs_var"])
            + (current[:, :, child] - prediction) ** 2
            / mechanism["obs_var"]
        )
        log_likelihood_intervention = -0.5 * (
            np.log(2 * np.pi * mechanism["int_var"])
            + (current[:, :, child] - mechanism["int_mean"]) ** 2
            / mechanism["int_var"]
        )
        output += np.where(
            observational[:, None],
            log_likelihood_observational,
            log_likelihood_intervention,
        ).sum(1) / (cfg.TRAIN_L - 1)
    return output / cfg.A


def update_responsibilities(
    latent: np.ndarray,
    pair_indices: np.ndarray,
    mechanisms: list[list[dict]],
    mixture_weights: np.ndarray,
    graph_space: GraphSpace,
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, np.ndarray]:
    log_likelihoods = np.stack(
        [
            trajectory_log_likelihood(
                latent,
                pair_indices[component],
                mechanisms[component],
                graph_space,
                cfg,
                data,
            )
            for component in range(len(pair_indices))
        ],
        axis=1,
    )
    logits = (
        np.log(np.clip(mixture_weights, 1e-12, None))[None, :]
        + log_likelihoods / cfg.RESP_TEMP
    )
    logits -= logits.max(1, keepdims=True)
    responsibilities = np.exp(logits)
    responsibilities /= responsibilities.sum(1, keepdims=True)
    responsibilities = np.maximum(responsibilities, cfg.RESP_FLOOR)
    responsibilities /= responsibilities.sum(1, keepdims=True)
    return responsibilities, log_likelihoods


def trajectory_features(latent: np.ndarray, cfg: Any) -> np.ndarray:
    rows = []
    for trajectory in latent:
        previous, current = trajectory[:-1], trajectory[1:]
        previous_centered = previous - previous.mean(0)
        current_centered = current - current.mean(0)
        lag_covariance = (
            current_centered.T @ previous_centered
        ) / max(len(previous) - 1, 1)
        covariance = (
            current_centered.T @ current_centered
        ) / max(len(current) - 1, 1)
        rows.append(
            np.r_[
                trajectory.mean(0),
                trajectory.std(0),
                lag_covariance.ravel(),
                covariance[np.triu_indices(cfg.A)],
            ]
        )
    features = np.asarray(rows)
    return (features - features.mean(0)) / (features.std(0) + 1e-6)


def initialize_mixture(
    latent: np.ndarray,
    k_fit: int,
    noise: dict[str, np.ndarray],
    graph_space: GraphSpace,
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]], list[dict]]:
    base_labels = KMeans(
        n_clusters=cfg.K_DATA,
        random_state=cfg.SEED,
        n_init=50,
    ).fit(trajectory_features(latent, cfg)).labels_

    responsibilities = np.full((cfg.N, k_fit), cfg.RESP_FLOOR)
    for base_component in range(cfg.K_DATA):
        fitted_components = [
            component
            for component in range(k_fit)
            if component % cfg.K_DATA == base_component
        ]
        mask = base_labels == base_component
        responsibilities[np.ix_(mask, fitted_components)] = 1.0 / len(
            fitted_components
        )
    responsibilities /= responsibilities.sum(1, keepdims=True)
    mixture_weights = responsibilities.mean(0)

    pair_indices: list[int] = []
    mechanisms: list[list[dict]] = []
    caches: list[dict] = []
    for component in range(k_fit):
        cache = fit_component_cache(
            latent, responsibilities[:, component], cfg, data
        )
        scores = score_pairs(cache, graph_space, cfg)
        pair_index = int(np.argmax(scores))
        pair_indices.append(pair_index)
        caches.append(cache)
        mechanisms.append(
            [
                perturb_H(fit, child, component, noise, cfg)
                for child, fit in enumerate(
                    H_for_pair(cache, pair_index, graph_space, cfg)
                )
            ]
        )
    return (
        responsibilities,
        mixture_weights,
        np.array(pair_indices, int),
        mechanisms,
        caches,
    )


def draw_particle_noise(
    k_fit: int,
    rng: np.random.Generator,
    cfg: Any,
) -> dict[str, np.ndarray]:
    return {
        "H": rng.normal(
            size=(k_fit, cfg.A, canonical_dimension(cfg))
        ),
        "lv": rng.normal(size=(k_fit, cfg.A)),
        "im": rng.normal(size=(k_fit, cfg.A)),
        "ilv": rng.normal(size=(k_fit, cfg.A)),
    }


def torch_standardize(
    latent_raw: torch.Tensor,
    state: dict[str, np.ndarray],
) -> torch.Tensor:
    mean = torch.tensor(
        state["zm"], dtype=latent_raw.dtype, device=latent_raw.device
    )
    std = torch.tensor(
        state["zs"], dtype=latent_raw.dtype, device=latent_raw.device
    )
    return (latent_raw - mean) / std


def structural_component_nll_torch(
    latent: torch.Tensor,
    intervention: torch.Tensor,
    pair_index: int,
    mechanisms: list[dict],
    graph_space: GraphSpace,
    cfg: Any,
) -> torch.Tensor:
    lag_np, inst_np = graph_space.graph_pairs[int(pair_index)]
    lag_graph = torch.tensor(
        lag_np, dtype=latent.dtype, device=latent.device
    )
    inst_graph = torch.tensor(
        inst_np, dtype=latent.dtype, device=latent.device
    )
    previous, current = latent[:, :-1], latent[:, 1:]
    output = torch.zeros(
        latent.shape[0], dtype=latent.dtype, device=latent.device
    )

    for child in range(cfg.A):
        mechanism = mechanisms[child]
        beta = torch.tensor(
            mechanism["beta_full"], dtype=latent.dtype, device=latent.device
        )
        obs_variance = torch.tensor(
            mechanism["obs_var"], dtype=latent.dtype, device=latent.device
        ).clamp_min(1e-8)
        int_mean = torch.tensor(
            mechanism["int_mean"], dtype=latent.dtype, device=latent.device
        )
        int_variance = torch.tensor(
            mechanism["int_var"], dtype=latent.dtype, device=latent.device
        ).clamp_min(1e-8)

        prediction = beta[0] + beta[1] * previous[:, :, child]
        for parent in range(cfg.A):
            prediction = prediction + (
                beta[2 + parent]
                * lag_graph[parent, child]
                * torch.tanh(previous[:, :, parent])
            )
            prediction = prediction + (
                beta[2 + cfg.A + parent]
                * inst_graph[parent, child]
                * torch.tanh(current[:, :, parent])
            )

        nll_observational = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(obs_variance)
            + (current[:, :, child] - prediction) ** 2 / obs_variance
        )
        nll_intervention = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(int_variance)
            + (current[:, :, child] - int_mean) ** 2 / int_variance
        )
        intervened = intervention[:, child] > 0.5
        output = output + torch.where(
            intervened[:, None], nll_intervention, nll_observational
        ).mean(1) / cfg.A
    return output


def structural_transition_nll_torch(
    latent: torch.Tensor,
    intervention: torch.Tensor,
    responsibility_batch: np.ndarray,
    pair_indices: np.ndarray,
    mechanisms: list[list[dict]],
    graph_space: GraphSpace,
    cfg: Any,
) -> torch.Tensor:
    component_nll = torch.stack(
        [
            structural_component_nll_torch(
                latent,
                intervention,
                pair_indices[component],
                mechanisms[component],
                graph_space,
                cfg,
            )
            for component in range(len(pair_indices))
        ],
        dim=1,
    )
    responsibilities = torch.tensor(
        responsibility_batch,
        dtype=latent.dtype,
        device=latent.device,
    )
    return torch.mean(torch.sum(responsibilities * component_nll, dim=1))


def graph_dist_to_truth(
    pair_index: int,
    graph_space: GraphSpace,
    data: SimulationData,
) -> list[int]:
    lag_graph, inst_graph = graph_space.graph_pairs[int(pair_index)]
    return [
        int(
            np.abs(lag_graph - data.true_model.lag_graphs[regime]).sum()
            + np.abs(inst_graph - data.true_model.inst_graphs[regime]).sum()
        )
        for regime in range(len(data.true_model.lag_graphs))
    ]
