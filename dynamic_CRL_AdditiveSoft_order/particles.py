from __future__ import annotations
import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from models import CausalEncoder, ResidualDecoder
from sim import SimulationData
from utils import is_dag


@dataclass
class GraphSpace:
    """Sparse registry of graph states visited by the moving particles.

    Unlike the old implementation, build() does not enumerate the graph space.
    New graph states are registered only when a particle proposes them.
    """

    lag_edge_list: list[tuple[int, int]]
    inst_pair_list: list[tuple[int, int]]
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
        space = cls(
            lag_edge_list=lag_edges,
            inst_pair_list=inst_pairs,
            graph_pairs=[],
            pair_lookup={},
        )
        zero = np.zeros((cfg.A, cfg.A), np.float32)
        space.register_pair(zero, zero)
        return space

    def parent_tuple(self, graph: np.ndarray, child: int) -> tuple[int, ...]:
        return tuple(np.where(graph[:, child] > 0.5)[0].tolist())

    def register_pair(
        self, lag_graph: np.ndarray, inst_graph: np.ndarray
    ) -> int:
        lag = np.asarray(lag_graph, dtype=np.float32)
        inst = np.asarray(inst_graph, dtype=np.float32)
        key = (lag.astype(np.int8).tobytes(), inst.astype(np.int8).tobytes())
        existing = self.pair_lookup.get(key)
        if existing is not None:
            return int(existing)
        index = len(self.graph_pairs)
        self.graph_pairs.append((lag.copy(), inst.copy()))
        self.pair_lookup[key] = index
        return index

    def pair_index(
        self, lag_graph: np.ndarray, inst_graph: np.ndarray
    ) -> int:
        return self.register_pair(lag_graph, inst_graph)

    @property
    def empty_pair_index(self) -> int:
        return 0

    def true_pair_indices(self, data: SimulationData) -> list[int]:
        return [
            self.register_pair(
                data.true_model.lag_graphs[regime],
                data.true_model.inst_graphs[regime],
            )
            for regime in range(len(data.true_model.lag_graphs))
        ]

    def project_pair_to_order(
        self,
        pair_index: int,
        order: tuple[int, ...],
    ) -> int:
        lag_graph, inst_graph = self.graph_pairs[int(pair_index)]
        position = {node: rank for rank, node in enumerate(order)}
        projected = inst_graph.copy()
        for parent in range(len(order)):
            for child in range(len(order)):
                if projected[parent, child] > 0.5 and position[parent] >= position[child]:
                    projected[parent, child] = 0.0
        return self.register_pair(lag_graph, projected)

    def propose_ordered_pair(
        self,
        pair_index: int,
        order: tuple[int, ...],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Symmetric one-edge toggle respecting the supplied instantaneous order."""
        lag_graph, inst_graph = self.graph_pairs[int(pair_index)]
        if rng.random() < 0.5 or len(self.inst_pair_list) == 0:
            parent, child = self.lag_edge_list[
                rng.integers(len(self.lag_edge_list))
            ]
            proposed_lag = lag_graph.copy()
            proposed_lag[parent, child] = 1.0 - proposed_lag[parent, child]
            return self.register_pair(proposed_lag, inst_graph), int(child)

        left, right = self.inst_pair_list[
            rng.integers(len(self.inst_pair_list))
        ]
        position = {node: rank for rank, node in enumerate(order)}
        if position[left] < position[right]:
            parent, child = left, right
        else:
            parent, child = right, left
        proposed_inst = inst_graph.copy()
        proposed_inst[parent, child] = 1.0 - proposed_inst[parent, child]
        proposed_inst[child, parent] = 0.0
        return self.register_pair(lag_graph, proposed_inst), int(child)


def pca_warmstart(
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=cfg.D, random_state=cfg.SEED).fit(
        data.Y_np[data.train_idx].reshape(-1, cfg.P)
    )
    return pca.components_.copy(), -pca.mean_ @ pca.components_.T


def intervention_aware_warmstart(
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA followed by one intervention-target-aware orthogonal rotation.

    Regime-specific intervention contrasts are combined through outer products,
    so the contrast may reverse sign across regimes without cancelling.
    """
    pca = PCA(n_components=cfg.D, random_state=cfg.SEED).fit(
        data.Y_np[data.train_idx].reshape(-1, cfg.P)
    )
    base_weight = pca.components_.copy()
    base_bias = -pca.mean_ @ base_weight.T
    latent = data.Y_np @ base_weight.T + base_bias

    baseline = data.train_idx[data.I_np[data.train_idx].sum(1) == 0]
    if len(baseline) < max(cfg.K_DATA, 2):
        return base_weight, base_bias
    flat = latent[baseline].reshape(-1, cfg.D)
    mean = flat.mean(0)
    std = flat.std(0) + 1e-8
    latent_std = (latent - mean) / std

    labels = KMeans(
        n_clusters=cfg.K_DATA,
        random_state=cfg.SEED,
        n_init=50,
    ).fit(trajectory_features(latent_std, cfg)).labels_

    directions: list[np.ndarray] = []
    for target in range(cfg.A):
        scatter = np.zeros((cfg.D, cfg.D), dtype=np.float64)
        reference = None
        for component in range(cfg.K_DATA):
            control_ids = np.where(
                (labels == component) & (data.I_np.sum(1) == 0)
            )[0]
            target_ids = np.where(
                (labels == component) & (data.I_np[:, target] > 0.5)
            )[0]
            control_ids = np.intersect1d(control_ids, data.train_idx)
            target_ids = np.intersect1d(target_ids, data.train_idx)
            if len(control_ids) == 0 or len(target_ids) == 0:
                continue
            contrast = (
                latent_std[target_ids].mean(axis=(0, 1))
                - latent_std[control_ids].mean(axis=(0, 1))
            )
            norm = np.linalg.norm(contrast)
            if norm <= 1e-8:
                continue
            unit = contrast / norm
            scatter += np.outer(unit, unit)
            if reference is None:
                reference = contrast
        if reference is None or np.linalg.norm(scatter) <= 1e-8:
            return base_weight, base_bias
        _, eigenvectors = np.linalg.eigh(scatter)
        direction = eigenvectors[:, -1]
        if float(direction @ reference) < 0.0:
            direction = -direction
        directions.append(direction)

    raw_rotation = np.stack(directions, axis=0)
    left, _, right_t = np.linalg.svd(raw_rotation, full_matrices=False)
    rotation = left @ right_t
    for target in range(cfg.A):
        if float(rotation[target] @ raw_rotation[target]) < 0.0:
            rotation[target] *= -1.0

    whitening = np.diag(1.0 / std)
    weight = rotation @ whitening @ base_weight
    bias = rotation @ whitening @ (base_bias - mean)
    return weight.astype(np.float32), bias.astype(np.float32)


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
    weight, bias = intervention_aware_warmstart(cfg, data)
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



def _clip_self_coefficients(alpha: np.ndarray, cfg: Any) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64).copy()
    if alpha.shape[0] >= 2:
        rho = float(getattr(cfg, "SELF_COEF_MAX", 0.95))
        alpha[1] = float(np.clip(alpha[1], -rho, rho))
    return alpha


def _stabilize_mechanism(alpha0: np.ndarray, alpha1: np.ndarray, obs_var: float, int_var: float, cfg: Any) -> dict:
    alpha0 = _clip_self_coefficients(alpha0, cfg)
    alpha1 = _clip_self_coefficients(alpha1, cfg)
    return {
        "alpha0": alpha0,
        "delta_alpha": alpha1 - alpha0,
        "obs_var": float(np.clip(obs_var, 0.05**2, 2.0**2)),
        "int_var": float(np.clip(int_var, 0.05**2, 2.0**2)),
    }

def weighted_ridge(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_features = design.shape[1]
    if design.shape[0] == 0:
        precision = cfg.RIDGE * np.eye(n_features)
        beta = np.zeros(n_features, dtype=np.float64)
        variance = 1.0
        covariance = variance * np.linalg.inv(precision)
        return beta, variance, covariance
    weights = np.clip(weights, 1e-8, None)
    xtw = design.T * weights
    precision = xtw @ design + cfg.RIDGE * np.eye(n_features)
    beta = np.linalg.solve(precision, xtw @ response)
    beta = _clip_self_coefficients(beta, cfg)
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


def coefficient_dimension(cfg: Any) -> int:
    return 2 + 2 * cfg.A


def coefficient_mask(
    lag_graph: np.ndarray,
    inst_graph: np.ndarray,
    child: int,
    cfg: Any,
) -> np.ndarray:
    return np.r_[
        np.ones(2, np.float32),
        lag_graph[:, child].astype(np.float32),
        inst_graph[:, child].astype(np.float32),
    ]


def features_for_graph_np(
    previous: np.ndarray,
    current: np.ndarray,
    lag_graph: np.ndarray,
    inst_graph: np.ndarray,
    child: int,
) -> np.ndarray:
    return np.concatenate(
        [
            np.ones((*previous.shape[:-1], 1), dtype=np.float64),
            previous[..., child : child + 1],
            np.tanh(previous) * lag_graph[:, child],
            np.tanh(current) * inst_graph[:, child],
        ],
        axis=-1,
    )


def _parent_configurations(
    child: int, cfg: Any
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    other = [node for node in range(cfg.A) if node != child]
    output = []
    for lag_bits in itertools.product([0, 1], repeat=len(other)):
        lag_parents = tuple(node for node, bit in zip(other, lag_bits) if bit)
        for inst_bits in itertools.product([0, 1], repeat=len(other)):
            inst_parents = tuple(node for node, bit in zip(other, inst_bits) if bit)
            output.append((lag_parents, inst_parents))
    return output


def _graphs_from_parents(
    child: int,
    lag_parents: tuple[int, ...],
    inst_parents: tuple[int, ...],
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray]:
    lag_graph = np.zeros((cfg.A, cfg.A), np.float32)
    inst_graph = np.zeros((cfg.A, cfg.A), np.float32)
    if lag_parents:
        lag_graph[list(lag_parents), child] = 1.0
    if inst_parents:
        inst_graph[list(inst_parents), child] = 1.0
    return lag_graph, inst_graph


def _expand_fit(
    beta: np.ndarray,
    covariance: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.where(mask > 0.5)[0]
    full_beta = np.zeros(len(mask), dtype=np.float64)
    full_covariance = np.zeros((len(mask), len(mask)), dtype=np.float64)
    full_beta[indices] = beta
    full_covariance[np.ix_(indices, indices)] = covariance
    return full_beta, full_covariance


def fit_component_cache(
    latent: np.ndarray,
    gamma_column: np.ndarray,
    cfg: Any,
    data: SimulationData,
    graph_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    cache: dict = {}
    previous_train = latent[data.train_idx, :-1].reshape(-1, cfg.A)
    current_train = latent[data.train_idx, 1:].reshape(-1, cfg.A)
    intervention_train = np.repeat(
        data.I_np[data.train_idx], cfg.TRAIN_L - 1, axis=0
    )
    weight_train = np.repeat(
        gamma_column[data.train_idx], cfg.TRAIN_L - 1
    )
    previous_val = latent[data.val_idx, :-1].reshape(-1, cfg.A)
    current_val = latent[data.val_idx, 1:].reshape(-1, cfg.A)
    intervention_val = np.repeat(
        data.I_np[data.val_idx], cfg.TRAIN_L - 1, axis=0
    )
    weight_val = np.repeat(gamma_column[data.val_idx], cfg.TRAIN_L - 1)

    for child in range(cfg.A):
        observational_train = intervention_train[:, child] < 0.5
        interventional_train = ~observational_train
        observational_val = intervention_val[:, child] < 0.5
        interventional_val = ~observational_val

        if graph_pair is None:
            configurations = _parent_configurations(child, cfg)
        else:
            lag_full, inst_full = graph_pair
            configurations = [
                (
                    tuple(np.where(lag_full[:, child] > 0.5)[0].tolist()),
                    tuple(np.where(inst_full[:, child] > 0.5)[0].tolist()),
                )
            ]

        for lag_parents, inst_parents in configurations:
            lag_graph, inst_graph = _graphs_from_parents(
                child, lag_parents, inst_parents, cfg
            )
            mask = coefficient_mask(lag_graph, inst_graph, child, cfg)
            indices = np.where(mask > 0.5)[0]
            design_train_full = features_for_graph_np(
                previous_train, current_train, lag_graph, inst_graph, child
            )
            design_val_full = features_for_graph_np(
                previous_val, current_val, lag_graph, inst_graph, child
            )
            design_train = design_train_full[:, indices]
            design_val = design_val_full[:, indices]

            alpha0_active, obs_var, alpha0_cov_active = weighted_ridge(
                design_train[observational_train],
                current_train[observational_train, child],
                weight_train[observational_train],
                cfg,
            )
            alpha1_active, int_var, alpha1_cov_active = weighted_ridge(
                design_train[interventional_train],
                current_train[interventional_train, child],
                weight_train[interventional_train],
                cfg,
            )
            alpha0, alpha0_cov = _expand_fit(
                alpha0_active, alpha0_cov_active, mask
            )
            alpha1, alpha1_cov = _expand_fit(
                alpha1_active, alpha1_cov_active, mask
            )

            obs_residual = (
                current_val[observational_val, child]
                - design_val[observational_val] @ alpha0_active
            )
            int_residual = (
                current_val[interventional_val, child]
                - design_val[interventional_val] @ alpha1_active
            )
            obs_ll = -0.5 * (
                np.log(2 * np.pi * obs_var) + obs_residual**2 / obs_var
            )
            int_ll = -0.5 * (
                np.log(2 * np.pi * int_var) + int_residual**2 / int_var
            )
            total = float(
                np.sum(weight_val[observational_val] * obs_ll)
                + np.sum(weight_val[interventional_val] * int_ll)
            )
            denominator = float(
                np.sum(weight_val[observational_val])
                + np.sum(weight_val[interventional_val])
                + 1e-12
            )
            cache[(child, lag_parents, inst_parents)] = {
                "avg_ll": total / denominator,
                "alpha0": alpha0,
                "alpha1": alpha1,
                "alpha0_cov": alpha0_cov,
                "alpha1_cov": alpha1_cov,
                "obs_var": obs_var,
                "int_var": int_var,
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
    output = []
    for child in range(cfg.A):
        fit = deepcopy(
            cache[
                (
                    child,
                    graph_space.parent_tuple(lag_graph, child),
                    graph_space.parent_tuple(inst_graph, child),
                )
            ]
        )
        fit["alpha0"] = _clip_self_coefficients(fit["alpha0"], cfg)
        alpha1 = _clip_self_coefficients(fit["alpha1"], cfg)
        fit["alpha1"] = alpha1
        output.append(fit)
    return output


def perturb_H(
    fit: dict,
    child: int,
    component: int,
    noise: dict[str, np.ndarray],
    cfg: Any,
) -> dict:
    alpha0_sd = np.sqrt(np.clip(np.diag(fit["alpha0_cov"]), 0, None))
    alpha1_sd = np.sqrt(np.clip(np.diag(fit["alpha1_cov"]), 0, None))
    alpha0 = fit["alpha0"] + np.sqrt(cfg.TAU_H) * alpha0_sd * noise[
        "a0"
    ][component, child]
    alpha1 = fit["alpha1"] + np.sqrt(cfg.TAU_H) * alpha1_sd * noise[
        "a1"
    ][component, child]
    obs_var = float(
        fit["obs_var"]
        * np.exp(cfg.TAU_LOGVAR * noise["lv0"][component, child])
    )
    int_var = float(
        fit["int_var"]
        * np.exp(cfg.TAU_LOGVAR * noise["lv1"][component, child])
    )
    return _stabilize_mechanism(alpha0, alpha1, obs_var, int_var, cfg)


def mechanism_from_fit(fit: dict, cfg: Any) -> dict:
    """Convert a weighted-ridge fit to the deterministic Block-1 mechanism."""
    return _stabilize_mechanism(
        fit["alpha0"],
        fit["alpha1"],
        fit["obs_var"],
        fit["int_var"],
        cfg,
    )


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
        design = features_for_graph_np(
            previous, current, lag_graph, inst_graph, child
        )
        intervention = data.I_np[:, child]
        coefficients = (
            mechanism["alpha0"][None, None, :]
            + intervention[:, None, None]
            * mechanism["delta_alpha"][None, None, :]
        )
        prediction = np.sum(design * coefficients, axis=-1)
        variance = np.where(
            intervention > 0.5,
            mechanism["int_var"],
            mechanism["obs_var"],
        )
        log_likelihood = -0.5 * (
            np.log(2 * np.pi * variance[:, None])
            + (current[:, :, child] - prediction) ** 2
            / variance[:, None]
        )
        output += log_likelihood.sum(axis=1)
    return output


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

    # Start every graph particle at the empty graph.  No graph enumeration is
    # performed here; the soft-intervention anchored MH moves build the graph
    # locally after burn-in.
    pair_indices = np.full(k_fit, graph_space.empty_pair_index, dtype=int)
    mechanisms: list[list[dict]] = []
    caches: list[dict] = []
    empty_pair = graph_space.graph_pairs[graph_space.empty_pair_index]
    for component in range(k_fit):
        cache = fit_component_cache(
            latent,
            responsibilities[:, component],
            cfg,
            data,
            graph_pair=empty_pair,
        )
        caches.append(cache)
        mechanisms.append(
            [
                mechanism_from_fit(fit, cfg)
                for fit in H_for_pair(
                    cache, graph_space.empty_pair_index, graph_space, cfg
                )
            ]
        )
    return responsibilities, mixture_weights, pair_indices, mechanisms, caches


def draw_particle_noise(
    k_fit: int,
    rng: np.random.Generator,
    cfg: Any,
) -> dict[str, np.ndarray]:
    shape = (k_fit, cfg.A, coefficient_dimension(cfg))
    return {
        "a0": rng.normal(size=shape),
        "a1": rng.normal(size=shape),
        "lv0": rng.normal(size=(k_fit, cfg.A)),
        "lv1": rng.normal(size=(k_fit, cfg.A)),
    }


def _features_for_graph_torch(
    previous: torch.Tensor,
    current: torch.Tensor,
    lag_graph: torch.Tensor,
    inst_graph: torch.Tensor,
    child: int,
) -> torch.Tensor:
    ones = torch.ones(
        *previous.shape[:-1], 1, dtype=previous.dtype, device=previous.device
    )
    return torch.cat(
        [
            ones,
            previous[..., child : child + 1],
            torch.tanh(previous) * lag_graph[:, child],
            torch.tanh(current) * inst_graph[:, child],
        ],
        dim=-1,
    )


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
        features = _features_for_graph_torch(
            previous, current, lag_graph, inst_graph, child
        )
        alpha0 = torch.tensor(
            mechanism["alpha0"], dtype=latent.dtype, device=latent.device
        )
        delta_alpha = torch.tensor(
            mechanism["delta_alpha"],
            dtype=latent.dtype,
            device=latent.device,
        )
        intervened = intervention[:, child]
        coefficients = (
            alpha0[None, None, :]
            + intervened[:, None, None] * delta_alpha[None, None, :]
        )
        prediction = torch.sum(features * coefficients, dim=-1)
        obs_variance = torch.tensor(
            mechanism["obs_var"], dtype=latent.dtype, device=latent.device
        ).clamp_min(1e-8)
        int_variance = torch.tensor(
            mechanism["int_var"], dtype=latent.dtype, device=latent.device
        ).clamp_min(1e-8)
        variance = torch.where(
            intervened[:, None] > 0.5, int_variance, obs_variance
        )
        nll = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(variance)
            + (current[:, :, child] - prediction) ** 2 / variance
        )
        output = output + nll.mean(1) / cfg.A
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

# -----------------------------------------------------------------------------
# Soft-intervention graph selector
# -----------------------------------------------------------------------------

def _quantile_wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(y) == 0:
        return np.inf
    q = np.linspace(0.01, 0.99, 99)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def _intervention_mean_shifts(
    latent: np.ndarray,
    labels: np.ndarray,
    component: int,
    cfg: Any,
    data: SimulationData,
) -> np.ndarray:
    ids = np.where(labels == component)[0]
    obs = ids[data.I_np[ids].sum(1) == 0]
    if len(obs) == 0:
        raise ValueError("each component needs observational trajectories")
    obs_mean = latent[obs].reshape(-1, cfg.A).mean(0)
    shifts = []
    for target in range(cfg.A):
        target_ids = ids[data.I_np[ids, target] > 0.5]
        if len(target_ids) == 0:
            raise ValueError("each component needs every single-target intervention")
        target_mean = latent[target_ids].reshape(-1, cfg.A).mean(0)
        shifts.append(target_mean - obs_mean)
    return np.stack(shifts)


def _generalized_min_direction(
    forbidden: np.ndarray,
    own: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    metric = own + eps * np.eye(own.shape[0])
    evals, evecs = np.linalg.eigh(metric)
    whitening = (
        evecs
        @ np.diag(1.0 / np.sqrt(np.maximum(evals, eps)))
        @ evecs.T
    )
    transformed = whitening @ forbidden @ whitening
    values, vectors = np.linalg.eigh(transformed)
    direction = whitening @ vectors[:, np.argmin(values)]
    return direction / (np.linalg.norm(direction) + 1e-12)


def _soft_order_demixing(
    latent: np.ndarray,
    labels: np.ndarray,
    orders: tuple[tuple[int, ...], ...],
    cfg: Any,
    data: SimulationData,
) -> np.ndarray:
    shifts = [
        _intervention_mean_shifts(latent, labels, component, cfg, data)
        for component in range(cfg.K_FIT)
    ]
    columns = []
    for child in range(cfg.A):
        forbidden = np.zeros((cfg.A, cfg.A), dtype=np.float64)
        own = np.zeros((cfg.A, cfg.A), dtype=np.float64)
        reference = np.zeros(cfg.A, dtype=np.float64)
        for component, order in enumerate(orders):
            position = {node: rank for rank, node in enumerate(order)}
            for target in range(cfg.A):
                direction = shifts[component][target][:, None]
                if position[target] > position[child]:
                    forbidden += direction @ direction.T
            direction = shifts[component][child][:, None]
            own += direction @ direction.T
            reference += shifts[component][child]
        column = _generalized_min_direction(forbidden, own)
        if float(column @ reference) < 0.0:
            column = -column
        columns.append(column)
    return np.stack(columns, axis=1)


def _soft_order_score(
    corrected: np.ndarray,
    labels: np.ndarray,
    orders: tuple[tuple[int, ...], ...],
    cfg: Any,
    data: SimulationData,
) -> float:
    forbidden_change = 0.0
    own_change = 0.0
    for component, order in enumerate(orders):
        ids = np.where(labels == component)[0]
        obs = ids[data.I_np[ids].sum(1) == 0]
        if len(obs) == 0:
            return np.inf
        position = {node: rank for rank, node in enumerate(order)}
        for target in range(cfg.A):
            target_ids = ids[data.I_np[ids, target] > 0.5]
            if len(target_ids) == 0:
                return np.inf
            for child in range(cfg.A):
                obs_values = corrected[obs, :, child].reshape(-1)
                target_values = corrected[target_ids, :, child].reshape(-1)
                scale = float(np.std(obs_values) + 1e-8)
                change = _quantile_wasserstein_1d(
                    obs_values, target_values
                ) / scale
                if position[target] > position[child]:
                    forbidden_change += change**2
                if target == child:
                    own_change += change**2
    return forbidden_change / (own_change + 1e-8)


def _selector_standardize(
    latent: np.ndarray,
    cfg: Any,
    data: SimulationData,
) -> np.ndarray:
    baseline = np.where(data.I_np.sum(1) == 0)[0]
    if len(baseline) == 0:
        baseline = np.arange(cfg.N)
    flat = latent[baseline].reshape(-1, cfg.A)
    mean = flat.mean(0)
    std = flat.std(0) + 1e-8
    return (latent - mean) / std


def _selector_design(
    previous: np.ndarray,
    current: np.ndarray,
    child: int,
    lag_parents: tuple[int, ...],
    inst_parents: tuple[int, ...],
    degree: int,
) -> np.ndarray:
    columns = [
        np.ones((len(previous), 1), dtype=np.float64),
        previous[:, child : child + 1],
    ]
    for parent in lag_parents:
        value = previous[:, parent : parent + 1]
        columns.extend(value**power for power in range(1, degree + 1))
    for parent in inst_parents:
        value = current[:, parent : parent + 1]
        columns.extend(value**power for power in range(1, degree + 1))
    return np.concatenate(columns, axis=1)


def _selector_ridge(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, float]:
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None)
    precision = (
        design.T @ (weights[:, None] * design)
        + cfg.RIDGE * np.eye(design.shape[1])
    )
    beta = np.linalg.solve(precision, design.T @ (weights * response))
    residual = response - design @ beta
    variance = float(
        np.clip(
            np.sum(weights * residual**2) / np.sum(weights),
            0.05**2,
            2.0**2,
        )
    )
    return beta, variance


def _prepare_selector_cache(
    latent: np.ndarray,
    gamma_column: np.ndarray,
    cfg: Any,
    data: SimulationData,
) -> dict:
    return {
        "degree": int(getattr(cfg, "SOFT_GRAPH_POLY_DEGREE", 3)),
        "previous_train": latent[data.train_idx, :-1].reshape(-1, cfg.A),
        "current_train": latent[data.train_idx, 1:].reshape(-1, cfg.A),
        "intervention_train": np.repeat(
            data.I_np[data.train_idx], cfg.TRAIN_L - 1, axis=0
        ),
        "weight_train": np.repeat(
            gamma_column[data.train_idx], cfg.TRAIN_L - 1
        ),
        "previous_val": latent[data.val_idx, :-1].reshape(-1, cfg.A),
        "current_val": latent[data.val_idx, 1:].reshape(-1, cfg.A),
        "intervention_val": np.repeat(
            data.I_np[data.val_idx], cfg.TRAIN_L - 1, axis=0
        ),
        "weight_val": np.repeat(
            gamma_column[data.val_idx], cfg.TRAIN_L - 1
        ),
        "scores": {},
    }


def _selector_parent_score(
    cache: dict,
    child: int,
    lag_parents: tuple[int, ...],
    inst_parents: tuple[int, ...],
    cfg: Any,
) -> float:
    key = (int(child), tuple(lag_parents), tuple(inst_parents))
    if key in cache["scores"]:
        return float(cache["scores"][key])

    previous_train = cache["previous_train"]
    current_train = cache["current_train"]
    intervention_train = cache["intervention_train"]
    weight_train = cache["weight_train"]
    previous_val = cache["previous_val"]
    current_val = cache["current_val"]
    intervention_val = cache["intervention_val"]
    weight_val = cache["weight_val"]

    design_train = _selector_design(
        previous_train,
        current_train,
        child,
        lag_parents,
        inst_parents,
        cache["degree"],
    )
    design_val = _selector_design(
        previous_val,
        current_val,
        child,
        lag_parents,
        inst_parents,
        cache["degree"],
    )
    obs_train = intervention_train[:, child] < 0.5
    int_train = ~obs_train
    obs_val = intervention_val[:, child] < 0.5
    int_val = ~obs_val

    beta0, var0 = _selector_ridge(
        design_train[obs_train],
        current_train[obs_train, child],
        weight_train[obs_train],
        cfg,
    )
    beta1, var1 = _selector_ridge(
        design_train[int_train],
        current_train[int_train, child],
        weight_train[int_train],
        cfg,
    )
    residual0 = current_val[obs_val, child] - design_val[obs_val] @ beta0
    residual1 = current_val[int_val, child] - design_val[int_val] @ beta1
    ll0 = -0.5 * (np.log(2 * np.pi * var0) + residual0**2 / var0)
    ll1 = -0.5 * (np.log(2 * np.pi * var1) + residual1**2 / var1)
    total = float(
        np.sum(weight_val[obs_val] * ll0)
        + np.sum(weight_val[int_val] * ll1)
    )
    score = total / (np.sum(weight_val) + 1e-12)
    cache["scores"][key] = float(score)
    return float(score)


def _selector_local_graph_score(
    pair_index: int,
    child: int,
    cache: dict,
    graph_space: GraphSpace,
    cfg: Any,
) -> float:
    lag_graph, inst_graph = graph_space.graph_pairs[int(pair_index)]
    lag_parents = graph_space.parent_tuple(lag_graph, child)
    inst_parents = graph_space.parent_tuple(inst_graph, child)
    average_term = _selector_parent_score(
        cache, child, lag_parents, inst_parents, cfg
    ) / cfg.A
    edge_penalty = float(getattr(cfg, "SOFT_GRAPH_EDGE_PENALTY", 0.002))
    incoming_edges = len(lag_parents) + len(inst_parents)
    return average_term - edge_penalty * incoming_edges


def infer_soft_intervention_anchor(
    latent: np.ndarray,
    gamma: np.ndarray,
    cfg: Any,
    data: SimulationData,
) -> dict:
    """Infer the intervention-anchored coordinate correction and causal orders.

    This step uses only learned latents, observed intervention labels, and fitted
    regime responsibilities.  It does not use true latents, true regimes, or
    true graphs.
    """
    labels = np.asarray(gamma).argmax(1)
    permutations = list(itertools.permutations(range(cfg.A)))
    candidate_count = len(permutations) ** cfg.K_FIT
    if candidate_count > 10000:
        raise ValueError(
            "exact order search is intended for the current small-A experiment"
        )

    best_score = np.inf
    best_orders = None
    best_demixing = None
    for orders in itertools.product(permutations, repeat=cfg.K_FIT):
        try:
            demixing = _soft_order_demixing(
                latent, labels, orders, cfg, data
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(demixing).all():
            continue
        condition = np.linalg.cond(demixing)
        if not np.isfinite(condition) or condition > 1e4:
            continue
        corrected = latent @ demixing
        score = _soft_order_score(corrected, labels, orders, cfg, data)
        if score < best_score:
            best_score = score
            best_orders = orders
            best_demixing = demixing

    if best_orders is None or best_demixing is None:
        raise RuntimeError("soft intervention anchor found no valid order")

    corrected = _selector_standardize(latent @ best_demixing, cfg, data)
    return {
        "orders": tuple(tuple(map(int, order)) for order in best_orders),
        "demixing": best_demixing,
        "order_score": float(best_score),
        "corrected": corrected,
    }


def move_soft_graph_particle(
    pair_index: int,
    order: tuple[int, ...],
    corrected: np.ndarray,
    gamma_column: np.ndarray,
    graph_space: GraphSpace,
    rng: np.random.Generator,
    cfg: Any,
    data: SimulationData,
) -> tuple[int, int, float]:
    """Move one graph particle using only local anchored-score evaluations."""
    current = graph_space.project_pair_to_order(pair_index, order)
    cache = _prepare_selector_cache(corrected, gamma_column, cfg, data)
    accepted = 0
    temperature = max(float(cfg.TAU_GRAPH), 1e-12)

    for _ in range(int(cfg.GRAPH_MOVES_PER_OUTER)):
        proposal, child = graph_space.propose_ordered_pair(current, order, rng)
        if proposal == current:
            continue
        current_local = _selector_local_graph_score(
            current, child, cache, graph_space, cfg
        )
        proposal_local = _selector_local_graph_score(
            proposal, child, cache, graph_space, cfg
        )
        log_alpha = (proposal_local - current_local) / temperature
        if np.log(rng.random()) < min(0.0, log_alpha):
            current = proposal
            accepted += 1

    total_score = sum(
        _selector_local_graph_score(current, child, cache, graph_space, cfg)
        for child in range(cfg.A)
    )
    return int(current), int(accepted), float(total_score)

