from __future__ import annotations
from typing import Any, Optional

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fm import (
    FMMetadata,
    H_target_mask,
    inst_labels_to_graph,
    lag_labels_to_graph,
    pack_H_scaled,
    prepare_context_for_test,
    sample_H_given_fixed_graph,
    sample_fm_given_context,
)
from particles import GraphSpace, H_for_pair, fit_component_cache, graph_dist_to_truth
from sim import SimulationData
from utils import graph_distance, is_dag


def block1_summary(
    result: dict,
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": np.arange(cfg.K_FIT) + 1,
            "pi": result["pi"],
            "effective N": result["gamma"].sum(0),
            "graph pair index": result["pairs"],
            "distance to truth 1": [
                graph_dist_to_truth(pair, graph_space, data)[0]
                for pair in result["pairs"]
            ],
            "distance to truth 2": [
                graph_dist_to_truth(pair, graph_space, data)[1]
                for pair in result["pairs"]
            ],
        }
    )


def latent_correlation_table(
    result: dict,
    cfg: Any,
    data: SimulationData,
) -> pd.DataFrame:
    learned = result["state"]["Z"]
    correlation = np.corrcoef(
        learned[data.train_idx].reshape(-1, cfg.A).T,
        data.Z_true_np[data.train_idx].reshape(-1, cfg.A).T,
    )[: cfg.A, cfg.A :]
    return pd.DataFrame(
        correlation,
        index=[f"learned z{index + 1}" for index in range(cfg.A)],
        columns=[f"true z{index + 1}" for index in range(cfg.A)],
    )


def affine_align_latents_to_truth(
    learned: np.ndarray,
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Diagnostic only: align learned latents to true latents by permutation + 1D affine maps.

    The permutation and affine coefficients are fit using the training split only,
    then applied to every trajectory.
    """
    learned_train = learned[data.train_idx].reshape(-1, cfg.A)
    truth_train = data.Z_true_np[data.train_idx].reshape(-1, cfg.A)

    best = None
    for permutation in itertools.permutations(range(cfg.A)):
        slopes = np.zeros(cfg.A, dtype=np.float64)
        intercepts = np.zeros(cfg.A, dtype=np.float64)
        squared_error = 0.0
        for true_index, learned_index in enumerate(permutation):
            x = learned_train[:, learned_index].astype(np.float64)
            y = truth_train[:, true_index].astype(np.float64)
            x_centered = x - x.mean()
            denominator = float(np.dot(x_centered, x_centered))
            slope = (
                float(np.dot(x_centered, y - y.mean())) / denominator
                if denominator > 1e-12
                else 0.0
            )
            intercept = float(y.mean() - slope * x.mean())
            residual = y - (slope * x + intercept)
            slopes[true_index] = slope
            intercepts[true_index] = intercept
            squared_error += float(np.dot(residual, residual))
        if best is None or squared_error < best[0]:
            best = (squared_error, permutation, slopes, intercepts)

    _, permutation, slopes, intercepts = best
    aligned = np.empty_like(learned, dtype=np.float64)
    rows = []
    for true_index, learned_index in enumerate(permutation):
        aligned[..., true_index] = (
            slopes[true_index] * learned[..., learned_index]
            + intercepts[true_index]
        )
        x = learned_train[:, learned_index]
        y = truth_train[:, true_index]
        fitted = slopes[true_index] * x + intercepts[true_index]
        rows.append(
            {
                "true latent": f"z{true_index + 1}",
                "learned latent": f"z{learned_index + 1}",
                "slope": float(slopes[true_index]),
                "intercept": float(intercepts[true_index]),
                "train RMSE": float(np.sqrt(np.mean((fitted - y) ** 2))),
                "train corr": float(np.corrcoef(fitted, y)[0, 1]),
            }
        )
    return aligned, pd.DataFrame(rows)


def plot_stage0_trace(stage0: dict) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.plot(np.arange(1, len(stage0["trace"]) + 1), stage0["trace"])
    plt.xlabel("optimization step")
    plt.ylabel("reconstruction MSE")
    plt.title("Stage 0: observation-compressor training")
    plt.tight_layout()
    plt.show()


def plot_training_trace(result: dict) -> None:
    trace = result["trace"]
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.5), constrained_layout=True
    )
    axes[0].plot(trace["outer"], trace["ARI"])
    axes[0].set_title("Block 1: trajectory-regime ARI")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(trace["outer"], trace["fm_lag"], label="lag")
    axes[1].plot(trace["outer"], trace["fm_inst"], label="inst")
    axes[1].set_title("Discrete FM losses")
    axes[1].legend()
    axes[2].plot(trace["outer"], trace["fm_H"])
    axes[2].set_title("Continuous H-FM loss")
    for axis in axes:
        axis.set_xlabel("outer iteration")
    plt.show()


def evaluate_test_posterior(
    result: dict,
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    metadata: FMMetadata,
) -> dict:
    test_context = prepare_context_for_test(
        result["context_net"],
        data.Y_test_np,
        data.I_test_np,
        cfg,
        data.device,
    )
    lag_labels: list[np.ndarray] = []
    inst_labels: list[np.ndarray] = []
    h_samples: list[np.ndarray] = []
    sample_owner: list[int] = []
    for test_index in range(cfg.N_TEST):
        lag, inst, h = sample_fm_given_context(
            result["fm"],
            test_context[test_index],
            cfg,
            metadata,
            graph_space,
            data.device,
            n_samples=cfg.FM_SAMPLES_PER_TEST,
        )
        lag_labels.append(lag)
        inst_labels.append(inst)
        h_samples.append(h)
        sample_owner.extend([test_index] * cfg.FM_SAMPLES_PER_TEST)
    lag_labels_array = np.concatenate(lag_labels)
    inst_labels_array = np.concatenate(inst_labels)
    h_array = np.concatenate(h_samples)
    owner_array = np.asarray(sample_owner)
    lag_graphs = np.stack(
        [
            lag_labels_to_graph(labels, graph_space, cfg)
            for labels in lag_labels_array
        ]
    )
    inst_graphs = np.stack(
        [
            inst_labels_to_graph(labels, graph_space, cfg)
            for labels in inst_labels_array
        ]
    )
    dag_validity = np.mean([is_dag(graph) for graph in inst_graphs])
    lag_exact = np.mean(
        [
            np.array_equal(
                lag_graphs[sample],
                data.true_model.lag_graphs[
                    data.test_regime_labels[owner_array[sample]]
                ],
            )
            for sample in range(len(owner_array))
        ]
    )
    inst_exact = np.mean(
        [
            np.array_equal(
                inst_graphs[sample],
                data.true_model.inst_graphs[
                    data.test_regime_labels[owner_array[sample]]
                ],
            )
            for sample in range(len(owner_array))
        ]
    )
    joint_exact = np.mean(
        [
            np.array_equal(
                lag_graphs[sample],
                data.true_model.lag_graphs[
                    data.test_regime_labels[owner_array[sample]]
                ],
            )
            and np.array_equal(
                inst_graphs[sample],
                data.true_model.inst_graphs[
                    data.test_regime_labels[owner_array[sample]]
                ],
            )
            for sample in range(len(owner_array))
        ]
    )
    return {
        "test_context": test_context,
        "lag_labels": lag_labels_array,
        "inst_labels": inst_labels_array,
        "H": h_array,
        "sample_owner": owner_array,
        "lag_graphs": lag_graphs,
        "inst_graphs": inst_graphs,
        "dag_validity": dag_validity,
        "lag_exact": lag_exact,
        "inst_exact": inst_exact,
        "joint_exact": joint_exact,
    }


def posterior_metric_table(posterior: dict, cfg: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "test recording length",
                "DeFoG eta",
                "instantaneous DAG validity",
                "exact lag graph",
                "exact instantaneous graph",
                "exact joint graph pair",
            ],
            "value": [
                cfg.TEST_L,
                cfg.FM_ETA,
                posterior["dag_validity"],
                posterior["lag_exact"],
                posterior["inst_exact"],
                posterior["joint_exact"],
            ],
        }
    )


def _graph_grid(
    samples: np.ndarray,
    truth_one: np.ndarray,
    truth_two: np.ndarray,
    max_distance: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(samples)) / max(len(samples), 1)
    grid = np.zeros((max_distance + 1, max_distance + 1), np.float64)
    for graph, weight in zip(samples, weights):
        distance_one = graph_distance(graph, truth_one)
        distance_two = graph_distance(graph, truth_two)
        if distance_one <= max_distance and distance_two <= max_distance:
            grid[distance_two, distance_one] += float(weight)
    return grid


def plot_graph_distribution(
    result: dict,
    posterior: dict,
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
) -> None:
    max_distance = len(graph_space.lag_edge_list)
    true_lag_samples = np.stack(
        [data.true_model.lag_graphs[regime] for regime in data.test_regime_labels]
    )
    true_inst_samples = np.stack(
        [data.true_model.inst_graphs[regime] for regime in data.test_regime_labels]
    )
    particle_rows = result["trace"][
        result["trace"]["outer"] > cfg.PARTICLE_PLOT_BURNIN
    ]
    if len(particle_rows) == 0:
        particle_rows = result["trace"].iloc[-1:]
    particle_lag: list[np.ndarray] = []
    particle_inst: list[np.ndarray] = []
    particle_weight: list[float] = []
    for _, row in particle_rows.iterrows():
        for pair_index, weight in zip(row["pair_ids"], row["pi_values"]):
            lag_graph, inst_graph = graph_space.graph_pairs[int(pair_index)]
            particle_lag.append(lag_graph)
            particle_inst.append(inst_graph)
            particle_weight.append(float(weight))
    particle_lag_array = np.stack(particle_lag)
    particle_inst_array = np.stack(particle_inst)
    particle_weight_array = np.asarray(particle_weight, dtype=np.float64)
    particle_weight_array /= particle_weight_array.sum()
    lag_grids = [
        _graph_grid(
            true_lag_samples,
            data.true_model.lag_graphs[0],
            data.true_model.lag_graphs[1],
            max_distance,
        ),
        _graph_grid(
            particle_lag_array,
            data.true_model.lag_graphs[0],
            data.true_model.lag_graphs[1],
            max_distance,
            particle_weight_array,
        ),
        _graph_grid(
            posterior["lag_graphs"],
            data.true_model.lag_graphs[0],
            data.true_model.lag_graphs[1],
            max_distance,
        ),
    ]
    inst_grids = [
        _graph_grid(
            true_inst_samples,
            data.true_model.inst_graphs[0],
            data.true_model.inst_graphs[1],
            max_distance,
        ),
        _graph_grid(
            particle_inst_array,
            data.true_model.inst_graphs[0],
            data.true_model.inst_graphs[1],
            max_distance,
            particle_weight_array,
        ),
        _graph_grid(
            posterior["inst_graphs"],
            data.true_model.inst_graphs[0],
            data.true_model.inst_graphs[1],
            max_distance,
        ),
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(13.5, 7.2), constrained_layout=True
    )
    titles = [
        "True test distribution",
        f"Block-1 particle stream (outer > {cfg.PARTICLE_PLOT_BURNIN})",
        f"Conditional FM on test Y (L={cfg.TEST_L})",
    ]
    for row, (grids, row_name) in enumerate(
        [(lag_grids, r"$G^{lag}$"), (inst_grids, r"$G^{inst}$")]
    ):
        maximum = max(grid.max() for grid in grids)
        for column, (grid, title) in enumerate(zip(grids, titles)):
            image = axes[row, column].imshow(
                grid, origin="lower", cmap="magma", vmin=0, vmax=maximum
            )
            axes[row, column].set_title(f"{row_name}: {title}")
            axes[row, column].set_xlabel("distance to true regime 1")
            axes[row, column].set_ylabel("distance to true regime 2")
            axes[row, column].set_xticks(range(max_distance + 1))
            axes[row, column].set_yticks(range(max_distance + 1))
        figure.colorbar(
            image,
            ax=axes[row, :].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
    plt.show()


def representative_test_indices(data: SimulationData) -> list[int]:
    return [
        int(np.where(data.test_regime_labels == regime)[0][0])
        for regime in range(len(data.true_model.lag_graphs))
    ]


def plot_representative_graph_samples(
    result: dict,
    posterior: dict,
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    metadata: FMMetadata,
    n_samples: int = 3,
) -> list[int]:
    chosen = representative_test_indices(data)
    graph_samples: dict[int, dict[str, np.ndarray]] = {}
    for regime, test_index in enumerate(chosen):
        lag, inst, _ = sample_fm_given_context(
            result["fm"],
            posterior["test_context"][test_index],
            cfg,
            metadata,
            graph_space,
            data.device,
            n_samples=n_samples,
        )
        graph_samples[regime] = {
            "lag": np.stack(
                [lag_labels_to_graph(labels, graph_space, cfg) for labels in lag]
            ),
            "inst": np.stack(
                [inst_labels_to_graph(labels, graph_space, cfg) for labels in inst]
            ),
        }
    figure, axes = plt.subplots(
        4,
        1 + n_samples,
        figsize=(3.0 * (1 + n_samples), 10.5),
        constrained_layout=True,
    )
    row_specs = [
        (0, "lag", data.true_model.lag_graphs[0], r"Regime 1: $G^{lag}$"),
        (0, "inst", data.true_model.inst_graphs[0], r"Regime 1: $G^{inst}$"),
        (1, "lag", data.true_model.lag_graphs[1], r"Regime 2: $G^{lag}$"),
        (1, "inst", data.true_model.inst_graphs[1], r"Regime 2: $G^{inst}$"),
    ]
    for row, (regime, graph_type, truth, row_label) in enumerate(row_specs):
        axes[row, 0].imshow(truth, vmin=0, vmax=1, cmap="Greys")
        axes[row, 0].set_title("Ground truth")
        for sample in range(n_samples):
            axes[row, sample + 1].imshow(
                graph_samples[regime][graph_type][sample],
                vmin=0,
                vmax=1,
                cmap="Greys",
            )
            axes[row, sample + 1].set_title(f"FM sample {sample + 1}")
        for column in range(1 + n_samples):
            axis = axes[row, column]
            axis.set_xticks(range(cfg.A))
            axis.set_yticks(range(cfg.A))
            axis.set_xlabel("child")
            axis.set_ylabel("parent")
        axes[row, 0].set_ylabel(row_label + "\nparent", fontsize=11)
    plt.suptitle(
        "Conditional FM graph samples given two individual test trajectories",
        fontsize=14,
    )
    plt.show()
    return chosen


def true_H_scaled(
    cfg: Any,
    data: SimulationData,
    metadata: FMMetadata,
) -> np.ndarray:
    """Ground-truth soft-intervention H in the FM-scaled coordinates."""
    vectors: list[np.ndarray] = []
    true = data.true_model
    for regime in range(cfg.K_DATA):
        values: list[float] = []
        for child in range(cfg.A):
            alpha0 = np.zeros(metadata.coefficient_dim, np.float32)
            alpha1 = np.zeros(metadata.coefficient_dim, np.float32)
            alpha0[0] = true.bias0[regime][child]
            alpha0[1] = true.self_coef0[regime][child]
            alpha1[0] = true.bias1[regime][child]
            alpha1[1] = true.self_coef1[regime][child]
            for parent in range(cfg.A):
                alpha0[2 + parent] = (
                    true.lag_coef0[regime][parent, child]
                    * true.lag_graphs[regime][parent, child]
                )
                alpha1[2 + parent] = (
                    true.lag_coef1[regime][parent, child]
                    * true.lag_graphs[regime][parent, child]
                )
                alpha0[2 + cfg.A + parent] = (
                    true.inst_coef0[regime][parent, child]
                    * true.inst_graphs[regime][parent, child]
                )
                alpha1[2 + cfg.A + parent] = (
                    true.inst_coef1[regime][parent, child]
                    * true.inst_graphs[regime][parent, child]
                )
            log_obs_var = np.log(true.noise_sd0[regime][child] ** 2)
            delta_log_var = (
                np.log(true.noise_sd1[regime][child] ** 2)
                - log_obs_var
            )
            values.extend(alpha0.tolist())
            values.extend((alpha1 - alpha0).tolist())
            values.extend([log_obs_var, delta_log_var])
        vector = np.asarray(values, np.float32)
        if vector.shape != (metadata.h_dim,):
            raise ValueError(
                f"Unexpected true H shape {vector.shape}; expected {(metadata.h_dim,)}"
            )
        vectors.append(vector / metadata.h_scale_vector)
    return np.stack(vectors)


def h_rmse_to_truth(
    posterior: dict,
    cfg: Any,
    data: SimulationData,
    metadata: FMMetadata,
    active_only: bool = False,
) -> float:
    """RMSE of joint conditional-FM H samples against the true regime H."""
    truth = true_H_scaled(cfg, data, metadata)
    errors: list[float] = []
    for sample in range(len(posterior["H"])):
        owner = int(posterior["sample_owner"][sample])
        regime = int(data.test_regime_labels[owner])
        if active_only:
            mask = H_target_mask(
                data.true_model.lag_graphs[regime],
                data.true_model.inst_graphs[regime],
                cfg,
                metadata,
            ).astype(bool)
        else:
            mask = np.ones(metadata.h_dim, dtype=bool)
        errors.append(
            float(np.mean((posterior["H"][sample, mask] - truth[regime, mask]) ** 2))
        )
    return float(np.sqrt(np.mean(errors)))


def H_parameter_labels(cfg: Any) -> np.ndarray:
    labels: list[str] = []
    for child in range(cfg.A):
        j = child + 1
        labels.extend([fr"$b^0_{{{j}}}$", fr"$self^0_{{{j}}}$"])
        for parent in range(cfg.A):
            labels.append(fr"$lag^0\ {parent + 1}\to{j}$")
        for parent in range(cfg.A):
            labels.append(fr"$inst^0\ {parent + 1}\to{j}$")
        labels.extend([fr"$\Delta b_{{{j}}}$", fr"$\Delta self_{{{j}}}$"])
        for parent in range(cfg.A):
            labels.append(fr"$\Delta lag\ {parent + 1}\to{j}$")
        for parent in range(cfg.A):
            labels.append(fr"$\Delta inst\ {parent + 1}\to{j}$")
        labels.extend(
            [
                fr"$\log (\sigma^0_{{{j}}})^2$",
                fr"$\Delta\log \sigma^2_{{{j}}}$",
            ]
        )
    return np.asarray(labels)


def plot_oracle_H_regression_diagnostics(
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    metadata: FMMetadata,
) -> pd.DataFrame:
    """
    Check Block-1 H fitting with true Z, true regime weights, and true G.

    This isolates the weighted ridge mechanism estimator from representation,
    clustering, graph search, and conditional FM errors.
    """
    labels = H_parameter_labels(cfg)
    truth = true_H_scaled(cfg, data, metadata)
    true_pairs = graph_space.true_pair_indices(data)
    figure, axes = plt.subplots(
        cfg.K_DATA,
        1,
        figsize=(15, 4 * cfg.K_DATA),
        constrained_layout=True,
    )
    if cfg.K_DATA == 1:
        axes = [axes]
    summaries: list[dict] = []
    for regime, pair_index in enumerate(true_pairs):
        regime_weight = (data.regime_labels == regime).astype(float)
        cache = fit_component_cache(
            data.Z_true_np.astype(np.float64),
            regime_weight,
            cfg,
            data,
        )
        fits = H_for_pair(
            cache,
            int(pair_index),
            graph_space,
            cfg,
        )
        mechanisms = [
            {
                "alpha0": fit["alpha0"],
                "delta_alpha": fit["alpha1"] - fit["alpha0"],
                "obs_var": fit["obs_var"],
                "int_var": fit["int_var"],
            }
            for fit in fits
        ]
        estimate = pack_H_scaled(mechanisms, metadata)
        active_mask = H_target_mask(
            data.true_model.lag_graphs[regime],
            data.true_model.inst_graphs[regime],
            cfg,
            metadata,
        ).astype(bool)
        active_index = np.where(active_mask)[0]
        rmse = float(
            np.sqrt(
                np.mean(
                    (estimate[active_index] - truth[regime, active_index]) ** 2
                )
            )
        )
        axis = axes[regime]
        x = np.arange(len(active_index))
        axis.plot(x, estimate[active_index], "o-", label="Oracle ridge fit")
        axis.plot(
            x,
            truth[regime, active_index],
            "x--",
            linewidth=2,
            markersize=7,
            label="Ground-truth H",
        )
        axis.axhline(0, linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(
            labels[active_index], rotation=70, ha="right", fontsize=8
        )
        axis.set_ylabel("standardized H parameter")
        axis.set_title(
            f"Regime {regime + 1}: oracle Block-1 H fit "
            f"(true Z, true regime, true G; RMSE={rmse:.3f})"
        )
        if regime == 0:
            axis.legend(ncol=2)
        summaries.append(
            {
                "true regime": regime + 1,
                "true graph pair": int(pair_index),
                "active H dimensions": int(active_mask.sum()),
                "oracle ridge H RMSE": rmse,
            }
        )
    plt.show()
    return pd.DataFrame(summaries)


def plot_h_diagnostics(
    result: dict,
    posterior: dict,
    chosen_test_indices: list[int],
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    metadata: FMMetadata,
    n_samples: int = 30,
    interval: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Original-style diagnostic of H | Y, I, with the true graph fixed."""
    labels = H_parameter_labels(cfg)
    truth = true_H_scaled(cfg, data, metadata)
    figure, axes = plt.subplots(
        cfg.K_DATA, 1, figsize=(15, 4 * cfg.K_DATA), constrained_layout=True
    )
    if cfg.K_DATA == 1:
        axes = [axes]
    summaries: list[dict] = []
    for regime, test_index in enumerate(chosen_test_indices):
        samples = sample_H_given_fixed_graph(
            result["fm"],
            posterior["test_context"][test_index],
            data.true_model.lag_graphs[regime],
            data.true_model.inst_graphs[regime],
            cfg,
            metadata,
            graph_space,
            data.device,
            n_samples=n_samples,
        )
        active_mask = H_target_mask(
            data.true_model.lag_graphs[regime],
            data.true_model.inst_graphs[regime],
            cfg,
            metadata,
        ).astype(bool)
        active_index = np.where(active_mask)[0]
        mean = samples.mean(axis=0)
        lower = np.quantile(samples, interval[0], axis=0)
        upper = np.quantile(samples, interval[1], axis=0)
        rmse = float(
            np.sqrt(
                np.mean((mean[active_index] - truth[regime, active_index]) ** 2)
            )
        )
        coverage = float(
            np.mean(
                (truth[regime, active_index] >= lower[active_index])
                & (truth[regime, active_index] <= upper[active_index])
            )
        )
        axis = axes[regime]
        x = np.arange(len(active_index))
        axis.fill_between(
            x,
            lower[active_index],
            upper[active_index],
            alpha=0.25,
            label="90% FM interval",
        )
        axis.plot(x, mean[active_index], "o-", label="FM posterior mean")
        axis.plot(
            x,
            truth[regime, active_index],
            "x--",
            linewidth=2,
            markersize=7,
            label="Ground-truth H",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            labels[active_index], rotation=70, ha="right", fontsize=8
        )
        axis.axhline(0, linewidth=0.8)
        axis.set_title(
            f"True regime {regime + 1}: H | Y, I, true G "
            f"(RMSE={rmse:.3f}, 90% coverage={coverage:.3f})"
        )
        axis.set_ylabel("standardized H parameter")
        if regime == 0:
            axis.legend(ncol=3)
        summaries.append(
            {
                "true regime": regime + 1,
                "test trajectory index": int(test_index),
                "I": data.I_test_np[test_index].astype(int).tolist(),
                "active H dimensions": int(active_mask.sum()),
                "posterior mean RMSE": rmse,
                "90% interval coverage": coverage,
            }
        )
    plt.show()
    return pd.DataFrame(summaries)




def oracle_representation_correlation_table(
    diagnostic: dict,
    cfg: Any,
    data: SimulationData,
    initial: bool = False,
) -> pd.DataFrame:
    state = diagnostic["initial_state"] if initial else diagnostic["state"]
    learned = state["Z"][data.train_idx].reshape(-1, cfg.A)
    truth = data.Z_true_np[data.train_idx].reshape(-1, cfg.A)
    correlation = np.corrcoef(learned.T, truth.T)[: cfg.A, cfg.A :]
    return pd.DataFrame(
        correlation,
        index=[f"learned z{index + 1}" for index in range(cfg.A)],
        columns=[f"true z{index + 1}" for index in range(cfg.A)],
    )


def plot_oracle_representation_trace(diagnostic: dict) -> None:
    trace = diagnostic["trace"]
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.5), constrained_layout=True
    )
    axes[0].plot(trace["outer"], trace["best_abs_corr"])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Best permutation mean |corr|")
    axes[1].plot(trace["outer"], trace["latent_rms_change"])
    axes[1].set_title("RMS change from PCA Z")
    axes[2].plot(trace["outer"], trace["rec"], label="reconstruction")
    axes[2].plot(trace["outer"], trace["struct"], label="structural")
    axes[2].set_title("Oracle-Z training losses")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("outer iteration")
    plt.show()
