from __future__ import annotations
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from fm import FMMetadata, fm_update_from_batch, make_fm_simulation_batch
from models import ContextReconstructionDecoder, RecordingContext, TemporalCoupledFM
from particles import (
    GraphSpace,
    H_for_pair,
    draw_particle_noise,
    encode_all_np,
    fit_component_cache,
    graph_dist_to_truth,
    init_representation_models,
    initialize_mixture,
    mechanism_from_fit,
    perturb_H,
    infer_soft_intervention_anchor,
    move_soft_graph_particle,
    standardize_latents,
    structural_transition_nll_torch,
    torch_standardize,
    update_responsibilities,
)
from sim import SimulationData
from utils import pad_y_list


def sample_context_batch(
    rng: np.random.Generator,
    cfg: Any,
    data: SimulationData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = rng.choice(
        data.train_idx,
        size=min(cfg.CONTEXT_BATCH, len(data.train_idx)),
        replace=False,
    )
    recordings: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    for index in ids:
        length = int(
            rng.integers(cfg.CONTEXT_MIN_LENGTH, cfg.TRAIN_L + 1)
        )
        start = int(rng.integers(0, cfg.TRAIN_L - length + 1))
        recordings.append(data.Y_np[index, start : start + length])
        interventions.append(data.I_np[index])
    y_padded, valid_mask = pad_y_list(recordings, cfg.P, data.device)
    intervention_tensor = torch.tensor(
        np.stack(interventions), dtype=torch.float32, device=data.device
    )
    return y_padded, valid_mask, intervention_tensor


def context_reconstruction_loss(
    context_net: RecordingContext,
    decoder: ContextReconstructionDecoder,
    y_padded: torch.Tensor,
    valid_mask: torch.Tensor,
    intervention: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    context = context_net(y_padded, valid_mask, intervention)
    reconstruction = decoder(context, valid_mask)
    mask = valid_mask[:, :, None].float()
    return (((reconstruction - y_padded) ** 2) * mask).sum() / (
        mask.sum() * cfg.P
    ).clamp_min(1.0)


def run_stage0(
    cfg: Any,
    data: SimulationData,
    verbose: bool = True,
) -> dict:
    """Train C(Y, I) only through observation reconstruction, then freeze it."""
    torch.manual_seed(cfg.SEED + 1000)
    context_net = RecordingContext(cfg).to(data.device)
    context_decoder = ContextReconstructionDecoder(cfg).to(data.device)
    optimizer = torch.optim.Adam(
        list(context_net.parameters()) + list(context_decoder.parameters()),
        lr=cfg.CONTEXT_LR,
    )
    rng = np.random.default_rng(cfg.SEED + 1001)
    trace: list[float] = []
    context_net.train()
    context_decoder.train()
    for step in range(1, cfg.CONTEXT_STEPS + 1):
        y_padded, valid_mask, intervention = sample_context_batch(
            rng, cfg, data
        )
        loss = context_reconstruction_loss(
            context_net,
            context_decoder,
            y_padded,
            valid_mask,
            intervention,
            cfg,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(context_net.parameters())
            + list(context_decoder.parameters()),
            10.0,
        )
        optimizer.step()
        trace.append(float(loss.detach()))
        if verbose and (
            step == 1 or step % 100 == 0 or step == cfg.CONTEXT_STEPS
        ):
            print(
                f"context step={step:04d} | reconstruction={trace[-1]:.5f}"
            )

    context_net.eval()
    for parameter in context_net.parameters():
        parameter.requires_grad_(False)
    context_decoder.eval()
    with torch.no_grad():
        y_padded, valid_mask = pad_y_list(
            [data.Y_np[index] for index in data.val_idx],
            cfg.P,
            data.device,
        )
        intervention = torch.tensor(
            data.I_np[data.val_idx],
            dtype=torch.float32,
            device=data.device,
        )
        context = context_net(y_padded, valid_mask, intervention)
        reconstruction = context_decoder(context, valid_mask)
        mask = valid_mask[:, :, None].float()
        validation_mse = float(
            (
                (((reconstruction - y_padded) ** 2) * mask).sum()
                / (mask.sum() * cfg.P).clamp_min(1.0)
            ).cpu()
        )
    return {
        "context_net": context_net,
        "context_decoder": context_decoder,
        "trace": np.asarray(trace),
        "validation_mse": validation_mse,
    }


def run_stage1(
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    metadata: FMMetadata,
    context_net: RecordingContext,
    verbose: bool = True,
) -> dict:
    """Run Stage 1: Block 1 particles/representation and Block 2 FM."""
    k_fit = cfg.K_FIT
    encoder, decoder, rep_optimizer, rep_scheduler = init_representation_models(
        cfg, data
    )
    particle_rng = np.random.default_rng(cfg.SEED + 900 + k_fit)
    state = standardize_latents(
        encode_all_np(encoder, cfg, data), cfg, data
    )
    gamma, pi, pairs, mechanisms, caches = initialize_mixture(
        state["Z"],
        k_fit,
        graph_space,
        cfg,
        data,
    )
    dead = np.zeros(k_fit, int)
    context_net.eval()
    if any(parameter.requires_grad for parameter in context_net.parameters()):
        raise ValueError("Stage-0 context encoder must be frozen before Stage 1")

    torch.manual_seed(cfg.SEED + 1234)
    fm = TemporalCoupledFM(
        cfg, metadata.e_lag, metadata.e_inst, metadata.h_dim
    ).to(data.device)
    fm_optimizer = torch.optim.Adam(fm.parameters(), lr=cfg.FM_LR)
    trace: list[dict] = []
    last_fm_batch = None
    representation_rng = np.random.default_rng(cfg.SEED + 1700)
    soft_graph_info = None

    for outer in range(1, cfg.MAX_OUTER + 1):
        rep_logs = []
        for _ in range(cfg.REP_UPDATES_PER_OUTER):
            batch_size = min(cfg.REP_BATCH_TRAJ, len(data.train_idx))
            ids = representation_rng.choice(
                data.train_idx, size=batch_size, replace=False
            )
            observations = data.Y[ids]
            rep_optimizer.zero_grad()
            latent_raw_flat = encoder(observations.reshape(-1, cfg.P))
            latent_raw = latent_raw_flat.reshape(
                batch_size, cfg.TRAIN_L, cfg.D
            )
            reconstruction = decoder(latent_raw_flat).reshape(
                batch_size, cfg.TRAIN_L, cfg.P
            )
            reconstruction_loss = 0.5 * (
                ((reconstruction - observations) / cfg.RECON_SIGMA) ** 2
            ).mean()
            latent = torch_standardize(latent_raw, state)
            structural_loss = structural_transition_nll_torch(
                latent,
                data.I_torch[ids],
                gamma[ids],
                pairs,
                mechanisms,
                graph_space,
                cfg,
            )
            representation_loss = (
                reconstruction_loss + cfg.BETA_STRUCT * structural_loss
            )
            representation_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 10.0
            )
            rep_optimizer.step()
            rep_logs.append(
                [
                    float(representation_loss.detach()),
                    float(reconstruction_loss.detach()),
                    float(structural_loss.detach()),
                ]
            )
        rep_scheduler.step()
        rep_mean = np.mean(rep_logs, axis=0)

        new_state = standardize_latents(
            encode_all_np(encoder, cfg, data), cfg, data
        )

        fm_noise = draw_particle_noise(k_fit, particle_rng, cfg)
        new_mechanisms = []
        fm_mechanisms = []
        changed = 0
        accepted_moves = 0
        for component in range(k_fit):
            current = int(pairs[component])
            cache = fit_component_cache(
                new_state["Z"],
                gamma[:, component],
                cfg,
                data,
                graph_pair=graph_space.graph_pairs[current],
            )
            fits = H_for_pair(cache, current, graph_space, cfg)

            # Deterministic weighted-ridge H is used by Block 1.
            new_mechanisms.append(
                [mechanism_from_fit(fit, cfg) for fit in fits]
            )

            # Independent H perturbations are used only as FM augmentation.
            fm_mechanisms.append(
                [
                    perturb_H(
                        fit, child, component, fm_noise, cfg
                    )
                    for child, fit in enumerate(fits)
                ]
            )
        mechanisms = new_mechanisms

        gamma_new, log_likelihoods = update_responsibilities(
            new_state["Z"],
            pairs,
            mechanisms,
            pi,
            graph_space,
            cfg,
            data,
        )
        pi_new = (gamma_new.sum(0) + cfg.PI_ALPHA) / (
            cfg.N + k_fit * cfg.PI_ALPHA
        )
        effective_n = gamma_new.sum(0)
        dead = np.where(effective_n < cfg.MIN_EFFECTIVE_N, dead + 1, 0)
        for component in np.where(dead >= cfg.DEAD_PATIENCE)[0]:
            donor = int(np.argmax(effective_n))
            pairs[component] = pairs[donor]
            mechanisms[component] = deepcopy(mechanisms[donor])
            fm_mechanisms[component] = deepcopy(fm_mechanisms[donor])
            gamma_new[:, component] = 0.5 * gamma_new[:, donor]
            gamma_new[:, donor] *= 0.5
            gamma_new /= gamma_new.sum(1, keepdims=True)
            pi_new = (gamma_new.sum(0) + cfg.PI_ALPHA) / (
                cfg.N + k_fit * cfg.PI_ALPHA
            )
            dead[component] = 0

        burnin = int(getattr(cfg, "SOFT_GRAPH_BURNIN", 1))
        if outer >= burnin:
            anchor = infer_soft_intervention_anchor(
                new_state["Z"], gamma_new, cfg, data
            )
            corrected = anchor.pop("corrected")
            selector_scores = []
            previous_pairs = pairs.copy()
            for component, order in enumerate(anchor["orders"]):
                moved, accepted, selector_score = move_soft_graph_particle(
                    int(pairs[component]),
                    order,
                    corrected,
                    gamma_new[:, component],
                    graph_space,
                    particle_rng,
                    cfg,
                    data,
                )
                pairs[component] = moved
                accepted_moves += accepted
                selector_scores.append(selector_score)
            changed += int(np.sum(previous_pairs != pairs))
            soft_graph_info = {
                **anchor,
                "selector_scores": tuple(map(float, selector_scores)),
            }

            # Refit H under the moved graph particles, then refresh regimes.
            mechanisms = []
            fm_mechanisms = []
            for component in range(k_fit):
                current = int(pairs[component])
                cache = fit_component_cache(
                    new_state["Z"],
                    gamma_new[:, component],
                    cfg,
                    data,
                    graph_pair=graph_space.graph_pairs[current],
                )
                fits = H_for_pair(cache, current, graph_space, cfg)
                mechanisms.append(
                    [mechanism_from_fit(fit, cfg) for fit in fits]
                )
                fm_mechanisms.append(
                    [
                        perturb_H(
                            fit, child, component, fm_noise, cfg
                        )
                        for child, fit in enumerate(fits)
                    ]
                )
            gamma_new, log_likelihoods = update_responsibilities(
                new_state["Z"],
                pairs,
                mechanisms,
                pi_new,
                graph_space,
                cfg,
                data,
            )
            pi_new = (gamma_new.sum(0) + cfg.PI_ALPHA) / (
                cfg.N + k_fit * cfg.PI_ALPHA
            )
            if verbose and (outer == burnin or outer % 5 == 0):
                print(
                    "soft intervention anchor | "
                    f"orders={soft_graph_info['orders']} | "
                    f"pairs={pairs.tolist()} | "
                    f"accepted={accepted_moves} | "
                    f"score={soft_graph_info['order_score']:.6g}"
                )

        state, gamma, pi = new_state, gamma_new, pi_new

        fm_batch = make_fm_simulation_batch(
            decoder,
            state,
            pairs,
            fm_mechanisms,
            pi,
            outer,
            cfg,
            data,
            graph_space,
            metadata,
        )
        fm_losses = fm_update_from_batch(
            context_net,
            fm,
            fm_optimizer,
            fm_batch,
            cfg,
            metadata,
            data.device,
        )
        last_fm_batch = fm_batch

        ari = adjusted_rand_score(data.regime_labels, gamma.argmax(1))
        trace.append(
            {
                "outer": outer,
                "rep_loss": rep_mean[0],
                "rec": rep_mean[1],
                "struct": rep_mean[2],
                "ARI": ari,
                "graph_changed": changed,
                "graph_moves_accepted": accepted_moves,
                "registered_graph_states": len(graph_space.graph_pairs),
                "unique_graphs": len(set(map(int, pairs))),
                "pair_ids": tuple(map(int, pairs)),
                "pi_values": tuple(map(float, pi)),
                "graph_distances": tuple(
                    tuple(graph_dist_to_truth(pair, graph_space, data))
                    for pair in pairs
                ),
                "fm_total": fm_losses[0],
                "fm_lag": fm_losses[1],
                "fm_inst": fm_losses[2],
                "fm_H": fm_losses[3],
            }
        )
        if verbose and (
            outer == 1 or outer % 5 == 0 or outer == cfg.MAX_OUTER
        ):
            distances = [
                graph_dist_to_truth(pair, graph_space, data)
                for pair in pairs
            ]
            print(
                f"outer={outer:02d} | ARI={ari:.3f} | "
                f"pi={np.round(pi, 3)} | d={distances} | "
                f"G changes={changed} | accepted={accepted_moves} | FM={fm_losses[0]:.3f}"
            )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "state": state,
        "gamma": gamma,
        "pi": pi,
        "pairs": pairs.copy(),
        "Hs": mechanisms,
        "context_net": context_net,
        "fm": fm,
        "trace": pd.DataFrame(trace),
        "last_fm_batch": last_fm_batch,
        "fm_metadata": metadata,
        "soft_graph_info": soft_graph_info,
    }



def _true_regime_responsibilities(cfg: Any, data: SimulationData) -> np.ndarray:
    gamma = np.zeros((cfg.N, cfg.K_DATA), dtype=np.float64)
    gamma[np.arange(cfg.N), data.regime_labels.astype(int)] = 1.0
    return gamma


def _fit_oracle_graph_mechanisms(
    state: dict[str, np.ndarray],
    gamma_true: np.ndarray,
    true_pairs: np.ndarray,
    graph_space: GraphSpace,
    cfg: Any,
    data: SimulationData,
) -> list[list[dict]]:
    mechanisms: list[list[dict]] = []
    for regime, pair_index in enumerate(true_pairs):
        cache = fit_component_cache(
            state["Z"], gamma_true[:, regime], cfg, data
        )
        fits = H_for_pair(cache, int(pair_index), graph_space, cfg)
        mechanisms.append(
            [mechanism_from_fit(fit, cfg) for fit in fits]
        )
    return mechanisms


def _latent_correlation_metrics(
    state: dict[str, np.ndarray],
    initial_state: dict[str, np.ndarray],
    cfg: Any,
    data: SimulationData,
) -> tuple[np.ndarray, float, float]:
    import itertools

    learned = state["Z"][data.train_idx].reshape(-1, cfg.A)
    truth = data.Z_true_np[data.train_idx].reshape(-1, cfg.A)
    correlation = np.corrcoef(learned.T, truth.T)[: cfg.A, cfg.A :]
    best = max(
        np.mean([abs(correlation[row, col]) for row, col in enumerate(perm)])
        for perm in itertools.permutations(range(cfg.A))
    )
    rms_change = float(
        np.sqrt(np.mean((state["Z"] - initial_state["Z"]) ** 2))
    )
    return correlation, float(best), rms_change


def _encoder_gradient_norms_oracle(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    state: dict[str, np.ndarray],
    gamma_true: np.ndarray,
    true_pairs: np.ndarray,
    mechanisms: list[list[dict]],
    graph_space: GraphSpace,
    cfg: Any,
    data: SimulationData,
) -> dict[str, float]:
    ids = data.train_idx[: min(cfg.REP_BATCH_TRAJ, len(data.train_idx))]
    observations = data.Y[ids]
    latent_raw_flat = encoder(observations.reshape(-1, cfg.P))
    latent_raw = latent_raw_flat.reshape(len(ids), cfg.TRAIN_L, cfg.D)
    reconstruction = decoder(latent_raw_flat).reshape(
        len(ids), cfg.TRAIN_L, cfg.P
    )
    reconstruction_loss = 0.5 * (
        ((reconstruction - observations) / cfg.RECON_SIGMA) ** 2
    ).mean()
    latent = torch_standardize(latent_raw, state)
    structural_loss = structural_transition_nll_torch(
        latent,
        data.I_torch[ids],
        gamma_true[ids],
        true_pairs,
        mechanisms,
        graph_space,
        cfg,
    )
    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    rec_grad = torch.autograd.grad(
        reconstruction_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    struct_grad = torch.autograd.grad(
        cfg.BETA_STRUCT * structural_loss,
        parameters,
        allow_unused=True,
    )

    def norm(grads: tuple[Optional[torch.Tensor], ...]) -> float:
        total = torch.zeros((), device=data.device)
        for grad in grads:
            if grad is not None:
                total = total + grad.detach().pow(2).sum()
        return float(torch.sqrt(total).cpu())

    return {
        "reconstruction": norm(rec_grad),
        "weighted_structural": norm(struct_grad),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "structural_loss": float(structural_loss.detach().cpu()),
    }


def run_oracle_representation_diagnostic(
    cfg: Any,
    data: SimulationData,
    graph_space: GraphSpace,
    max_outer: int = 40,
    verbose: bool = True,
) -> dict:
    """Learn Z while fixing the true graph and true regime responsibilities.

    This leaves the encoder/decoder objective unchanged. H is re-fit from the
    current Z after each outer iteration. No graph learning, clustering, or FM
    is involved, so the experiment isolates representation learning.
    """
    encoder, decoder, optimizer, scheduler = init_representation_models(cfg, data)
    state = standardize_latents(encode_all_np(encoder, cfg, data), cfg, data)
    initial_state = {key: value.copy() for key, value in state.items()}

    gamma_true = _true_regime_responsibilities(cfg, data)
    true_pairs = np.asarray(graph_space.true_pair_indices(data), dtype=int)
    mechanisms = _fit_oracle_graph_mechanisms(
        state, gamma_true, true_pairs, graph_space, cfg, data
    )
    gradient_norms = _encoder_gradient_norms_oracle(
        encoder,
        decoder,
        state,
        gamma_true,
        true_pairs,
        mechanisms,
        graph_space,
        cfg,
        data,
    )

    _, initial_best_corr, _ = _latent_correlation_metrics(
        state, initial_state, cfg, data
    )
    if verbose:
        ratio = gradient_norms["weighted_structural"] / max(
            gradient_norms["reconstruction"], 1e-12
        )
        print(
            "oracle-Z initial encoder gradients | "
            f"reconstruction={gradient_norms['reconstruction']:.4g} | "
            f"beta*structural={gradient_norms['weighted_structural']:.4g} | "
            f"ratio={ratio:.4g}"
        )
        print(f"oracle-Z initial best |corr|={initial_best_corr:.4f}")

    rng = np.random.default_rng(cfg.SEED + 2700)
    trace: list[dict] = []
    for outer in range(1, int(max_outer) + 1):
        logs = []
        for _ in range(cfg.REP_UPDATES_PER_OUTER):
            batch_size = min(cfg.REP_BATCH_TRAJ, len(data.train_idx))
            ids = rng.choice(data.train_idx, size=batch_size, replace=False)
            observations = data.Y[ids]

            optimizer.zero_grad()
            latent_raw_flat = encoder(observations.reshape(-1, cfg.P))
            latent_raw = latent_raw_flat.reshape(
                batch_size, cfg.TRAIN_L, cfg.D
            )
            reconstruction = decoder(latent_raw_flat).reshape(
                batch_size, cfg.TRAIN_L, cfg.P
            )
            reconstruction_loss = 0.5 * (
                ((reconstruction - observations) / cfg.RECON_SIGMA) ** 2
            ).mean()
            latent = torch_standardize(latent_raw, state)
            structural_loss = structural_transition_nll_torch(
                latent,
                data.I_torch[ids],
                gamma_true[ids],
                true_pairs,
                mechanisms,
                graph_space,
                cfg,
            )
            loss = reconstruction_loss + cfg.BETA_STRUCT * structural_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 10.0
            )
            optimizer.step()
            logs.append(
                [
                    float(loss.detach().cpu()),
                    float(reconstruction_loss.detach().cpu()),
                    float(structural_loss.detach().cpu()),
                ]
            )

        scheduler.step()
        state = standardize_latents(encode_all_np(encoder, cfg, data), cfg, data)
        mechanisms = _fit_oracle_graph_mechanisms(
            state, gamma_true, true_pairs, graph_space, cfg, data
        )
        correlation, best_corr, rms_change = _latent_correlation_metrics(
            state, initial_state, cfg, data
        )
        mean_log = np.mean(logs, axis=0)
        trace.append(
            {
                "outer": outer,
                "rep_loss": mean_log[0],
                "rec": mean_log[1],
                "struct": mean_log[2],
                "best_abs_corr": best_corr,
                "latent_rms_change": rms_change,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if verbose and (outer == 1 or outer % 5 == 0 or outer == max_outer):
            print(
                f"oracle-Z outer={outer:02d} | "
                f"best |corr|={best_corr:.4f} | "
                f"latent RMS change={rms_change:.4f} | "
                f"rec={mean_log[1]:.4f} | struct={mean_log[2]:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    final_correlation, final_best_corr, final_rms_change = _latent_correlation_metrics(
        state, initial_state, cfg, data
    )
    return {
        "encoder": encoder,
        "decoder": decoder,
        "state": state,
        "initial_state": initial_state,
        "gamma": gamma_true,
        "pairs": true_pairs,
        "Hs": mechanisms,
        "trace": pd.DataFrame(trace),
        "initial_gradient_norms": gradient_norms,
        "initial_best_abs_corr": initial_best_corr,
        "final_best_abs_corr": final_best_corr,
        "final_latent_rms_change": final_rms_change,
        "final_correlation": final_correlation,
    }
