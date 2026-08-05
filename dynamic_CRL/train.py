from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from fm import FMMetadata, fm_update_from_batch, make_fm_simulation_batch
from models import ContextReconstructionDecoder, RecordingContext, TemporalCoupledFM
from particles import (
    GraphSpace,
    H_for_pair,
    blend_H,
    draw_particle_noise,
    encode_all_np,
    fit_component_cache,
    graph_dist_to_truth,
    init_representation_models,
    initialize_mixture,
    perturb_H,
    score_pairs,
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
            list(context_net.parameters()) + list(context_decoder.parameters()),
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
            data.I_np[data.val_idx], dtype=torch.float32, device=data.device
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
    initial_noise = draw_particle_noise(k_fit, particle_rng, cfg)
    state = standardize_latents(
        encode_all_np(encoder, cfg, data), cfg, data
    )
    gamma, pi, pairs, mechanisms, caches = initialize_mixture(
        state["Z"],
        k_fit,
        initial_noise,
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

    for outer in range(1, cfg.MAX_OUTER + 1):
        # Block 1A: representation update under current G,H
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

        # Block 1B: re-encode and restandardize
        new_state = standardize_latents(
            encode_all_np(encoder, cfg, data), cfg, data
        )

        # Block 1C: persistent local-MH graph/mechanism update
        particle_noise = draw_particle_noise(k_fit, particle_rng, cfg)
        new_mechanisms = []
        changed = 0
        accepted_moves = 0
        hard_labels = gamma.argmax(1)

        for component in range(k_fit):
            fit_weight = (hard_labels == component).astype(float)
            if fit_weight.sum() < 2:
                fit_weight = gamma[:, component]

            cache = fit_component_cache(
                new_state["Z"], fit_weight, cfg, data
            )
            scores = score_pairs(cache, graph_space, cfg)
            current = int(pairs[component])
            current, n_accept = graph_space.local_mh_update(
                current, scores, particle_rng, cfg
            )

            changed += int(current != pairs[component])
            accepted_moves += n_accept
            pairs[component] = current

            fits = H_for_pair(cache, current, graph_space, cfg)
            new_mechanisms.append(
                [
                    blend_H(
                        mechanisms[component][child],
                        perturb_H(
                            fit,
                            child,
                            component,
                            particle_noise,
                            cfg,
                        ),
                        cfg,
                    )
                    for child, fit in enumerate(fits)
                ]
            )
        mechanisms = new_mechanisms

        # Block 1D: responsibilities and mixture weights
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
            gamma_new[:, component] = 0.5 * gamma_new[:, donor]
            gamma_new[:, donor] *= 0.5
            gamma_new /= gamma_new.sum(1, keepdims=True)
            pi_new = (gamma_new.sum(0) + cfg.PI_ALPHA) / (
                cfg.N + k_fit * cfg.PI_ALPHA
            )
            dead[component] = 0

        state, gamma, pi = new_state, gamma_new, pi_new

        # Block 2: simulate from updated particles and train FM
        fm_batch = make_fm_simulation_batch(
            decoder,
            state,
            pairs,
            mechanisms,
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
                f"MH accepts={accepted_moves} | FM={fm_losses[0]:.3f}"
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
    }
