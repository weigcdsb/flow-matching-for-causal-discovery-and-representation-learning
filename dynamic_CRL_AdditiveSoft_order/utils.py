from __future__ import annotations
import math
import random
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(cfg: Any) -> torch.device:
    requested = torch.device(cfg.DEVICE)
    device = requested
    if requested.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    np.set_printoptions(precision=3, suppress=True)
    return device


def adj_from_edges(edges: Iterable[tuple[int, int]], d: int) -> np.ndarray:
    graph = np.zeros((d, d), dtype=np.float32)
    for parent, child in edges:
        graph[parent, child] = 1.0
    return graph


def is_dag(graph: np.ndarray) -> bool:
    graph = (graph > 0.5).astype(int)
    indegree = graph.sum(0).astype(int).tolist()
    queue = [node for node in range(len(graph)) if indegree[node] == 0]
    seen = 0
    while queue:
        parent = queue.pop(0)
        seen += 1
        for child in range(len(graph)):
            if graph[parent, child]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
    return seen == len(graph)


def topo_order(graph: np.ndarray) -> list[int]:
    graph = (graph > 0.5).astype(int)
    indegree = graph.sum(0).astype(int).tolist()
    queue = [node for node in range(len(graph)) if indegree[node] == 0]
    order: list[int] = []
    while queue:
        parent = queue.pop(0)
        order.append(parent)
        for child in range(len(graph)):
            if graph[parent, child]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
    if len(order) != len(graph):
        raise ValueError("Graph is not a DAG")
    return order


def normalized_time_encoding(valid_mask: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal encoding on normalized time [0, 1], valid for any length."""
    batch_size, max_length = valid_mask.shape
    lengths = valid_mask.sum(1).clamp_min(1)
    positions = torch.arange(max_length, device=valid_mask.device)[None, :].expand(
        batch_size, -1
    )
    denominator = (lengths - 1).clamp_min(1)[:, None]
    tau = positions / denominator
    half = width // 2
    frequency = torch.exp(
        torch.linspace(0.0, math.log(1000.0), half, device=valid_mask.device)
    )
    angle = 2 * math.pi * tau[:, :, None] * frequency[None, None, :]
    encoding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    if encoding.shape[-1] < width:
        encoding = torch.nn.functional.pad(
            encoding, (0, width - encoding.shape[-1])
        )
    return encoding * valid_mask[:, :, None].float()


def pad_y_list(
    recordings: list[np.ndarray],
    observation_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(recording) for recording in recordings)
    padded = np.zeros(
        (len(recordings), max_length, observation_dim), np.float32
    )
    valid = np.zeros((len(recordings), max_length), bool)
    for index, recording in enumerate(recordings):
        padded[index, : len(recording)] = recording
        valid[index, : len(recording)] = True
    return (
        torch.tensor(padded, dtype=torch.float32, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )


def graph_distance(graph_a: np.ndarray, graph_b: np.ndarray) -> int:
    return int(np.abs(graph_a - graph_b).sum())
