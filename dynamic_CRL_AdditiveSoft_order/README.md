# Dynamic CRL with soft-intervention anchored moving graph particles

Block 1 keeps the successful soft-intervention anchor, but graph search is now local and persistent rather than exhaustive or frozen.

At outer iteration `r`:

1. Update `Z` with reconstruction + structural loss under the current graph particles.
2. Refit deterministic `H` and update regime responsibilities `gamma`.
3. Infer the soft-intervention coordinate correction and instantaneous order from `(Z, I, gamma)`.
4. For each component, make local one-edge graph proposals consistent with that order and accept them using the anchored validation score.
5. Refit `H` under the moved graph particles, refresh `gamma`, and continue.

For a proposal `G -> G'`, only the affected child's degree-3 additive regression is evaluated. Parent-set scores are cached on demand. `GraphSpace.build()` registers only the empty graph; new graph states are stored only when visited. Thus Block-1 graph search does not enumerate all lag graphs, DAGs, or graph pairs.

The intervention anchor and graph score use learned `Z`, observed intervention labels, and fitted responsibilities only. True latents/graphs/regimes are used only by simulation diagnostics.

Default settings:

- `SOFT_GRAPH_BURNIN = 1`
- `GRAPH_MOVES_PER_OUTER = 3`
- `TAU_GRAPH = 5e-4`
- `SOFT_GRAPH_POLY_DEGREE = 3`
- `SOFT_GRAPH_EDGE_PENALTY = 0.002`

Backend integration test (seed 13, same simulation, 3 graph moves/outer) recovered both exact true graphs by outer 20 while the representation continued to update; only 55 graph states had been registered after 30 outers.

The exact order search is intentionally left unchanged for the current small-`A` experiment; this pack addresses graph-space scalability only.
