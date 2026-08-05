# Dynamic CRL organized experiment

## Files

- `01_FM_dynamicCRL_hardInt.ipynb` — all experiment/training settings, orchestration, and displayed results.
- `sim.py` — true latent process, observation model, and train/validation/test simulation.
- `models.py` — PyTorch model definitions.
- `particles.py` — graph space, mechanism fitting, responsibilities, local-MH particle movement, and structural losses.
- `fm.py` — graph/H packing, Block-2 teacher simulation, DeFoG training, and posterior sampling.
- `train.py` — Stage 0 and Stage 1 training loops.
- `diagnostics.py` — tables, metrics, and plots.
- `utils.py` — reproducibility, graph utilities, padding, and time encoding.
