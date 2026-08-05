# Flow Matching for Causal Discovery and Representation Learning

Using **conditional flow matching** as an amortized posterior sampler over causal graphs and mechanism parameters.

The repository currently contains two related proof-of-concept projects:

1. **Observed-variable causal discovery:** $q_\theta(G,H\mid X)\approx p(G,H\mid X)$, where $G$ is a directed acyclic graph and $H$ contains continuous mechanism parameters.

2. **Dynamic causal representation learning:** $q_\theta\left(G^{\mathrm{lag}},G^{\mathrm{inst}},H\mid Y_{1:L},I\right)$, where high-dimensional trajectories $Y_{1:L}$ are generated from latent causal variables ($L =$ recording length), $I$ denotes a intervention condition (**static and hard now, to be update later**), $G^{\mathrm{lag}}$ represents lagged causal relations, and $G^{\mathrm{inst}}$ is an instantaneous DAG.


## Motivation

Causal structure can be weakly identifiable or multimodal. A single point estimate can be bad as it ignores...

- multiple plausible graphs or Markov-equivalent structures;
- multiple trajectory-level causal regimes;
- uncertainty in discrete graph structure and continuous mechanisms.

The goal here is to learn a conditional generative model that can **sample** plausible causal structures and mechanisms, rather than returning only one estimate.

Flow matching (FM) is attractive because it provides a common framework for:

- discrete graph variables;
- continuous mechanism parameters;
- posterior sampling and uncertainty quantification;
- amortized inference for new observations after training (good for generalization).


## Core idea

Simulation-based inference (SBI):

1. **Sample (and move) particles.**  
   Sample noisy particles $(G,H)$, or regime-specific particles $(G^{\mathrm{lag}},G^{\mathrm{inst}},H)$, using approximate likelihoods, mixture responsibilities, and local graph proposals. (very cheap and rough. Just to obtain sample pair for the following FM step)

2. **Simulate conditional training data.**  
   Generate synthetic observations from the current particles:

   - $(G,H)\longrightarrow X_{\mathrm{sim}}$ for the causal discovery task;
   - $(G^{\mathrm{lag}},G^{\mathrm{inst}},H,I)\longrightarrow Y_{1:L}^{\mathrm{sim}}$ for the causal representation learning task.

3. **Train a conditional flow.**  
   Learn an amortized map from an observation or recording to a posterior distribution over graphs and mechanisms.

The particles act as an adaptive, approximate teacher. The flow model then smooths and amortizes this particle-based approximation.


## Causal discovery

All causal variables are directly observed. A rough particle approximation first provides candidate graph-mechanism pairs $(G,H)$. These particles are then used to simulate training triples

$$
(x_{\mathrm{sim}},G,H),
$$

and a conditional flow is trained to generate

$$
(G,H)\sim q_\theta(G,H\mid x).
$$

The notebooks mainly serve as a proof of concept that discrete graph structure and continuous mechanism parameters can be generated jointly with flow matching. See more details in FM for dyanmical CRL.


## Dynamic causal representation learning

The high-dimensional trajectories $Y_{1:L}$ is generated from lower-dimensional latent causal variables $Z_{1:L}$. Each trajectory belongs to one fixed causal regime.

For a non-intervened latent variable $j$, the simulator and fitted particle models use nonlinear autoregressive mechanisms of the form

$$
z_{t,j}=b_j+a_jz_{t-1,j}+\sum_k G^{\mathrm{lag}}_{kj}\beta_{kj}\tanh(z_{t-1,k})+\sum_k G^{\mathrm{inst}}_{kj}\gamma_{kj}\tanh(z_{t,k})+\epsilon_{t,j}.
$$

A **hard intervention** (need to relax to soft) replaces the corresponding structural mechanism with an intervention distribution (Gaussian). The latent trajectory is mapped to a high-dimensional observation through a noisy nonlinear observation model.


### Training stages

#### Stage 0: variable-length recording encoder

To handle variable-length time series, a Transformer-based context network compresses a padded recording and its intervention condition into a fixed-dimensional context $C_\psi(Y_{1:L},I)=c$. The context network is trained through observation reconstruction and then frozen.


#### Stage 1, Block 1: representation learning and particle updates

An encoder-decoder learns the latent causal coordinates as $z_t=E_\phi(y_t)$ and $\hat y_t=D_\omega(z_t)$. Motivated by iCITRIS, each observed intervention is assumed to correspond to a 1-D latent coordinate. For particle $k$, the conditional distribution of coordinate $j$ is

$$
p_k(z_{n,t,j}\mid z_{n,<t},z_{n,t,<j},I_{n,j})
=
\begin{cases}
\mathcal{N}\left(\mu_{k,t,j},\sigma_{k,j}^2\right), & I_{n,j}=0, \cr
\mathcal{N}\left(m_{k,j}^{\mathrm{int}},(\sigma_{k,j}^{\mathrm{int}})^2\right), & I_{n,j}=1.
\end{cases}
$$

where $\mu_{k,t,j}$ is determined by $(G_k^{\mathrm{lag}},G_k^{\mathrm{inst}},H_k)$ (This is where intervention and causal structure kicks in).

Let $\ell_{nk}=\log p_k(z_{n,1:L}\mid I_n)$ be the resulting trajectory log-likelihood. The regime responsibility is $\gamma_{nk}=\mathrm{softmax}_k\left(\log\pi_k+\ell_{nk}/\tau_{\mathrm{resp}}\right)$.

The representation is trained using

$$
\mathcal{L}_{\mathrm{rep}}=\mathcal{L}_{\mathrm{rec}}+\beta_{\mathrm{struct}}\mathcal{L}_{\mathrm{struct}},
$$

where $\mathcal{L}_{\mathrm{rec}}=\frac{1}{2\sigma_{\mathrm{rec}}^2}\mathbb{E}_{n,t}\left[\|D_\omega(E_\phi(y_{n,t}))-y_{n,t}\|_2^2\right]$ and $\mathcal{L}_{\mathrm{struct}}=-\frac{1}{N}\sum_n\sum_k\gamma_{nk}\ell_{nk}$ (This is where intervention and causal structure kicks in). After each representation update, the latent trajectories are re-encoded and the particles are updated by:

- fitting $H_k$ using responsibility-weighted regression;
- updating $(G_k^{\mathrm{lag}},G_k^{\mathrm{inst}})$ using local Metropolis-Hastings proposals;
- updating $\gamma_{nk}$ and $\pi_k$;
- reinitializing persistently empty components.


#### Stage 1, Block 2: coupled graph-and-mechanism flow

The updated particles simulate variable-length recordings under all supported intervention conditions. Conditioned on $c=C_\psi(Y_{1:L},I)$, a shared flow jointly models:

- each lagged edge as a binary variable;
- each instantaneous pair as one of $\{\text{no edge},\,i\to j,\,j\to i\}$;
- the mechanism vector $H$ through a continuous velocity field.

During sampling, cycle-forming instantaneous updates are masked, ensuring that $G^{\mathrm{inst}}$ remains a DAG.


## Repository structure

```text
.
├── README.md
├── discovery/
│   ├── 00_causal_discovery_check_particle.ipynb
│   └── 01_causal_discovery_flow.ipynb
├── dynamic_CRL/
│   ├── 01_FM_dynamicCRL_hardInt.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── sim.py
│   ├── models.py
│   ├── particles.py
│   ├── fm.py
│   ├── train.py
│   ├── diagnostics.py
│   └── utils.py
└── recycle/
    ├── 10_dynamicCRL_1regime_particle.ipynb
    ├── 11_dynamicCRL_2regimes_particle.ipynb
    └── 12_dynamicCRL_2regimes_flow.ipynb
```


### `discovery/`

Early observed-variable causal-discovery experiments.

- `00_causal_discovery_check_particle.ipynb`: check the particles.
- `01_causal_discovery_flow.ipynb`: train flow for $q_\theta(G,H\mid X)$ in a five-dimensional, two-graph simulation.


### `dynamic_CRL/`

The active, modular dynamic causal-representation-learning experiment.

- `01_FM_dynamicCRL_hardInt.ipynb`: notebook for experiment and results display.
- `sim.py`: simulations.
- `models.py`: recording encoder (compress $(Y,I)$ into $c$), causal encoder-decoder, and coupled flow network.
- `particles.py`: everything about particle sampling, including the graph space, mechanism fitting, responsibilities, structural losses, and local particle movement.
- `fm.py`: everything about Flow Matching, including graph/mechanism representations, particle-based simulation, flow training, and posterior sampling.
- `train.py`: training.
- `diagnostics.py`: metrics, summaries, and plots.
- `utils.py`:graph, padding, time-encoding, and reproducibility utilities.


## Related work

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [DeFoG: Discrete Flow Matching for Graph Generation](https://proceedings.mlr.press/v267/qin25d.html) (FM for graph)
- [iCITRIS: Causal Representation Learning for Instantaneous and Temporal Effects in Interactive Systems](https://arxiv.org/abs/2206.06169)
