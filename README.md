# Flow Matching for Causal Discovery and Representation Learning

Just playing around with using flow matching (maybe flow map, an easy upgrade) for causal stuff. We begin with causal discovery. If it seems valuable to push forward, we can move to representation learning, dynamic versions, and intervention versions. For now, we start with causal discovery without intervention.

**Why FM**:
1. sampling rather than estimation: good for Markov equivalance class (weak identifiability is enough) & multimodality (e.g., mixture causal graph). Also, of course, uncertainty quantification.
2. high-D causal graph
3. generalization

## 1. causal discovery

Currently, the causal variables are observed as (x). For causal representation learning, we can add one more encoder-decoder layer to link observations and latent causal variables. The task is to fit a flow-based generative model for $p(G, H \mid x)$, where...
- $G$: causal graph, DAG
- $H$: parameters for mechanistic model

Core idea: similar to simulation-based inference. 
- Given observation $x$, get and move noisy particles $(G, H)$ by rough mixture model (very cheap approximated likelihood). The movement is driven by an EM-like procedure, with noise/ jitters.
- For each $(G, H)$, generate several $x_{\text{sim}}$ to get training data for FM $(G, H, x_{\text{sim}})$
- train FM $f(G, H \mid x)$. the network for G and H is shared: H is continuous, and G is a DAG. The FM for DAG is essentially discrete-FM in DAG-constraint space, see details in DeFog (only graph, not DAG, https://openreview.net/forum?id=KPRIwWhqAZ)

In short, 1) the particles give a very rough (via mixture model with rough likelihood) and noisy (good for exploration) approximated samples around posterior of $(G, H)$ given $x$, and 2) FM is trained based on simulator generated $(G, H, x_{\text{sim}})$. The resulting FM should be relatively good for generalization.

The ``1_causal_discovery_flow.ipynb`` is a simulation example.
1. data (train/ test splitted): 5-D $x$ from a mixture of 2 graph $G_1$ and $G_2$
2. figures: generated $(G, H)$ given $x_{\text{test}}$
   - distribution of $G$ for truth, particles and FM-generated given $x_{\text{test}}$
   - representative generated samples of $G$, given $x_{\text{test}, G_1}$ and $x_{\text{test}, G_2}$
   - Given $G$, check the distribution of $H$
   - posterior predictive data: given $x_{\text{test}}$, generate $(G, H)$. Then given $(G, H)$, generate $\tilde{x}$. Compare.







