## `experiments/`

Reproducible notebooks for the empirical demonstrations in the paper.

- `exp_caseB_density_diagnostic.ipynb`  
  E1: 2D corridor protocol-learning study with deterministic source, learned guide centerline, learned anisotropic stiffness, density-level corridor loss, terminal-accuracy checks, and diagnostic plots.

- `exp_E2_multi_entrance.ipynb`  
  E2: Gaussian-mixture-to-Gaussian-mixture transport via the coordinate-shift construction, including the two-entrance source, density-level optimization, entrance-merging diagnostics, and terminal-accuracy checks.

- `exp_E3_symsigma_rescaling_fixed.ipynb`  
  E3: fixed symmetric-$\sigma_k$ study for the 2D multi-entrance corridor task. Compares zero, positive, and negative prescribed linear drift schedules while optimizing only $\nu_k$ and $\beta_k$, with analytic loss/control/stiffness diagnostics and EM trajectory visualizations.

- `exp_H1_highdim_scaling.ipynb`  
  H1: high-dimensional coarse-to-fine scaling study with trunk--branch--local Gaussian-mixture targets, hand-crafted matrix-valued protocols, dimension/mode sweeps, subspace-variance diagnostics, and EM-vs-closed-form consistency checks.
