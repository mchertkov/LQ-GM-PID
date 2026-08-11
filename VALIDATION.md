# Reproducibility validation

All three final notebooks were executed from top to bottom in a clean rerun on 2026-08-11 using Python 3.13.5 and CPU PyTorch 2.10.0.

- `experiments/exp_E1_corridor.ipynb`: 20 code cells; 0 error outputs; 0 unexecuted code cells.
- `experiments/exp_E2_multi_entrance.ipynb`: 22 code cells; 0 error outputs; 0 unexecuted code cells.
- `experiments/exp_H1_highdim_scaling.ipynb`: 18 code cells; 0 error outputs; 0 unexecuted code cells.
- `python -m py_compile experiments/*.py lqgm_pid/*.py`: passed.

Key regenerated metrics:

- E1 corridor loss: 0.7024818890 -> 0.4542449088 (35.3371% reduction).
- E2 corridor loss: 0.7733474346 -> 0.5074254244 (34.3858% reduction).
- H1-A at d=16, M=8: branch times B0/B2/B1 = 0.3917/0.4617/0.5083.
- H1-A at d=32, M=8: branch times B0/B2/B1 = 0.4250/0.4983/0.5983.

The generated figures are in `figs/` and numerical summaries in `results/`.
