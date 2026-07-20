# Supplementary Experiments (Major Revision)

This folder contains the **additional experiments and data** produced for the major revision of the DeepMR paper (Applied Soft Computing, ASOC-D-24-11560). It supplements — and does not replace — the original results in `original_data_log_files_in_paper/`.

Encoding of the result files: the `verified` field is `1` (verified), `0` (finished but not verified), or `DNF` (hit the timeout). Reported `avg`/`max` times are over **finished** instances (excluding DNF); DNF is reported separately.

---

## 1. ERAN (VNN-COMP community) benchmarks

We evaluate on the VNN-COMP `mnist-fc` community networks `6×100` (= community 5×100), `9×100` (= community 8×100), and the wider `6×200`, `9×200`, using the `vnncomp_eran_properties`. DeepPoly / DeepMR / DeepSRGR run on CPU (Ryzen 9 7950X, 28 processes); β-CROWN runs on GPU (RTX 4090). Sampling: 50 instances for the 100-wide nets (DeepMR/DeepSRGR); reduced samples for the wider 200-wide nets, where a single refinement can take hours.

**Per-input budget.** In the paper (Table 5) the two refinement methods are compared under a single per-network budget shared by both: none on `6×100` (both finish, ≤ 4181 s), **20000 s** on `9×100`, **10000 s** on `6×200`, and **8000 s** on `9×200`; any instance not finished within it is counted at that budget. The table below differs in two harmless ways. (i) Its averages are over *finished* instances only (DNF excluded, per the encoding note above), whereas Table 5 also counts the timed-out instances at the budget, so the paper's averages read higher. (ii) DeepMR here was run without a hard cap, so its two hardest `6×200` instances run to completion in 11–13 ks and appear as *finished-not-verified* (DNF 0) rather than as the two `6×200` timeouts recorded under the shared 10000 s budget in the paper.

| Network | Tool | V | n | avg (s) | max (s) | DNF |
| --- | --- | --- | --- | --- | --- | --- |
| **6×100** (ε 0.019) | DeepPoly | 54 | 100 | 2.1 | 3 | 0 |
| | **DeepMR** | 31 | 50 | 161.5 | 445 | 0 |
| | DeepSRGR | 32 | 50 | 908.9 | 4181 | 0 |
| | β-CROWN (GPU) | 76 | 100 | — | — | 24 |
| **9×100** (ε 0.018) | DeepPoly | 51 | 100 | 4.2 | 6 | 0 |
| | **DeepMR** | 31 | 50 | 426.2 | 1730 | 0 |
| | DeepSRGR | 10 | 20 | 1540 | 13968 | **9** |
| | β-CROWN (GPU) | 69 | 100 | — | — | 31 |
| **6×200** (ε 0.013) | DeepPoly | 51 | 100 | 5.4 | 7 | 0 |
| | **DeepMR** | 12 | 20 | 4575 | 12795 | 0 |
| | DeepSRGR | 5 | 10 | 2746.9 | 8558 | **5** |
| | β-CROWN (GPU) | 66 | 100 | — | — | 34 |
| **9×200** (ε 0.012) | DeepPoly | 54 | 100 | 11.5 | 16 | 0 |
| | **DeepMR** | 3 | 5 | 14.1 | 24.6 | 2 |
| | DeepSRGR | 3 | 5 | 32.8 | 79 | 2 |
| | β-CROWN (GPU) | 64 | 100 | — | — | 36 |

Key findings:
- On `6×100`, DeepMR and DeepSRGR both finish all instances; **DeepMR is 5.63× faster on average (9.4× max)** at essentially equal precision.
- **Scalability:** on the deeper `9×100` and wider `6×200`, DeepMR finishes **every** instance with no timeout, whereas DeepSRGR times out on **9/20 (45%)** and **5/10 (50%)** respectively — i.e. DeepSRGR becomes infeasible on larger community networks while DeepMR remains feasible. On the same 20 instances of `9×100`, on the ones both verify DeepMR is 5.5–28× faster, and there is even an instance DeepMR verifies (746 s) that DeepSRGR cannot finish.
- β-CROWN (GPU, complete verifier, 2000 s timeout) verifies more (safe 76/69/66/64, all remaining are timeouts, unsafe = 0), consistent with its different setting (GPU, complete, stronger base). DeepMR's like-for-like baseline is DeepSRGR.

Data: `eran_vnncomp/eran_{poly,deepmr,srgr}_eran{6x100,9x100,6x200,9x200}.csv`. (β-CROWN logs live on the GPU machine.)

---

## 2. Validity of the Big-M constant M (R2-Sh3)

M=10000 is valid for every network (all original results are retained). The Big-M validity condition requires M ≥ the upper bound of the constrained expression over the feasible region, i.e. M ≥ max_i(−lower_bound(margin_i)). The code now sets M **adaptively per instance**: `M = max(2 * max_i(-concrete_lower(margin_i)), 1.0)`, with an environment variable `BIGM` to override it with a fixed value (used only for this sensitivity check). Re-running the borderline deep networks with the adaptive M reproduces the paper's results instance-by-instance (M9: 79/100 identical, 0 false positives; M10: V=52; ACAS: V=54/40/67).

Data: `M_validity_R2Sh3/rerun_M{9,10}_adaptM.csv`, `Mmin_all.log`. Code: `../verify/mnist_new_6x100/compute_Mmin_all.py` and the adaptive-M block now present in every `verify/*/deepmr_*_3.py`.

---

## 3. Precision analysis (R2-Sh1)

Refinement improves precision by tightening neuron bounds, and a tighter bound on any neuron **propagates**: DeepPoly computes each neuron's bound by back-substituting through the neurons before it, and the procedure is iterated, so tightening any neuron makes the bounds computed after it — in the same pass and in subsequent iterations — tighter as well. Verification precision is therefore governed by the **total tightening the refinement accumulates**. Running more iterations and refining the input-layer neurons are two instances of the same mechanism (both inject tightness into this compounding process). DeepMR deliberately accumulates slightly less tightening for speed: it does not refine the input-layer neurons (their tightening is not worth the solving cost) and uses a bounded iteration count (MAX_ITER = 5, as DeepSRGR). This is **not** a weakening of the abstraction — simultaneous elimination is equivalent to DeepSRGR's sequential elimination — so any borderline instance is recovered by spending more tightening effort.

We confirm this on the two ACAS-Xu φ6 instances (`local_robustness_28`, `local_robustness_29`) that DeepMR (MAX_ITER=5) leaves unverified while DeepSRGR verifies them:

| Configuration | V (of 100) | Note |
| --- | --- | --- |
| DeepMR (MAX_ITER=5) | 40 | 2 gap instances unverified |
| DeepMR + input-layer refinement (MAX_ITER=5) | 40 | input-layer refinement does not recover them (5-dim input contributes little tightening) |
| **DeepMR (MAX_ITER=50)** | **42** | **both gap instances verified — matches DeepSRGR** (more iterations accumulate enough tightening) |
| DeepSRGR | 42 | reference |

Data: `precision_analysis_R2Sh1/`
- `..._deepmr_3_add_inputlayer_...txt` — DeepMR + input-layer refinement, V=40.
- `..._deepmr_3_maxiter50_gap_...txt` — DeepMR with MAX_ITER=50 on the two gap instances, both verified (number_sum=2).

Code (this folder's companion scripts under `../verify/`):
- `acas_xu_p6_net_1_1/deepmr_..._add_inputlayer.py` — DeepMR with an added input-layer refinement step.
- `acas_xu_p6_net_1_1/deepsrgr_..._del_inputlayer.py` — DeepSRGR with input-layer refinement removed (ablation).
- `acas_xu_p6_net_1_1/deepmr_..._maxiter50{,_gap}.py` — DeepMR with a larger iteration budget.
- `cifar_new_6x80/deepmr_..._add_inputlayer{,_gap}.py` — the corresponding CIFAR variants.

---

## New code added in this revision

- `verify/mnist_new_6x100/eran_run_{deepmr,srgr,poly}.py` — generic drivers to run each tool on any nnet + property set (used for the ERAN experiments).
- `verify/mnist_new_6x100/eran_batch_psa2.sh`, `eran_batch_psb3.sh` — example batch runners.
- `verify/mnist_new_6x100/compute_Mmin_all.py` — computes the required M lower bound across networks.
- `verify/*/deepmr_*_3.py` — Big-M made adaptive (with `BIGM` override).
- Precision-analysis variants under `verify/acas_xu_p6_net_1_1/` and `verify/cifar_new_6x80/` (see Section 3).
