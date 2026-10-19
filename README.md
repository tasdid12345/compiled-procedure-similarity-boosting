# Robustness of Pair Similarity Under Unseen Compiler Profiles

Estimating whether two compiled procedures come from the same source is useful for binary code search, but a
classifier trained on one compiler setup may not transfer when the optimisation settings change. The paper
"Estimating similarity between differently compiled procedures using neural networks" (2019) proposes neural
networks for this task. This repository tests how well a similarity classifier keeps its ranking when the
test procedures are compiled under a profile it never saw during training.

**Question:** Does a class-balanced, cross-validated gradient-boosting pair classifier keep its similarity
ranking better than a standard MLP pair classifier when the test procedures are compiled under an unseen
optimisation profile?

**Dataset:** Simulated in numpy, not the paper's own data. Procedures are opcode-count histograms over about
60 opcode classes. Same-source pairs are produced by compiler-like transforms (instruction substitution,
dead-code insertion, register-renaming noise, drop/merge optimisation). Training uses profiles A and B; the
test uses unseen profile C.

**Baseline:** a standardised MLPClassifier on pair features [|a-b|, a*b], standing in for the paper's neural
similarity network (the paper gives no architecture). This is not the paper's method.

**Improved model:** a class-balanced LightGBM, tuned by 3-fold grouped cross-validation on training
procedures only. It uses richer pair features: L1 distance, cosine similarity and per-opcode count differences
normalised with training-profile statistics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json`. A bar chart is saved to `figures/auc_comparison.png`.

## Results (latest run)

On a simulated opcode-histogram dataset (profiles A/B train, unseen profile C test), a standard MLP pair classifier reached ROC-AUC 0.995 while a class-balanced LightGBM reached 0.999, a small gain that came with lower F1@0.5 (0.869 to 0.810) and worse log-loss (0.590 to 0.771).

| Metric | Value |
|---|---|
| baseline_score | 0.9947 |
| improved_score | 0.9989 |
| improvement | 0.004158 |
| baseline_score_std | 0.0008213 |
| improved_score_std | 0.00000699 |
| baseline_f1 | 0.8693 |
| improved_f1 | 0.8099 |
| baseline_logloss | 0.5902 |
| improved_logloss | 0.7713 |
| best_cv_roc_auc | 1 |
| selected_num_leaves | 15 |
| selected_learning_rate | 0.1 |
| n_train_procedures | 3000 |
| n_test_procedures | 1000 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data, not real binaries; results are not comparable to the paper. Baseline has no tuning (paper gives no architecture). Test uses only the unseen profile C and procedures held out from training. Std is over 3 model seeds (ddof=0) on one fixed simulated dataset. Normalisation statistics come from training procedures under profiles A and B only.

## Reference

This project is motivated by the following paper:

> N Pejić, M Cvetanović (2019). *Estimating similarity between differently compiled procedures using neural networks*. 2019 27th. https://ieeexplore.ieee.org/abstract/document/8971103/

Paper author: Prof. Miloš Cvetanović, University of Belgrade (cmilos@etf.bg.ac.rs)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
