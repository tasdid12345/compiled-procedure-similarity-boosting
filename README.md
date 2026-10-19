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

## Reference

This project is motivated by the following paper:

> N Pejić, M Cvetanović (2019). *Estimating similarity between differently compiled procedures using neural networks*. 2019 27th. https://ieeexplore.ieee.org/abstract/document/8971103/

Paper author: Prof. Miloš Cvetanović, University of Belgrade (cmilos@etf.bg.ac.rs)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
