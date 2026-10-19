"""Pair similarity of simulated compiled procedures: MLP baseline vs class-balanced LightGBM.
Question: does the improved model keep a higher ROC-AUC than the baseline on an unseen compiler profile (C)?
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score, log_loss

SEED = 42
SEEDS = [42, 43, 44]          # model seeds for the randomised models
N_OPS = 60                    # opcode classes in each histogram
N_SRC = 4000                  # simulated source procedures
N_TRAIN_PAIRS = 60000
N_TEST_PAIRS = 20000
OUT_JSON = "results.json"


def make_profile(rng, sub_rate, n_dead, dead_rate, ren_sigma, drop, n_merge):
    """One simulated compiler profile: its own substitution map, dead-code ops, noise and merges."""
    return dict(sub_rate=sub_rate, sub_map=rng.permutation(N_OPS),
                dead=rng.choice(N_OPS, n_dead, replace=False), dead_rate=dead_rate,
                ren_sigma=ren_sigma, drop=drop,
                merges=rng.permutation(N_OPS)[:2 * n_merge].reshape(-1, 2))  # disjoint (i -> j) merges


# Profiles A and B are used for training; C is the unseen optimisation profile used only for testing.
_prng = np.random.default_rng(SEED)
PROFILES = {
    "A": make_profile(_prng, 0.05, 5, 0.02, 0.05, 0.00, 3),
    "B": make_profile(_prng, 0.10, 8, 0.05, 0.10, 0.05, 5),
    "C": make_profile(_prng, 0.18, 12, 0.10, 0.15, 0.12, 8),
}


def compile_rows(x, prof, rng):
    """Apply one compiler profile to a batch of opcode-count histograms (non-negative ints)."""
    y = x.copy()
    # instruction substitution: move counts from opcode i to sub_map[i] (a permutation, so no clashes)
    moved = rng.binomial(y, prof["sub_rate"])
    y -= moved
    y[:, prof["sub_map"]] += moved
    # dead-code insertion: extra counts spread over a few dead opcodes
    lam = prof["dead_rate"] * x.sum(1, keepdims=True) / len(prof["dead"])
    y[:, prof["dead"]] += rng.poisson(np.broadcast_to(lam, (len(x), len(prof["dead"]))))
    # register-renaming noise: multiplicative jitter on counts
    y = rng.poisson(y * rng.lognormal(0.0, prof["ren_sigma"], y.shape))
    # optimisation level: drop some instructions, merge opcode pairs
    y = rng.binomial(y, 1.0 - prof["drop"])
    for i, j in prof["merges"]:
        y[:, j] += y[:, i]
        y[:, i] = 0
    return y


def compile_mixed(x, names, rng):
    """Compile each row with the profile named in names[row]."""
    out = np.empty_like(x)
    for n in np.unique(names):
        m = names == n
        out[m] = compile_rows(x[m], PROFILES[n], rng)
    return out


def make_procedures(rng):
    """Opcode histograms: procedures from 300 families with heterogeneous mixes and lengths."""
    base = rng.gamma(0.5, 1.0, (300, N_OPS))
    fam = rng.integers(0, 300, N_SRC)
    rate = base[fam] * rng.gamma(4.0, 0.25, (N_SRC, N_OPS))
    rate /= rate.sum(1, keepdims=True)
    length = np.maximum(rng.lognormal(np.log(300), 0.6, N_SRC), 20)
    return rng.poisson(rate * length[:, None]).astype(np.int64)


def make_pairs(pool, anchor_names, partner_names, n, rng):
    """Half same-source (label 1) and half different-source (label 0) pairs, each side compiled."""
    h = n // 2
    pos = rng.integers(0, len(pool), h)
    neg_a = rng.integers(0, len(pool), n - h)
    neg_b = (neg_a + rng.integers(1, len(pool), n - h)) % len(pool)  # guaranteed different procedure
    anchor = np.concatenate([pos, neg_a])
    partner = np.concatenate([pos, neg_b])
    y = np.concatenate([np.ones(h), np.zeros(n - h)]).astype(int)
    A = compile_mixed(pool[anchor], rng.choice(anchor_names, n), rng)
    B = compile_mixed(pool[partner], rng.choice(partner_names, n), rng)
    return A, B, y, anchor


def train_stats(train_pool, rng):
    """Per-opcode log-count mean/std from training procedures under training profiles A and B only."""
    v = np.vstack([compile_mixed(train_pool, np.full(len(train_pool), n), rng) for n in ("A", "B")])
    lv = np.log1p(v)
    return lv.mean(0), lv.std(0) + 1e-6


def feats(A, B, stats=None):
    """Baseline: [|fa-fb|, fa*fb] on frequencies. Improved adds L1, cosine and normalised count diffs."""
    fa = A / np.maximum(A.sum(1, keepdims=True), 1)
    fb = B / np.maximum(B.sum(1, keepdims=True), 1)
    f = [np.abs(fa - fb), fa * fb]
    if stats is not None:
        mu, sd = stats
        za, zb = (np.log1p(A) - mu) / sd, (np.log1p(B) - mu) / sd
        cos = (fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + 1e-12)
        f += [np.abs(fa - fb).sum(1, keepdims=True), cos[:, None], np.abs(za - zb)]
    return np.hstack(f)


def fit_mlp(X, y, seed):
    """Standard baseline: standardised MLP pair classifier (fixed architecture, no tuning)."""
    m = make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-4, early_stopping=True,
                                                      max_iter=300, random_state=seed))
    return m.fit(X, y)


def make_lgbm(nl, lr, seed):
    """Class-balanced LightGBM with the tuned leaves and learning rate."""
    return LGBMClassifier(n_estimators=300, num_leaves=nl, learning_rate=lr, class_weight="balanced",
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=seed, verbose=-1)


def tune_lgbm(X, y, groups):
    """Small grid, 3-fold CV grouped by anchor procedure, on training pairs only."""
    best = None
    for nl in (15, 31):
        for lr in (0.05, 0.1):
            aucs = []
            for tr, va in GroupKFold(n_splits=3).split(X, y, groups):
                m = make_lgbm(nl, lr, SEED).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
            s = float(np.mean(aucs))
            print(f"  CV ROC-AUC num_leaves={nl} lr={lr}: {s:.4f}")
            if best is None or s > best[0]:
                best = (s, nl, lr)
    return best


def evaluate(model, X, y):
    """Main metric ROC-AUC plus F1 at 0.5 and log-loss."""
    p = model.predict_proba(X)[:, 1]
    return dict(auc=roc_auc_score(y, p), f1=f1_score(y, p >= 0.5), logloss=log_loss(y, p))


def main():
    rng = np.random.default_rng(SEED)
    print("simulating procedures and compiling pairs ...")
    procs = make_procedures(rng)
    tr_ids, te_ids = train_test_split(np.arange(N_SRC), test_size=0.25, random_state=SEED)
    train_pool, test_pool = procs[tr_ids], procs[te_ids]
    Atr, Btr, ytr, gtr = make_pairs(train_pool, ["A", "B"], ["A", "B"], N_TRAIN_PAIRS, rng)
    Ate, Bte, yte, _ = make_pairs(test_pool, ["A", "B"], ["C"], N_TEST_PAIRS, rng)  # C is unseen
    stats = train_stats(train_pool, rng)

    Xb_tr, Xb_te = feats(Atr, Btr), feats(Ate, Bte)
    Xi_tr, Xi_te = feats(Atr, Btr, stats), feats(Ate, Bte, stats)

    print("tuning LightGBM by 3-fold grouped CV (training pairs only) ...")
    cv_auc, nl, lr = tune_lgbm(Xi_tr, ytr, gtr)
    print(f"  selected num_leaves={nl}, learning_rate={lr}")

    base_res, imp_res = [], []
    for s in SEEDS:
        print(f"seed {s}: fitting MLP baseline")
        base_res.append(evaluate(fit_mlp(Xb_tr, ytr, s), Xb_te, yte))
        print(f"seed {s}: fitting class-balanced LightGBM")
        imp_res.append(evaluate(make_lgbm(nl, lr, s).fit(Xi_tr, ytr), Xi_te, yte))

    def agg(res, k):
        v = np.array([r[k] for r in res])
        return float(v.mean()), float(v.std())

    b_auc, b_sd = agg(base_res, "auc")
    i_auc, i_sd = agg(imp_res, "auc")
    b_f1, _ = agg(base_res, "f1")
    i_f1, _ = agg(imp_res, "f1")
    b_ll, _ = agg(base_res, "logloss")
    i_ll, _ = agg(imp_res, "logloss")
    vals = [b_auc, i_auc, b_f1, i_f1, b_ll, i_ll]
    if not all(np.isfinite(vals)):
        print("ERROR: non-finite metrics")
        sys.exit(1)
    print(f"baseline ROC-AUC {b_auc:.4f} +/- {b_sd:.4f} | improved ROC-AUC {i_auc:.4f} +/- {i_sd:.4f}")

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["MLP baseline", "Balanced LightGBM"], [b_auc, i_auc], yerr=[b_sd, i_sd], capsize=6,
            color=["grey", "steelblue"])
    plt.ylabel("ROC-AUC on unseen profile C")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/auc_comparison.png", dpi=120)

    results = {
        "question": "Does a class-balanced, cross-validated gradient-boosting pair classifier keep its "
                    "similarity ranking better than a standard MLP pair classifier under an unseen optimisation profile?",
        "method": "Simulated opcode-count histograms (60 classes); same-source pairs from compiler profiles "
                  "(substitution, dead code, renaming noise, drop/merge). Train on profiles A/B, test on unseen C.",
        "dataset": {
            "name": "Simulated compiled-procedure opcode histograms (numpy, profiles A/B train, C test)",
            "source_url": "https://numpy.org",
            "n_rows": N_TRAIN_PAIRS + N_TEST_PAIRS,
            "n_train": N_TRAIN_PAIRS,
            "n_test": N_TEST_PAIRS,
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "ROC-AUC on held-out profile C (main); F1@0.5 and log-loss (secondary)",
        "higher_is_better": True,
        "baseline": "Standardised MLPClassifier (128-64, early stopping) on pair features [|fa-fb|, fa*fb]",
        "improved": "Class-balanced LightGBM, 3-fold grouped-CV tuned (num_leaves, learning_rate), with added "
                    "L1, cosine and training-normalised per-opcode count differences",
        "metrics": {
            "baseline_score": b_auc,
            "improved_score": i_auc,
            "improvement": i_auc - b_auc,
            "baseline_score_std": b_sd,
            "improved_score_std": i_sd,
            "baseline_f1": b_f1,
            "improved_f1": i_f1,
            "baseline_logloss": b_ll,
            "improved_logloss": i_ll,
            "best_cv_roc_auc": cv_auc,
            "selected_num_leaves": nl,
            "selected_learning_rate": lr,
            "n_train_procedures": int(len(tr_ids)),
            "n_test_procedures": int(len(te_ids)),
        },
        "notes": "Simulated data, not real binaries; results are not comparable to the paper. Baseline has no "
                 "tuning (paper gives no architecture). Test uses only the unseen profile C and procedures held "
                 "out from training. Std is over 3 model seeds (ddof=0) on one fixed simulated dataset. "
                 "Normalisation statistics come from training procedures under profiles A and B only.",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
