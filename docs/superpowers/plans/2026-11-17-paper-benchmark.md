# Sinc-Mamba-KAN benchmark implementation plan

**Goal:** Prepare a transparent paper-based training implementation and inspect the supplied V1-like data without training.

**Architecture:** Validate metadata and audio evidence before assigning folds; implement waveform and descriptor branches and fit preprocessing only on training data; use explicit configurations and isolated run directories. Document paper-only versus proxy assumptions.

**Tech stack:** Python 3.10+, NumPy, SciPy, librosa, scikit-learn, PyTorch and Mamba, subject to environment compatibility.

**Spec:** `docs/superpowers/specs/2026-11-17-sinc-mamba-kan-design.md`.

## Global constraints

- Do not train any model, alter source audio, push, or silently install into existing TensorFlow environment.
- Do not invent identities, exact 213 descriptors, hyperparameters, or paper-equivalence; retain provenance and deviations.
- Dataset conflicts block a scientifically defensible training run until resolved by evidence.

## Tasks

1. Read PDF and inspect dataset, metadata, versions and duplicates. Add tests for metadata paths, tier labels, hash conflicts and missing files before code; verify on fixtures and actual data.
2. Record paper methods, nine baselines, protocol and limitations in an accessible Thai document with page references; separate any reproduction decisions from what paper says.
3. Test-first implement waveform preprocessing, documented proxy descriptors and fold-local transforms. Validate only on synthetic fixtures, no training.
4. Test-first implement separate model and training lifecycle, explicit runtime checks, disjoint outer-fold reporting and metadata; retain assumptions and refuse unsafe data. Validate forward/shapes if dependencies are available; otherwise disclose unverified status.
5. Document setup and commands without claiming readiness until data and dependencies have passed checks. Review code and run complete applicable tests.

## Review focus

- Traversal and metadata references must not escape the dataset root.
- Identical recordings in different classes must be surfaced and never silently relabelled.
- Raw test-fold information must not fit scaler/selector or choose checkpoints.
- An unverified clip identifier must not become a subject-level guarantee.
- Missing PyTorch/Mamba/CUDA must produce a clear blocker rather than a false success.
