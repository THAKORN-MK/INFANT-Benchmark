# Implementation ledger — 2026-11-17-paper-benchmark

- User requested direct work in `D:\INFANT Benchmark`; directory is not a Git checkout. No worktree, commit or push was created.
- Ruling: use a separate WSL venv; preserve TensorFlow environment. CPU PyTorch was used for synthetic component checks; CUDA installation and Mamba compatibility are being checked separately.
- Ruling: process one tier per run, use folder/metadata label as dataset README specifies, report filename-suffix discrepancies and cross-tier conflicts, and require acknowledgement. Cross-tier results cannot be claimed from these runs.
- Ruling: group byte-identical files into folds; this departs from ungrouped clip-level CV and is not proof of subject independence. Test metrics are clip-weighted.
- Ruling: fixed 80 epochs by default; no outer evaluation data used to select checkpoint. Exact paper checkpoint policy is unavailable.
- Ruling: Gaussian KAN, proxy 213 features, augmentation magnitudes and loss epsilon are explicit approximations; source authors' implementation remains unavailable.
- Validation is limited to tests without any model fitting; user explicitly forbids assistant-run training.
