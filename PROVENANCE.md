# Snapshot provenance and exclusions

This public-preparation directory is a selective snapshot of the local research checkout at the original local research checkout, copied on 2026-10-26. It is not a clone of that checkout and has no remote configured. The original checkout and the separate university backup were inspected read-only; neither was modified.

The C/Python source paths are retained where practical. `experiments/validated/` is the reviewed implementation for the historical full-crossing model; `c/` and `python/` retain the earlier research implementations and analysis. The two PNGs under `img/` are copies of stored historical figures. `reproducibility/runs.csv` is the existing user-recorded 11-row historical summary; its original configurations are incomplete and it is not a complete record of the 52-job campaign.

Excluded by design: raw CSV traces, simulation outputs, build products, Python virtual environments, logs, RNG registries/stream-allocation records, university configuration and launch scripts, remote-compute templates, and operational campaign metadata. Third-party papers are cited by DOI and are not copied. No credentials or university server details are included. This selection is a portfolio convenience, not a university intellectual-property clearance; publication permission for research source/results remains the researcher's responsibility to confirm.

The historical figures were selected from stored project outputs and were not regenerated during snapshot preparation. Because their underlying traces are excluded, the figures cannot be reproduced from this repository alone. See the README for the model-convention distinction and scientific limits.
