"""Read archived CSVs without fitting normalizers, classifying atoms, or plotting.

Run from the repository root:
  .venv/bin/python -B -m experiments.validated.historical_diagnostics --output PATH
The output must be new. No input is modified and no reconstruction is attempted.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .config import HistogramSpec


def inspect_historical(root):
    root = Path(root)
    histogram = HistogramSpec(tuple(5 + .5*i for i in range(71)))
    entries = (
        ("csv/taxi_samples.csv", ["C", "N", "X"], 0, "unrestricted", None),
        ("csv/mcmc_trace.csv", ["t", "C", "accepted"], 5000, "finite_M", 40),
        ("csv/mcmc_theta_p5.csv", ["t", "C", "accepted"], 5000, "unknown", None),
        ("csv/mcmc_theta_m6.csv", ["t", "C", "accepted"], 5000, "finite_M", 40),
    )
    reports = []
    for name, header, burn, ensemble, cap in entries:
        path = root/name
        costs = []
        rows = 0
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != header:
                raise ValueError(f"unexpected archived CSV header: {name}")
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"malformed archived CSV row: {name}")
                values = [float(x) for x in row]
                if any(not math.isfinite(x) for x in values):
                    raise ValueError(f"nonfinite archived CSV value: {name}")
                if rows >= burn:
                    cost = values[header.index("C")]
                    if cost <= 0:
                        raise ValueError(f"invalid archived cost: {name}")
                    costs.append(cost)
                rows += 1
        if not costs:
            raise ValueError(f"no retained archived costs: {name}")
        reports.append({
            "path": name, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "header": header, "rows": rows, "diagnostic_burn_in_rows": burn,
            "retained_rows": len(costs), "minimum_cost": min(costs), "maximum_cost": max(costs),
            "mean_cost": math.fsum(costs)/len(costs),
            "underflow_rows": sum(c < histogram.edges[0] for c in costs),
            "overflow_rows": sum(c > histogram.edges[-1] for c in costs),
            "inside_reporting_range_rows": sum(histogram.edges[0] <= c <= histogram.edges[-1] for c in costs),
            "serialized_integer_cost_rows": sum(c.is_integer() for c in costs),
            "integer_costs_are_not_atom_labels": True,
            "structural_atom_classification": "unavailable: no long_steps field; costs are rounded",
            "inferred_sampling_ensemble": ensemble, "inferred_M": cap,
            "ensemble_metadata_status": "current-source inference, not complete generating provenance"
                if ensemble != "unknown" else "theta +5 exact generating configuration missing",
            "known_normalizer_available": False if burn else True,
        })
    return {
        "scope": "read-only descriptive diagnostics; no scientific combined reconstruction",
        "histogram": histogram.to_dict(),
        "files": reports,
        "combined_historical_curve_validated": False,
        "reconstruction_performed": False,
        "blocking_reasons": [
            "theta +5 generating configuration, cap and full run provenance are incomplete",
            "finite_M windows cannot be pooled as the unrestricted Direct MC baseline",
            "archived overlap-fitted normalizers are not exact known baseline-matched normalizers",
            "biased CSVs omit N and all CSVs omit structural above-threshold step counts",
            "rounded integer costs cannot establish atom membership",
            "equilibration, dependence and fitted-normalizer uncertainty remain unvalidated",
        ],
        "future_support_requirement": (
            "T09: explicitly verified u_k(C,N) with cap indicators and matched absolute "
            "normalizers, plus unrestricted coverage; or regenerate common-support windows"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_historical(args.root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote descriptive diagnostics: {args.output}")
    print("Historical combined reconstruction remains unvalidated.")


if __name__ == "__main__":
    main()
