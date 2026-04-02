#!/usr/bin/env python3
"""Sequential pair runner: runs baseline and candidate as fully separate
processes (one after the other, never overlapping) and compares val_bpb.

Usage:
    # Default baseline vs baseline (sanity check):
    python3 run_mlx_sequential_pair.py

    # Baseline vs leaky_relu2 candidate:
    python3 run_mlx_sequential_pair.py --candidate-env ACTIVATION_TYPE=leaky_relu2 LEAKY_RELU_SLOPE=0.5
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

STEP_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+)")
TARGET_STEPS = (600, 800, 1000)


def parse_env_pairs(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty env key in: {pair}")
        env[key] = value
    return env


def run_one(label: str, script_path: Path, repo_root: Path, out_dir: Path, env_overrides: dict[str, str]) -> Path:
    """Spawn a single training run as a subprocess and wait for it to finish."""
    env = os.environ.copy()
    env.update(env_overrides)
    run_id = env_overrides["RUN_ID"]
    log_path = out_dir / f"{run_id}.txt"
    cmd = [sys.executable, str(script_path)]

    print(f"\n{'='*60}")
    print(f"[{label}] Starting — RUN_ID={run_id}")
    print(f"[{label}] command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    # Force garbage collection between runs to free any lingering memory
    gc.collect()

    if not log_path.exists():
        raise FileNotFoundError(f"Expected log file was not created: {log_path}")
    print(f"\n[{label}] Done — log saved to {log_path}")
    return log_path


def parse_val_bpb_by_step(log_path: Path) -> dict[int, float]:
    by_step: dict[int, float] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = STEP_RE.match(line)
        if match:
            by_step[int(match.group(1))] = float(match.group(4))
    return by_step


def classify(delta_1000: float, mean_early_delta: float) -> str:
    if delta_1000 <= -0.003 and mean_early_delta <= -0.002:
        return "✅ PROMOTE"
    if delta_1000 >= 0.001 or mean_early_delta >= 0.001:
        return "❌ REJECT"
    return "⚠️  BORDERLINE"


def compare(baseline_log: Path, candidate_log: Path) -> None:
    baseline_vals = parse_val_bpb_by_step(baseline_log)
    candidate_vals = parse_val_bpb_by_step(candidate_log)

    for label, vals in [("baseline", baseline_vals), ("candidate", candidate_vals)]:
        missing = [s for s in TARGET_STEPS if s not in vals]
        if missing:
            raise ValueError(
                f"{label} log missing validation at steps {missing}. "
                "Ensure VAL_LOSS_EVERY is set to 200."
            )

    deltas = {step: candidate_vals[step] - baseline_vals[step] for step in TARGET_STEPS}
    mean_early_delta = sum(deltas.values()) / len(TARGET_STEPS)
    verdict = classify(deltas[1000], mean_early_delta)

    print(f"\n{'='*60}")
    print("  Step-1000 Sequential Pair Summary")
    print(f"{'='*60}")
    print(f"  baseline_log:  {baseline_log}")
    print(f"  candidate_log: {candidate_log}")
    print(f"{'─'*60}")
    print(f"  {'Step':<8} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}")
    print(f"  {'─'*44}")
    for step in TARGET_STEPS:
        print(
            f"  {step:<8} {baseline_vals[step]:>12.6f} {candidate_vals[step]:>12.6f} "
            f"{deltas[step]:>+12.6f}"
        )
    print(f"{'─'*60}")
    print(f"  mean_early_delta(600,800,1000): {mean_early_delta:+.6f}")
    print(f"  verdict: {verdict}")
    print(f"{'─'*60}")
    print("  Rule: promote if delta_1000 ≤ -0.003 AND mean_early ≤ -0.002")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline then candidate sequentially (never both in RAM) and compare val_bpb."
    )
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment override applied to both runs.",
    )
    parser.add_argument(
        "--candidate-env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment override applied only to the candidate run.",
    )
    parser.add_argument(
        "--baseline-env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment override applied only to the baseline run.",
    )
    parser.add_argument("--seed", default="1337", help="Shared seed (default: 1337).")
    parser.add_argument("--label", default="mlx_seq", help="Prefix for RUN_ID values.")
    parser.add_argument(
        "--compare-only", nargs=2, metavar=("BASELINE_LOG", "CANDIDATE_LOG"),
        help="Skip training — just compare two existing log files.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    script_path = here / "train_gpt_mlx.py"
    out_dir = here / "logs_mlx"

    if args.compare_only:
        compare(Path(args.compare_only[0]), Path(args.compare_only[1]))
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    common_env = {
        "SEED": args.seed,
        "OUT_DIR": str(out_dir),
        "PYTHONUNBUFFERED": "1",
    }
    common_env.update(parse_env_pairs(args.env))

    tag = uuid.uuid4().hex[:8]

    # --- Run 1: Baseline ---
    baseline_env = dict(common_env)
    baseline_env.update(parse_env_pairs(args.baseline_env))
    baseline_env["RUN_ID"] = f"{args.label}_baseline_{tag}"
    baseline_log = run_one("BASELINE", script_path, repo_root, out_dir, baseline_env)

    # --- Run 2: Candidate ---
    candidate_env = dict(common_env)
    candidate_env.update(parse_env_pairs(args.candidate_env))
    candidate_env["RUN_ID"] = f"{args.label}_candidate_{tag}"
    candidate_log = run_one("CANDIDATE", script_path, repo_root, out_dir, candidate_env)

    # --- Compare ---
    compare(baseline_log, candidate_log)


if __name__ == "__main__":
    main()
