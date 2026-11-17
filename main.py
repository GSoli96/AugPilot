#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

# Su Windows con output reindirizzato la console usa cp1252 e le emoji dei log fanno crashare il programma
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Prevent OpenMP and multiprocessing crashes on macOS/Linux
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# NumPy 1.x polyfill for SciPy/Scikit-learn builds referencing np.long or np.ulong
import numpy as np
for _attr in ("long", "ulong"):
    if not hasattr(np, _attr):
        setattr(np, _attr, int)

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

# Import project configuration and runners
from config import (
    BASE_DIR,
    DATASETS_DIR,
    AUGMENTED_DIR,
    RESULTS_DIR,
    CORE_ALGORITHMS,
    EXTRA_ALGORITHMS,
    DESCRIBED_ONLY,
    DEFAULT_EPOCHS,
)
from algorithms_runner import execute_augmentation, resolve_device

# ===========================================================================
# FLAG: USARE_ALTRI_ALGORITMI
# Set to True to include algorithms from NonUsati/ (tabsyn, tabdiff, octree, etc.)
# Set to False to run only core algorithms.
# Can also be overridden from CLI with --usare-altri-algoritmi
# ===========================================================================
USARE_ALTRI_ALGORITMI: bool = False

# ===========================================================================
# NUMERO DI TUPLE DA GENERARE
# - Imposta a un intero (es. 1000, 5000) per fissare il numero esatto di campioni
# - Imposta a None per generare lo stesso numero di tuple del dataset originale
# Può essere sovrascritto anche da riga di comando con: --n-samples <numero>
# ===========================================================================
NUMERO_TUPLE_DA_GENERARE: Optional[int] = None


# ---------------------------------------------------------------------------
# Checkpoint and Resume helpers
# ---------------------------------------------------------------------------
def checkpoint_file(results_dir: Path = RESULTS_DIR) -> Path:
    """Il checkpoint sta nella cartella dei risultati scelta con --results-dir (default runtime_results/)."""
    return Path(results_dir) / "checkpoint_state.json"


def load_checkpoint(results_dir: Path = RESULTS_DIR) -> Dict[str, Any]:
    """Load checkpoint state if it exists."""
    cp = checkpoint_file(results_dir)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass
    return {"completed_keys": [], "last_metrics_file": None}


def save_checkpoint(completed_keys: List[List[str]], last_metrics_file: Optional[str] = None,
                    results_dir: Path = RESULTS_DIR):
    """Save checkpoint state to disk."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "completed_keys": completed_keys,
        "last_metrics_file": str(last_metrics_file) if last_metrics_file else None,
        "updated_at": datetime.now().isoformat(),
    }
    checkpoint_file(results_dir).write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Generic Dataset Discovery
# ---------------------------------------------------------------------------
def discover_datasets(datasets_dir: Path) -> List[Path]:
    """Dynamically discover all CSV datasets in the specified directory."""
    if not datasets_dir.exists():
        return []
    csv_files = sorted(datasets_dir.glob("*.csv"))
    return [f for f in csv_files if f.stat().st_size > 0]


# ---------------------------------------------------------------------------
# Summary Statistics Computation
# ---------------------------------------------------------------------------
def compute_and_save_summary(metrics_df: pd.DataFrame, summary_file: Path) -> pd.DataFrame:
    """Compute mean, std, min, max, count per algorithm for all monitored metrics."""
    if metrics_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "total_time_s",
        "train_time_s",
        "gen_time_s",
        "train_peak_mem_mb",
        "gen_peak_mem_mb",
        "total_mem_peak_mb",
        "rss_delta_mb",
        "train_gpu_peak_mb",
        "gen_gpu_peak_mb",
        "n_orig",
        "n_augmented",
    ]
    present_cols = [c for c in numeric_cols if c in metrics_df.columns]

    # Only include successful runs in numerical stats
    valid_df = metrics_df[metrics_df["status"] == "SUCCESS"] if "status" in metrics_df.columns else metrics_df

    if valid_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            valid_df.groupby("algorithm")[present_cols]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .round(4)
        )
        # Flatten MultiIndex columns
        summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
        summary_df.reset_index(inplace=True)

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    return summary_df


# ---------------------------------------------------------------------------
# Denial Constraints Resolver for Kamino
# ---------------------------------------------------------------------------
def resolve_dc_file(ds_name: str, dc_file_arg: Optional[str] = None, dc_dir_arg: Optional[str] = None) -> Optional[Path]:
    """Find the Denial Constraints file for a dataset."""
    if dc_file_arg:
        p = Path(dc_file_arg)
        if p.is_file() and p.exists():
            return p
        elif p.is_dir():
            dc_dir_arg = str(p)

    dirs_to_check = []
    if dc_dir_arg:
        dirs_to_check.append(Path(dc_dir_arg))
    dirs_to_check.extend([
        BASE_DIR / "Algorithms" / "kamino-master" / "testdata" / ds_name,
        BASE_DIR / "datasets",
    ])

    stem_variants = [
        ds_name,
        ds_name.replace("_min", ""),
        f"{ds_name}_min",
    ]

    for d in dirs_to_check:
        if not d.exists():
            continue
        for stem in stem_variants:
            for ext in ["_dcs.out", ".ic", "_dc.txt", ".out", ".txt"]:
                cand = d / f"{stem}{ext}"
                if cand.is_file() and cand.stat().st_size > 0:
                    return cand
    return None


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Data Augmentation Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help="Algorithms to run: single ('smote'), multiple ('smote' 'castgan'), or 'all'.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to run on (stem or filename): 'all' or list of names (e.g., 'iris0_min').",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DATASETS_DIR),
        help="Directory containing the input datasets.",
    )
    parser.add_argument(
        "--augmented-dir",
        type=str,
        default=str(AUGMENTED_DIR),
        help="Directory where augmented datasets will be saved.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory where runtime metrics and summary files will be saved.",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Enable GPU acceleration for algorithms that support it.",
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU acceleration (force CPU computation).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Hardware device to target ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--usare-altri-algoritmi",
        dest="usare_altri_algoritmi",
        action="store_true",
        default=USARE_ALTRI_ALGORITMI,
        help="Include additional algorithms from NonUsati/ directory.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=NUMERO_TUPLE_DA_GENERARE,
        help="Number of synthetic samples to generate (default: NUMERO_TUPLE_DA_GENERARE or same as input dataset size if None).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs for deep learning / generative algorithms (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="augmentation_benchmark",
        help="Custom run name prefix for output metric files.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume execution from where it was interrupted (skip already completed tasks).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore previous checkpoint and start a fresh benchmark run.",
    )
    parser.add_argument(
        "--dc-file",
        type=str,
        default=None,
        help="Path to a Denial Constraints file (.out, .ic, .txt) or directory. REQUIRED for Kamino.",
    )
    parser.add_argument(
        "--dc-dir",
        type=str,
        default=str(BASE_DIR / "Algorithms" / "kamino-master" / "testdata"),
        help="Directory to locate Denial Constraints files (default: Algorithms/kamino-master/testdata; the per-dataset subfolder is searched automatically).",
    )
    parser.add_argument(
        "--allow-fallback",
        dest="allow_fallback",
        action="store_true",
        default=False,
        help="Se il metodo reale fallisce (es. tabsyn, tabdiff, imblearn SMOTE, SyRFDRes senza RFD) ripiega su un metodo "
             "piu' semplice: il record ha status FALLBACK e method_used indica cosa e' stato eseguito. "
             "Default: l'esecuzione e' FAILED e non scrive alcun file.",
    )
    parser.add_argument(
        "--kamino-weights",
        type=str,
        default="toggled",
        choices=["toggled", "hard", "raw", "dp", "uniform", "all"],
        help="Kamino weighting strategy: 'toggled' (default), 'hard', 'raw', 'dp', 'uniform', or 'all'.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main Execution Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Resolve GPU settings
    try:
        import torch
        gpu_detected = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except ImportError:
        gpu_detected = False

    if args.use_gpu is None:
        use_gpu = gpu_detected
    else:
        use_gpu = args.use_gpu

    device = resolve_device(use_gpu, args.device)
    print("=" * 70)
    print("🚀 UNIFIED DATA AUGMENTATION BENCHMARK")
    print("=" * 70)
    print(f"• GPU Acceleration: {'ENABLED' if use_gpu else 'DISABLED'} (target device: {device})")
    print(f"• Include NonUsati Algos (USARE_ALTRI_ALGORITMI): {args.usare_altri_algoritmi}")

    # 2. Build available algorithms pool
    available_algos = list(CORE_ALGORITHMS)
    if args.usare_altri_algoritmi:
        available_algos.extend(EXTRA_ALGORITHMS)

    if "all" in [a.lower() for a in args.algorithms]:
        # 'all' = the runnable algorithms only; the described-only ones must be asked for by name
        selected_algos = [a for a in available_algos if a not in DESCRIBED_ONLY]
        skipped = [a for a in available_algos if a in DESCRIBED_ONLY]
        if skipped:
            print(f"• 'all' skips the described-only algorithms (surrogates, no real run): {', '.join(skipped)}")
    else:
        # Normalize and filter
        selected_algos = [a.lower().strip() for a in args.algorithms if a.lower().strip() in available_algos]
        described = [a for a in selected_algos if a in DESCRIBED_ONLY]
        if described:
            print(f"⚠️  Warning: described-only algorithms requested by name (surrogate wrappers): {described}")
        # Check if user requested an unknown algorithm
        unknown = [a for a in args.algorithms if a.lower().strip() not in available_algos and a.lower() != "all"]
        if unknown:
            print(f"⚠️  Warning: unrecognized or unavailable algorithms: {unknown}")

    if not selected_algos:
        print("❌ Error: No valid algorithms selected. Available:", available_algos)
        sys.exit(1)

    print(f"• Selected Algorithms ({len(selected_algos)}): {', '.join(selected_algos)}")

    # 3. Discover datasets generically
    datasets_dir = Path(args.datasets_dir)
    all_dataset_paths = discover_datasets(datasets_dir)
    if not all_dataset_paths:
        print(f"❌ Error: No CSV datasets found in {datasets_dir}!")
        sys.exit(1)

    if "all" in [d.lower() for d in args.datasets]:
        selected_dataset_paths = all_dataset_paths
    else:
        req = set(d.lower().replace(".csv", "").strip() for d in args.datasets)
        selected_dataset_paths = [p for p in all_dataset_paths if p.stem.lower() in req or p.name.lower() in req]
        if not selected_dataset_paths:
            print(f"❌ Error: None of the requested datasets {args.datasets} were found in {datasets_dir}.")
            sys.exit(1)

    print(f"• Selected Datasets ({len(selected_dataset_paths)}): {', '.join(p.stem for p in selected_dataset_paths[:8])}{' ...' if len(selected_dataset_paths) > 8 else ''}")

    # 4. Setup output directories
    augmented_dir = Path(args.augmented_dir)
    results_dir = Path(args.results_dir)
    augmented_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 5. Setup file paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = results_dir / f"{args.run_name}_{timestamp}.csv"
    summary_file = results_dir / f"{args.run_name}_summary_{timestamp}.csv"

    # 6. Resume handling
    checkpoint = load_checkpoint(results_dir) if args.resume else {"completed_keys": [], "last_metrics_file": None}
    completed_keys: Set[Tuple[str, str]] = set((k[0], k[1]) for k in checkpoint.get("completed_keys", []))
    prior_records = []

    # If resuming and a previous metrics file exists, load prior records
    if args.resume and checkpoint.get("last_metrics_file") and Path(checkpoint["last_metrics_file"]).exists():
        try:
            prior_df = pd.read_csv(checkpoint["last_metrics_file"])
            prior_records = prior_df.to_dict("records")
            # Also sync completed_keys from file
            for r in prior_records:
                if r.get("status") == "SUCCESS":
                    completed_keys.add((str(r.get("algorithm")), str(r.get("dataset"))))
            print(f"🔄 Resumed from checkpoint: {len(completed_keys)} tasks already completed.")
        except Exception as e:
            print(f"⚠️  Could not load prior metrics file: {e}")

    # Write initial header / prior records to current metrics file
    current_records = list(prior_records)
    if current_records:
        pd.DataFrame(current_records).to_csv(metrics_file, index=False)

    # 7. Build task list
    effective_algos = []
    for a in selected_algos:
        if a == "kamino" and args.kamino_weights == "all":
            effective_algos.extend(["kamino-toggled", "kamino-hard", "kamino-raw", "kamino-dp", "kamino-uniform"])
        else:
            effective_algos.append(a)

    tasks: List[Tuple[str, Path]] = []
    for ds_path in selected_dataset_paths:
        for algo in effective_algos:
            tasks.append((algo, ds_path))

    total_tasks = len(tasks)
    pending_tasks = [t for t in tasks if (t[0], t[1].stem) not in completed_keys]
    skipped_count = total_tasks - len(pending_tasks)

    print(f"• Total planned tasks: {total_tasks} | Already done: {skipped_count} | Pending: {len(pending_tasks)}")
    print(f"• Live metrics file: {metrics_file}")
    print(f"• Summary output:    {summary_file}")
    print("=" * 70)

    # 8. Main execution loop with Progress Bar
    with tqdm(total=total_tasks, initial=skipped_count, desc="Augmentation Progress", unit="task") as pbar:
        for algo, ds_path in tasks:
            ds_name = ds_path.stem
            task_key = (algo, ds_name)

            if args.resume and task_key in completed_keys:
                pbar.set_postfix_str(f"[SKIP] {algo} on {ds_name}")
                continue

            pbar.set_postfix_str(f"Running {algo} on {ds_name}...")

            # Output file path for this algorithm and dataset
            algo_out_dir = augmented_dir / algo
            algo_out_dir.mkdir(parents=True, exist_ok=True)
            output_csv = algo_out_dir / f"{ds_name}_augmented.csv"

            # Resolve Denial Constraints for Kamino
            dc_path = None
            kam_strat = "toggled"
            if algo.startswith("kamino"):
                dc_path = resolve_dc_file(ds_name, args.dc_file, args.dc_dir)
                if not dc_path or not dc_path.exists():
                    err_msg = (
                        f"[KAMINO BLOCKED] Kamino cannot be executed without a specified Denial Constraints file! "
                        f"No valid DC file was specified or found for dataset '{ds_name}'."
                    )
                    tqdm.write(f"\n❌ {err_msg}")
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "algorithm": algo,
                        "dataset": ds_name,
                        "status": "FAILED",
                        "total_time_s": 0.0,
                        "train_time_s": 0.0,
                        "gen_time_s": 0.0,
                        "train_peak_mem_mb": 0.0,
                        "gen_peak_mem_mb": 0.0,
                        "total_mem_peak_mb": 0.0,
                        "rss_delta_mb": 0.0,
                        "train_gpu_peak_mb": 0.0,
                        "gen_gpu_peak_mb": 0.0,
                        "n_orig": 0,
                        "n_augmented": 0,
                        "output_path": "",
                        "method_used": "",
                        "error_message": err_msg,
                    }
                    current_records.append(record)
                    pd.DataFrame(current_records).to_csv(metrics_file, index=False)
                    pbar.update(1)
                    continue

                kam_strat = algo.split("kamino-")[1] if algo.startswith("kamino-") else args.kamino_weights

            # Execute algorithm
            result_dict = execute_augmentation(
                algo=algo,
                csv_path=ds_path,
                output_path=output_csv,
                use_gpu=use_gpu,
                device=device,
                n_samples=args.n_samples,
                epochs=args.epochs,
                dc_file=str(dc_path) if dc_path else None,
                kamino_weights=kam_strat,
                allow_fallback=args.allow_fallback,
            )

            # Record entry
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "algorithm": algo,
                "dataset": ds_name,
                "status": result_dict.get("status", "FAILED"),
                "total_time_s": result_dict.get("total_time_s", 0.0),
                "train_time_s": result_dict.get("train_time_s", 0.0),
                "gen_time_s": result_dict.get("gen_time_s", 0.0),
                "train_peak_mem_mb": result_dict.get("train_peak_mem_mb", 0.0),
                "gen_peak_mem_mb": result_dict.get("gen_peak_mem_mb", 0.0),
                "total_mem_peak_mb": result_dict.get("total_mem_peak_mb", 0.0),
                "rss_delta_mb": result_dict.get("rss_delta_mb", 0.0),
                "train_gpu_peak_mb": result_dict.get("train_gpu_peak_mb", 0.0),
                "gen_gpu_peak_mb": result_dict.get("gen_gpu_peak_mb", 0.0),
                "n_orig": result_dict.get("n_orig", 0),
                "n_augmented": result_dict.get("n_augmented", 0),
                "output_path": str(output_csv),
                "method_used": result_dict.get("method_used", ""),
                "error_message": result_dict.get("error_message", ""),
            }

            current_records.append(record)

            # Mark completed if successful
            if record["status"] == "SUCCESS":
                completed_keys.add(task_key)
            elif record["status"] == "FALLBACK":
                tqdm.write(f"~ FALLBACK {algo} on {ds_name}: eseguito '{record['method_used']}' - {record['error_message']}")
            else:
                tqdm.write(f"x FAILED {algo} on {ds_name}: {str(record['error_message'])[:300]}")

            # Incremental persistence: update CSV after EVERY run
            pd.DataFrame(current_records).to_csv(metrics_file, index=False)
            save_checkpoint(
                completed_keys=[[k[0], k[1]] for k in completed_keys],
                last_metrics_file=str(metrics_file),
                results_dir=results_dir,
            )

            status_symbol = {"SUCCESS": "✓", "FALLBACK": "~"}.get(record["status"], "✗")
            pbar.set_postfix_str(f"{status_symbol} {algo} on {ds_name} ({record['total_time_s']:.1f}s)")
            pbar.update(1)

    # 9. Final Summary Statistics Calculation
    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPLETE - CALCULATING SUMMARY STATISTICS...")
    print("=" * 70)

    all_results_df = pd.DataFrame(current_records)
    summary_df = compute_and_save_summary(all_results_df, summary_file)

    if not summary_df.empty:
        # Display key summary columns cleanly to console
        display_cols = [c for c in summary_df.columns if any(
            stat in c for stat in ["total_time_s_mean", "train_time_s_mean", "gen_time_s_mean", "total_mem_peak_mb_mean"]
        )]
        print("\nSummary Results (Mean values per algorithm):")
        print(summary_df[["algorithm"] + display_cols].to_string(index=False))

    print(f"\n📁 Runtime detailed metrics saved to: {metrics_file}")
    print(f"📁 Statistical summary saved to:      {summary_file}")
    print(f"📁 Augmented datasets stored in:      {augmented_dir}/<algorithm>/")
    print("=" * 70)


if __name__ == "__main__":
    main()
