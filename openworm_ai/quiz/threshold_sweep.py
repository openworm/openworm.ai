"""
Threshold Sweep — Sensitivity Analysis for Hybrid RAG
======================================================
Re-computes hybrid accuracy at different similarity thresholds from
existing detailed evaluation results (the *_detailed.json files produced
by quiz_full_rag_all.py or quiz_simple_rag_all.py).

Outputs:
  - JSON with per-threshold, per-LLM, per-category accuracy
  - Matplotlib sensitivity curve (PNG) if matplotlib is available

Usage:
    python -m openworm_ai.quiz.threshold_sweep \
        --input openworm_ai/quiz/scores/rag_full/eval_*_detailed.json

    # Custom threshold range:
    python -m openworm_ai.quiz.threshold_sweep \
        --input results.json --min 0.3 --max 0.95 --step 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Default sweep range
DEFAULT_MIN = 0.3
DEFAULT_MAX = 0.95
DEFAULT_STEP = 0.05


def load_detailed_results(input_paths: List[str]) -> List[Dict]:
    """Load all LLM x category results from one or more detailed JSON files."""
    all_results = []
    for path_str in input_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"! File not found: {path} — skipping")
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            results = data.get("Results", [])
            # Only keep results that have per-question detail
            for r in results:
                if "Question Details" in r and r["Question Details"]:
                    all_results.append(r)
                else:
                    print(
                        f"  ! {r.get('LLM', '?')} / {r.get('Quiz Category', '?')} "
                        f"— no Question Details, skipping"
                    )
        except Exception as e:
            print(f"! Error reading {path}: {e}")
    return all_results


def sweep_threshold(
    results: List[Dict],
    thresholds: List[float],
) -> Dict[str, Any]:
    """
    For each threshold, recompute hybrid accuracy from per-question data.

    Returns a structure:
    {
        "thresholds": [0.3, 0.35, ...],
        "by_llm_category": {
            "<LLM>|<Category>": {
                "pretrained_acc": float,
                "rag_acc": float,
                "hybrid_accs": [float, ...],  # one per threshold
                "rag_usage_pcts": [float, ...],  # % of questions using RAG
            }
        },
        "aggregated": {
            "mean_hybrid_accs": [float, ...],
            "mean_rag_usage_pcts": [float, ...],
        }
    }
    """
    by_key: Dict[str, Dict] = {}

    for result in results:
        llm = result.get("LLM", "unknown")
        cat = result.get("Quiz Category", "unknown")
        details = result.get("Question Details", [])
        n = len(details)
        if n == 0:
            continue

        key = f"{llm}|{cat}"
        pretrained_correct = sum(1 for d in details if d.get("pretrained_correct"))
        rag_correct = sum(1 for d in details if d.get("rag_correct"))

        hybrid_accs = []
        rag_usage_pcts = []

        for threshold in thresholds:
            hybrid_correct_count = 0
            rag_used_count = 0

            for d in details:
                score = d.get("best_similarity_score", 0.0)
                rag_avail = d.get("rag_available", False)

                if rag_avail and score >= threshold:
                    # Use RAG answer
                    rag_used_count += 1
                    if d.get("rag_correct", False):
                        hybrid_correct_count += 1
                else:
                    # Fall back to pretrained
                    if d.get("pretrained_correct", False):
                        hybrid_correct_count += 1

            hybrid_accs.append(round(100 * hybrid_correct_count / n, 2))
            rag_usage_pcts.append(round(100 * rag_used_count / n, 2))

        by_key[key] = {
            "llm": llm,
            "category": cat,
            "total_questions": n,
            "pretrained_acc": round(100 * pretrained_correct / n, 2),
            "rag_acc": round(100 * rag_correct / n, 2),
            "hybrid_accs": hybrid_accs,
            "rag_usage_pcts": rag_usage_pcts,
        }

    # Aggregate: mean across all LLM x category combos
    num_combos = len(by_key)
    num_thresholds = len(thresholds)

    if num_combos > 0:
        mean_hybrid = [
            round(sum(v["hybrid_accs"][i] for v in by_key.values()) / num_combos, 2)
            for i in range(num_thresholds)
        ]
        mean_rag_usage = [
            round(sum(v["rag_usage_pcts"][i] for v in by_key.values()) / num_combos, 2)
            for i in range(num_thresholds)
        ]
    else:
        mean_hybrid = [0.0] * num_thresholds
        mean_rag_usage = [0.0] * num_thresholds

    return {
        "thresholds": [round(t, 3) for t in thresholds],
        "by_llm_category": by_key,
        "aggregated": {
            "mean_hybrid_accs": mean_hybrid,
            "mean_rag_usage_pcts": mean_rag_usage,
        },
    }


def generate_plot(
    sweep_data: Dict,
    output_path: str,
    title: str = "Hybrid Accuracy vs RAG Similarity Threshold",
):
    """Generate matplotlib sensitivity curve (optional — no-op if matplotlib unavailable)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot generation")
        return

    thresholds = sweep_data["thresholds"]
    agg = sweep_data["aggregated"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot individual LLM lines (thin, transparent)
    for key, data in sweep_data["by_llm_category"].items():
        data["llm"].replace("huggingface:", "hf:").split("/")[-1][:25]
        data["category"][:12]
        ax1.plot(
            thresholds,
            data["hybrid_accs"],
            alpha=0.25,
            linewidth=0.8,
            label=None,
        )
        # Pretrained baseline as horizontal dotted line
        ax1.axhline(
            y=data["pretrained_acc"],
            alpha=0.05,
            linestyle=":",
            color="gray",
        )

    # Aggregated mean (bold)
    ax1.plot(
        thresholds,
        agg["mean_hybrid_accs"],
        color="blue",
        linewidth=2.5,
        label="Mean Hybrid Accuracy",
        zorder=10,
    )

    ax1.set_xlabel("RAG Similarity Threshold", fontsize=12)
    ax1.set_ylabel("Hybrid Accuracy (%)", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Secondary axis: RAG usage %
    ax2 = ax1.twinx()
    ax2.plot(
        thresholds,
        agg["mean_rag_usage_pcts"],
        color="red",
        linewidth=2,
        linestyle="--",
        label="Mean RAG Usage %",
        zorder=10,
    )
    ax2.set_ylabel("RAG Usage (%)", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Find optimal threshold
    best_idx = max(range(len(thresholds)), key=lambda i: agg["mean_hybrid_accs"][i])
    best_threshold = thresholds[best_idx]
    best_acc = agg["mean_hybrid_accs"][best_idx]
    ax1.axvline(
        x=best_threshold,
        color="green",
        linestyle="-.",
        alpha=0.7,
        label=f"Optimal: {best_threshold:.2f} ({best_acc:.1f}%)",
    )

    ax1.set_title(title, fontsize=14)
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Plot saved: {output_path}")
    plt.close()


def print_sweep_summary(sweep_data: Dict):
    """Print a compact threshold sweep summary table."""
    thresholds = sweep_data["thresholds"]
    agg = sweep_data["aggregated"]

    print(f"\n{'=' * 65}")
    print("  THRESHOLD SWEEP — Aggregated Results")
    print(f"{'=' * 65}")
    print(f"  {'Threshold':>10}  {'Hybrid Acc%':>12}  {'RAG Usage%':>12}  {'Note':>10}")
    print(f"  {'─' * 55}")

    best_idx = max(range(len(thresholds)), key=lambda i: agg["mean_hybrid_accs"][i])

    for i, t in enumerate(thresholds):
        note = " <-- best" if i == best_idx else ""
        print(
            f"  {t:>10.2f}  {agg['mean_hybrid_accs'][i]:>11.1f}%  "
            f"{agg['mean_rag_usage_pcts'][i]:>11.1f}%  {note}"
        )

    # Also show pretrained-only baseline
    all_pretrained = [
        v["pretrained_acc"] for v in sweep_data["by_llm_category"].values()
    ]
    if all_pretrained:
        mean_pre = sum(all_pretrained) / len(all_pretrained)
        print(f"  {'─' * 55}")
        print(f"  {'pretrained':>10}  {mean_pre:>11.1f}%  {'0.0':>11}%   baseline")

    print(f"{'=' * 65}")
    print(
        f"  Optimal threshold: {thresholds[best_idx]:.2f} "
        f"({agg['mean_hybrid_accs'][best_idx]:.1f}% hybrid accuracy)"
    )
    print(f"  Evaluated: {len(sweep_data['by_llm_category'])} LLM x category combos")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Threshold sweep — sensitivity analysis for hybrid RAG"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to detailed evaluation JSON files (*_detailed.json)",
    )
    parser.add_argument(
        "--min", type=float, default=DEFAULT_MIN, help="Minimum threshold"
    )
    parser.add_argument(
        "--max", type=float, default=DEFAULT_MAX, help="Maximum threshold"
    )
    parser.add_argument(
        "--step", type=float, default=DEFAULT_STEP, help="Threshold step size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="openworm_ai/quiz/scores/threshold_sweep",
        help="Output directory for results and plots",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build threshold list
    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 3))
        t += args.step

    print(
        f"\nThreshold sweep: {thresholds[0]:.2f} → {thresholds[-1]:.2f} "
        f"(step={args.step}, {len(thresholds)} points)"
    )

    # Load results
    results = load_detailed_results(args.input)
    if not results:
        print("ERROR: No valid detailed results found. Run evaluation first.")
        sys.exit(1)

    print(f"Loaded {len(results)} LLM x category result sets\n")

    # Sweep
    sweep_data = sweep_threshold(results, thresholds)

    # Print summary
    print_sweep_summary(sweep_data)

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "threshold_sweep.json"
    with open(json_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"  Results saved: {json_path}")

    # Generate plot
    if not args.no_plot:
        plot_path = str(output_dir / "threshold_sensitivity.png")
        generate_plot(sweep_data, plot_path)


if __name__ == "__main__":
    main()
