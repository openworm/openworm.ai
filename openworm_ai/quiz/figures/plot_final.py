"""
Final dissertation & presentation figures — clean, readable, no clutter.
Filters broken models, deduplicates, produces 7 key figures + scatter plots.
"""

import json
import glob
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
BROKEN_MODELS = {
    "huggingface:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "huggingface:microsoft/Phi-3-mini-4k-instruct",
    "huggingface:microsoft/Phi-3-medium-4k-instruct",
    "huggingface:Qwen/Qwen3-8B",
    "claude-3-7-sonnet-20250219",
}

MODEL_PARAMS_B = {
    "meta-llama/Llama-3.2-1B-Instruct": 1,
    "Qwen/Qwen2.5-7B-Instruct": 7,
    "mistralai/Mistral-7B-Instruct-v0.2": 7,
    "meta-llama/Llama-3.1-8B-Instruct": 8,
    "google/gemma-2-9b-it": 9,
    "Qwen/Qwen2.5-14B-Instruct": 14,
    "google/gemma-3-27b-it": 27,
    "Qwen/Qwen2.5-32B-Instruct": 32,
    "CohereLabs/aya-expanse-32b": 32,
    "Qwen/Qwen3-32B": 32,
    "meta-llama/Llama-3.3-70B-Instruct": 70,
    "Qwen/Qwen2.5-72B-Instruct": 72,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 141,
    "deepseek-ai/DeepSeek-R1": 671,
}

# Nice short names and company colors
SHORT_NAMES = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-1B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-8B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "google/gemma-3-27b-it": "Gemma-3-27B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "CohereLabs/aya-expanse-32b": "Aya-32B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-70B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
    "deepseek-ai/DeepSeek-R1": "DeepSeek-R1",
}

COMPANY_COLORS = {
    "Llama-1B": "#00A8E8", "Llama-8B": "#00A8E8", "Llama-70B": "#00A8E8",
    "Qwen2.5-7B": "#FF6A00", "Qwen2.5-14B": "#FF6A00", "Qwen2.5-32B": "#FF6A00",
    "Qwen2.5-72B": "#FF6A00", "Qwen3-32B": "#FF6A00",
    "Gemma-2-9B": "#EA4335", "Gemma-3-27B": "#EA4335",
    "Mistral-7B": "#F2A900", "Mixtral-8x22B": "#F2A900",
    "Aya-32B": "#39594D",
    "DeepSeek-R1": "#7F39FB",
}

COMPANY_LABELS = {
    "#00A8E8": "Meta (Llama)", "#FF6A00": "Alibaba (Qwen)",
    "#EA4335": "Google (Gemma)", "#F2A900": "Mistral AI",
    "#39594D": "Cohere (Aya)", "#7F39FB": "DeepSeek",
}

# Publication-quality style (Nature/Science aesthetic)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

MODE_COLORS = {
    "Pretrained": "#4878CF",
    "RAG Only": "#66c2a5",
    "Hybrid": "#fc8d62",
    "RAG-Informed": "#8da0cb",
}


def _short(llm):
    m = llm.replace("huggingface:", "")
    return SHORT_NAMES.get(m, m.split("/")[-1])


def _params(llm):
    m = llm.replace("huggingface:", "")
    return MODEL_PARAMS_B.get(m)


def load_and_clean(score_dir):
    """Load all eval JSONs, filter broken, deduplicate (keep best pretrained)."""
    all_results = []
    for f in sorted(glob.glob(f"{score_dir}/eval_*[!d].json")):
        if "detailed" in f:
            continue
        data = json.load(open(f))
        for r in data.get("Results", []):
            if r["LLM"] not in BROKEN_MODELS:
                all_results.append(r)

    # Deduplicate: keep entry with highest pretrained accuracy per LLM × category
    best = {}
    for r in all_results:
        key = (r["LLM"], r.get("Quiz Category", ""))
        pre = r.get("Pretrained Accuracy (%)", 0)
        if key not in best or pre > best[key].get("Pretrained Accuracy (%)", 0):
            best[key] = r

    return list(best.values())


def load_detailed(score_dir):
    """Load detailed results for calibration."""
    all_results = []
    for f in sorted(glob.glob(f"{score_dir}/eval_*_detailed.json")):
        data = json.load(open(f))
        for r in data.get("Results", []):
            if r["LLM"] not in BROKEN_MODELS and r.get("Question Details"):
                all_results.append(r)
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: 4-mode bar chart for a single category
# ═══════════════════════════════════════════════════════════════════════

def plot_4mode_bars(results, output_dir, category):
    """Clean grouped bar chart for one category."""
    subset = [r for r in results if r.get("Quiz Category") == category]
    if not subset:
        return

    # Sort by informed accuracy descending
    subset.sort(key=lambda r: r.get("Informed Accuracy (%)", 0), reverse=True)

    llms = [_short(r["LLM"]) for r in subset]
    pre = [r.get("Pretrained Accuracy (%)", 0) for r in subset]
    rag = [r.get("RAG Accuracy Overall (%)", 0) for r in subset]
    hyb = [r.get("Hybrid Accuracy (%)", 0) for r in subset]
    inf = [r.get("Informed Accuracy (%)", 0) for r in subset]

    x = np.arange(len(llms))
    width = 0.19

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - 1.5*width, pre, width, label="Pretrained", color=MODE_COLORS["Pretrained"], alpha=0.85)
    ax.bar(x - 0.5*width, rag, width, label="RAG Only", color=MODE_COLORS["RAG Only"], alpha=0.85)
    ax.bar(x + 0.5*width, hyb, width, label="Hybrid", color=MODE_COLORS["Hybrid"], alpha=0.85)
    ax.bar(x + 1.5*width, inf, width, label="RAG-Informed", color=MODE_COLORS["RAG-Informed"], alpha=0.85)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"4-Mode RAG Evaluation — {category}")
    ax.set_xticks(x)
    ax.set_xticklabels(llms, rotation=40, ha="right")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Add delta annotations on top of informed bars for corpus
    if "Corpus" in category:
        for i, (p, inf_val) in enumerate(zip(pre, inf)):
            delta = inf_val - p
            if delta > 0:
                ax.text(x[i] + 1.5*width, inf_val + 1, f"+{delta:.0f}",
                       ha="center", va="bottom", fontsize=7, fontweight="bold", color="#2E7D32")

    plt.tight_layout()
    safe = category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = output_dir / f"bars_{safe}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: RAG delta — corpus-only horizontal bars
# ═══════════════════════════════════════════════════════════════════════

def plot_rag_delta_corpus(results, output_dir):
    """Horizontal bar: informed - pretrained delta on corpus questions only."""
    subset = [r for r in results if r.get("Quiz Category") == "C. elegans (Corpus)"]
    if not subset:
        return

    items = [(r, r.get("Informed vs Pretrained Delta (pp)", 0)) for r in subset]
    items.sort(key=lambda x: x[1])

    llms = [_short(r["LLM"]) for r, _ in items]
    deltas = [d for _, d in items]
    pre_scores = [r.get("Pretrained Accuracy (%)", 0) for r, _ in items]
    inf_scores = [r.get("Informed Accuracy (%)", 0) for r, _ in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#D65F5F" if d < 0 else "#4CAF50" for d in deltas]
    bars = ax.barh(llms, deltas, color=colors, alpha=0.85, height=0.7)
    ax.axvline(x=0, color="black", linewidth=0.8)

    for i, (d, pre, inf_val) in enumerate(zip(deltas, pre_scores, inf_scores)):
        label = f"+{d:.0f}pp ({pre:.0f}%→{inf_val:.0f}%)"
        ax.text(d + (0.5 if d >= 0 else -0.5), i, label,
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, fontweight="bold")

    ax.set_xlabel("Accuracy Change (pp): RAG-Informed vs Pretrained")
    ax.set_title("RAG Impact on C. elegans Corpus Questions")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "rag_delta_corpus.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Scaling law scatter — accuracy vs params
# ═══════════════════════════════════════════════════════════════════════

def plot_scaling_scatter(results, output_dir, category=None):
    """Scatter: accuracy vs model size, colored by mode, with trend lines."""
    if category:
        data = [r for r in results if r.get("Quiz Category") == category]
        title_suffix = f" — {category}"
    else:
        # Average across categories
        by_llm = defaultdict(lambda: {"pre": [], "rag": [], "hyb": [], "inf": []})
        for r in results:
            llm = r["LLM"]
            by_llm[llm]["pre"].append(r.get("Pretrained Accuracy (%)", 0))
            by_llm[llm]["rag"].append(r.get("RAG Accuracy Overall (%)", 0))
            by_llm[llm]["hyb"].append(r.get("Hybrid Accuracy (%)", 0))
            by_llm[llm]["inf"].append(r.get("Informed Accuracy (%)", 0))
        data = [{"LLM": llm, "Pretrained Accuracy (%)": np.mean(d["pre"]),
                 "RAG Accuracy Overall (%)": np.mean(d["rag"]),
                 "Hybrid Accuracy (%)": np.mean(d["hyb"]),
                 "Informed Accuracy (%)": np.mean(d["inf"])}
                for llm, d in by_llm.items()]
        title_suffix = " (All Categories)"

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = [
        ("Pretrained Accuracy (%)", "Pretrained", "#4878CF", "o"),
        ("Informed Accuracy (%)", "RAG-Informed", "#FFB347", "^"),
    ]

    for key, label, color, marker in modes:
        xs, ys, names = [], [], []
        for r in data:
            p = _params(r["LLM"])
            if p is None:
                continue
            xs.append(p)
            ys.append(r.get(key, 0))
            names.append(_short(r["LLM"]))

        ax.scatter(xs, ys, c=color, marker=marker, s=80, alpha=0.85,
                  label=label, zorder=5, edgecolors="white", linewidth=0.5)

        # Trend line
        if len(xs) >= 3:
            log_x = np.log10(xs)
            z = np.polyfit(log_x, ys, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(log_x), max(log_x), 100)
            ax.plot(10**x_trend, p(x_trend), color=color, alpha=0.4,
                   linestyle="--", linewidth=2)

        # Label points
        for xi, yi, name in zip(xs, ys, names):
            ax.annotate(name, (xi, yi), textcoords="offset points",
                       xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (Billions)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Scaling Law: Pretrained vs RAG-Informed{title_suffix}")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    safe = category.lower().replace(" ", "_").replace("(", "").replace(")", "") if category else "all"
    path = output_dir / f"scaling_scatter_{safe}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Domain gap — accuracy vs category (line plot, clean)
# ═══════════════════════════════════════════════════════════════════════

def plot_domain_gap(results, output_dir):
    """Line plot: each model's accuracy dropping from General → C. elegans."""
    cat_order = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]

    # Build per-model, per-category pretrained scores
    by_llm = defaultdict(dict)
    for r in results:
        cat = r.get("Quiz Category", "")
        if cat in cat_order:
            by_llm[r["LLM"]][cat] = r.get("Pretrained Accuracy (%)", 0)

    # Only include models with all 4 categories
    complete = {llm: cats for llm, cats in by_llm.items()
                if all(c in cats for c in cat_order)}

    if not complete:
        print("  Not enough complete models for domain gap plot")
        return

    # Sort by General Knowledge score (top performers first)
    sorted_llms = sorted(complete.keys(),
                        key=lambda l: complete[l]["General Knowledge"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for llm in sorted_llms:
        name = _short(llm)
        color = COMPANY_COLORS.get(name, "#888888")
        scores = [complete[llm][c] for c in cat_order]
        params = _params(llm)
        lw = 2.5 if params and params >= 32 else 1.5
        alpha = 0.9 if params and params >= 32 else 0.6
        ax.plot(cat_order, scores, "o-", label=f"{name} ({params}B)" if params else name,
               color=color, linewidth=lw, alpha=alpha, markersize=6)

    ax.set_ylabel("Pretrained Accuracy (%)")
    ax.set_title("The Domain Gap: LLM Performance Across Knowledge Domains")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    path = output_dir / "domain_gap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Error breakdown (clean — only working models)
# ═══════════════════════════════════════════════════════════════════════

def plot_error_breakdown(results, output_dir):
    """Side-by-side stacked bar: pretrained vs informed error types."""
    error_types = ["correct", "wrong_answer", "format_error", "parse_failure", "no_response"]
    colors = {"correct": "#4CAF50", "wrong_answer": "#E53935", "format_error": "#FFB347",
              "parse_failure": "#9B59B6", "no_response": "#95A5A6"}

    # Aggregate per LLM across all categories
    llm_errors = defaultdict(lambda: {"pretrained": defaultdict(int), "informed": defaultdict(int)})
    for r in results:
        name = _short(r["LLM"])
        for etype in error_types:
            llm_errors[name]["pretrained"][etype] += r.get("Pretrained Errors", {}).get(etype, 0)
            llm_errors[name]["informed"][etype] += r.get("Informed Errors", {}).get(etype, 0)

    if not llm_errors:
        return

    # Sort by total correct (pretrained)
    sorted_llms = sorted(llm_errors.keys(),
                        key=lambda l: llm_errors[l]["pretrained"]["correct"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, mode, title in [(axes[0], "pretrained", "Pretrained"),
                             (axes[1], "informed", "RAG-Informed")]:
        bottom = np.zeros(len(sorted_llms))
        for etype in error_types:
            vals = [llm_errors[llm][mode][etype] for llm in sorted_llms]
            label = etype.replace("_", " ").title()
            ax.barh(sorted_llms, vals, left=bottom, label=label,
                   color=colors[etype], alpha=0.85)
            bottom += np.array(vals, dtype=float)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Question Count")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Error Type Breakdown: Pretrained vs RAG-Informed", fontsize=14, y=1.01)
    plt.tight_layout()
    path = output_dir / "error_breakdown.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Calibration curve
# ═══════════════════════════════════════════════════════════════════════

def plot_calibration(detailed_results, output_dir):
    """Similarity score vs RAG accuracy — proves retrieval quality matters."""
    all_questions = []
    for r in detailed_results:
        for q in r.get("Question Details", []):
            if q.get("rag_available"):
                all_questions.append(q)

    if len(all_questions) < 20:
        print("  Not enough RAG questions for calibration — skipping")
        return

    bins = np.arange(0.5, 1.01, 0.05)
    bin_accs, bin_counts, bin_centers = [], [], []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [q for q in all_questions if lo <= q.get("best_similarity_score", 0) < hi]
        if len(in_bin) >= 10:
            acc = sum(1 for q in in_bin if q.get("rag_correct")) / len(in_bin) * 100
            bin_accs.append(acc)
            bin_counts.append(len(in_bin))
            bin_centers.append((lo + hi) / 2)

    if len(bin_centers) < 3:
        print("  Not enough bins for calibration — skipping")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(bin_centers, bin_accs, "o-", color="#4878CF", linewidth=2.5,
            markersize=10, label="RAG Accuracy", zorder=5)

    ax1.set_xlabel("Retrieval Similarity Score")
    ax1.set_ylabel("RAG Accuracy (%)", color="#4878CF")
    ax1.set_title("Retrieval Quality vs RAG Accuracy")
    ax1.set_ylim(40, 100)

    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=0.04, alpha=0.2, color="gray", label="N questions")
    ax2.set_ylabel("Questions in Bin", color="gray")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "calibration.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Scatter — pretrained vs informed accuracy per model
# ═══════════════════════════════════════════════════════════════════════

def plot_pretrained_vs_informed_scatter(results, output_dir, category=None):
    """Scatter: x = pretrained, y = informed, diagonal = no change. Points above = RAG helps."""
    if category:
        data = [r for r in results if r.get("Quiz Category") == category]
        title = f"Pretrained vs RAG-Informed — {category}"
    else:
        data = results
        title = "Pretrained vs RAG-Informed (All Categories)"

    fig, ax = plt.subplots(figsize=(8, 8))

    xs, ys, names = [], [], []
    for r in data:
        pre = r.get("Pretrained Accuracy (%)", 0)
        inf = r.get("Informed Accuracy (%)", 0)
        if pre == 0 and inf == 0:
            continue
        xs.append(pre)
        ys.append(inf)
        name = _short(r["LLM"])
        names.append(name)

    # Color by company
    colors = [COMPANY_COLORS.get(n, "#888888") for n in names]
    ax.scatter(xs, ys, c=colors, s=80, alpha=0.85, zorder=5, edgecolors="white", linewidth=0.5)

    # Labels
    for xi, yi, name in zip(xs, ys, names):
        ax.annotate(name, (xi, yi), textcoords="offset points",
                   xytext=(5, 4), fontsize=7, alpha=0.8)

    # Diagonal
    lims = [0, 105]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No change")
    ax.fill_between(lims, lims, [105, 105], alpha=0.05, color="green", label="RAG helps")
    ax.fill_between(lims, [0, 0], lims, alpha=0.05, color="red", label="RAG hurts")

    ax.set_xlabel("Pretrained Accuracy (%)")
    ax.set_ylabel("RAG-Informed Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = category.lower().replace(" ", "_").replace("(", "").replace(")", "") if category else "all"
    path = output_dir / f"scatter_pre_vs_inf_{safe}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    score_dir = "openworm_ai/quiz/scores/rag_full_logprob"
    output_dir = Path("openworm_ai/quiz/figures/final")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning results...")
    results = load_and_clean(score_dir)
    print(f"  {len(results)} clean LLM × category entries")
    print(f"  Models: {len(set(r['LLM'] for r in results))}")
    print(f"  Categories: {sorted(set(r.get('Quiz Category','') for r in results))}")

    print(f"\nGenerating figures in {output_dir}/\n")

    # 1. 4-mode bars — key categories
    plot_4mode_bars(results, output_dir, "C. elegans (Corpus)")
    plot_4mode_bars(results, output_dir, "C. elegans")

    # 2. RAG delta — corpus only
    plot_rag_delta_corpus(results, output_dir)

    # 3. Scaling scatter — corpus and all
    plot_scaling_scatter(results, output_dir, "C. elegans (Corpus)")
    plot_scaling_scatter(results, output_dir)

    # 4. Domain gap
    plot_domain_gap(results, output_dir)

    # 5. Error breakdown
    plot_error_breakdown(results, output_dir)

    # 6. Calibration
    detailed = load_detailed(score_dir)
    if detailed:
        plot_calibration(detailed, output_dir)

    # 7. Scatter: pretrained vs informed
    plot_pretrained_vs_informed_scatter(results, output_dir, "C. elegans (Corpus)")
    plot_pretrained_vs_informed_scatter(results, output_dir, "C. elegans")
    plot_pretrained_vs_informed_scatter(results, output_dir)

    print(f"\nDone! All figures in {output_dir}/")


if __name__ == "__main__":
    main()
