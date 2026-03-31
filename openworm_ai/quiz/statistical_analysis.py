"""
Statistical Analysis of RAG Evaluation Results
================================================
Applies appropriate statistical tests to determine whether RAG
significantly improves LLM accuracy on C. elegans questions.

Tests selected based on data characteristics:
  1. McNemar's test — paired binary outcomes (same question, pre/post RAG)
  2. Wilcoxon signed-rank — paired model accuracies (non-parametric, n=14)
  3. Spearman correlation — model size vs RAG benefit
  4. Chi-squared — error type distribution changes
  5. Cohen's d — effect size of RAG improvement
  6. Bootstrap 95% CI — confidence intervals on accuracy differences
  7. Spearman correlation — retrieval similarity vs accuracy
"""

import json
import glob
import numpy as np
from collections import defaultdict
from scipy import stats

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


def short_name(llm):
    return llm.replace("huggingface:", "").split("/")[-1].replace("-Instruct", "").replace("-v0.2", "").replace("-v0.1", "")


def load_data(score_dir):
    """Load summary + detailed results, filter broken, deduplicate."""
    # Summary data
    summary = []
    for f in sorted(glob.glob(f"{score_dir}/eval_*.json")):
        if "detailed" in f:
            continue
        data = json.load(open(f))
        for r in data.get("Results", []):
            if r["LLM"] not in BROKEN_MODELS:
                summary.append(r)

    # Deduplicate
    best = {}
    for r in summary:
        key = (r["LLM"], r.get("Quiz Category", ""))
        pre = r.get("Pretrained Accuracy (%)", 0)
        if key not in best or pre > best[key].get("Pretrained Accuracy (%)", 0):
            best[key] = r
    summary = list(best.values())

    # Detailed (per-question)
    detailed = []
    for f in sorted(glob.glob(f"{score_dir}/eval_*_detailed.json")):
        data = json.load(open(f))
        for r in data.get("Results", []):
            if r["LLM"] not in BROKEN_MODELS and r.get("Question Details"):
                detailed.append(r)

    return summary, detailed


def mcnemar_test(detailed, category):
    """
    McNemar's test: are the marginal probabilities of correct different
    between pretrained and informed modes?

    Contingency table:
                    Informed Correct  Informed Wrong
    Pre Correct         a                 b
    Pre Wrong           c                 d

    We test whether b != c (discordant pairs).
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: McNemar's Test — {category}")
    print(f"{'='*60}")
    print(f"H0: RAG-Informed does not change accuracy (b = c)")
    print(f"Data: Paired binary outcomes on same questions\n")

    a, b, c, d = 0, 0, 0, 0
    n_questions = 0

    for r in detailed:
        if r.get("Quiz Category") != category:
            continue
        for q in r.get("Question Details", []):
            pre_correct = q.get("pretrained_correct", False)
            inf_correct = q.get("informed_correct", False)
            if pre_correct and inf_correct:
                a += 1
            elif pre_correct and not inf_correct:
                b += 1  # RAG made it worse
            elif not pre_correct and inf_correct:
                c += 1  # RAG fixed it
            else:
                d += 1
            n_questions += 1

    print(f"Contingency table (n={n_questions} question×model pairs):")
    print(f"                    Informed Correct  Informed Wrong")
    print(f"  Pre Correct           {a:>5}            {b:>5}")
    print(f"  Pre Wrong             {c:>5}            {d:>5}")
    print(f"\n  Concordant: {a+d} ({(a+d)/n_questions*100:.1f}%)")
    print(f"  RAG fixed (c): {c} questions")
    print(f"  RAG broke (b): {b} questions")

    # McNemar's test (with continuity correction for small samples)
    if b + c > 0:
        # Exact binomial test (more appropriate than chi-squared approximation)
        # Under H0, b/(b+c) ~ 0.5
        n_discord = b + c
        result = stats.binomtest(c, n_discord, 0.5, alternative='two-sided')
        p_exact = result.pvalue

        # Also compute chi-squared version for reference
        chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) >= 25 else None
        p_chi2 = 1 - stats.chi2.cdf(chi2, df=1) if chi2 is not None else None

        print(f"\n  Discordant pairs: {n_discord}")
        print(f"  RAG fixed / (fixed + broke) = {c}/{n_discord} = {c/n_discord:.3f}")
        print(f"  Exact binomial p-value: {p_exact:.2e}")
        if p_chi2 is not None:
            print(f"  Chi-squared approx p-value: {p_chi2:.2e} (chi2={chi2:.2f}, df=1)")

        if p_exact < 0.001:
            print(f"\n  *** HIGHLY SIGNIFICANT (p < 0.001) — RAG significantly changes accuracy ***")
        elif p_exact < 0.05:
            print(f"\n  * SIGNIFICANT (p < 0.05) — RAG significantly changes accuracy *")
        else:
            print(f"\n  Not significant (p >= 0.05)")

        # Direction
        if c > b:
            print(f"  Direction: RAG IMPROVES accuracy (fixed {c} > broke {b})")
        elif b > c:
            print(f"  Direction: RAG HURTS accuracy (broke {b} > fixed {c})")

    return {"a": a, "b": b, "c": c, "d": d, "n": n_questions}


def wilcoxon_test(summary, category):
    """
    Wilcoxon signed-rank test: do model accuracies systematically differ
    between pretrained and informed modes?

    Non-parametric paired test (appropriate for n=14, non-normal).
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Wilcoxon Signed-Rank Test — {category}")
    print(f"{'='*60}")
    print(f"H0: Median difference in accuracy = 0")
    print(f"Data: Paired model accuracies (n = number of models)\n")

    pairs = []
    for r in summary:
        if r.get("Quiz Category") != category:
            continue
        pre = r.get("Pretrained Accuracy (%)", 0)
        inf = r.get("Informed Accuracy (%)", 0)
        if pre > 0 or inf > 0:  # Skip completely broken
            pairs.append((short_name(r["LLM"]), pre, inf, inf - pre))

    if len(pairs) < 5:
        print(f"  Insufficient data (n={len(pairs)})")
        return

    pairs.sort(key=lambda x: x[3], reverse=True)

    print(f"  {'Model':<25s} {'Pre':>6s} {'Inf':>6s} {'Delta':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*7}")
    for name, pre, inf, delta in pairs:
        sign = "+" if delta > 0 else ""
        print(f"  {name:<25s} {pre:6.1f} {inf:6.1f} {sign}{delta:6.1f}")

    diffs = [inf - pre for _, pre, inf, _ in pairs]
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    se_diff = sd_diff / np.sqrt(len(diffs))

    print(f"\n  n = {len(pairs)} models")
    print(f"  Mean difference: {mean_diff:+.2f}pp")
    print(f"  SD of differences: {sd_diff:.2f}")
    print(f"  SE: {se_diff:.2f}")

    # Wilcoxon signed-rank test
    diffs_arr = np.array(diffs)
    nonzero = diffs_arr[diffs_arr != 0]
    if len(nonzero) >= 5:
        stat, p = stats.wilcoxon(nonzero, alternative='two-sided')
        print(f"\n  Wilcoxon W = {stat:.1f}, p = {p:.4f}")
        if p < 0.05:
            print(f"  *** SIGNIFICANT (p < 0.05) ***")
        else:
            print(f"  Not significant")
    else:
        print(f"\n  Too few non-zero differences for Wilcoxon test")

    # Also report paired t-test for comparison (even though assumptions may not hold)
    t_stat, t_p = stats.ttest_rel([pre for _, pre, _, _ in pairs],
                                   [inf for _, _, inf, _ in pairs])
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {t_p:.4f} (for reference)")

    # Cohen's d (paired)
    if sd_diff > 0:
        d = mean_diff / sd_diff
        print(f"\n  Cohen's d = {d:.3f}", end="")
        if abs(d) < 0.2:
            print(" (negligible)")
        elif abs(d) < 0.5:
            print(" (small)")
        elif abs(d) < 0.8:
            print(" (medium)")
        else:
            print(" (large)")


def spearman_size_vs_rag(summary, category):
    """
    Spearman rank correlation: does model size correlate with RAG benefit?
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Spearman Correlation — Model Size vs RAG Benefit ({category})")
    print(f"{'='*60}")
    print(f"H0: No monotonic relationship between model size and RAG delta")
    print(f"Data: log(params) vs accuracy delta\n")

    points = []
    for r in summary:
        if r.get("Quiz Category") != category:
            continue
        llm = r["LLM"].replace("huggingface:", "")
        params = MODEL_PARAMS_B.get(llm)
        delta = r.get("Informed vs Pretrained Delta (pp)", 0)
        pre = r.get("Pretrained Accuracy (%)", 0)
        if params and pre > 0:
            points.append((short_name(r["LLM"]), params, delta))

    if len(points) < 5:
        print(f"  Insufficient data (n={len(points)})")
        return

    points.sort(key=lambda x: x[1])
    for name, params, delta in points:
        print(f"  {name:<25s} {params:>6.0f}B  delta={delta:+.1f}pp")

    sizes = [p[1] for p in points]
    deltas = [p[2] for p in points]

    rho, p = stats.spearmanr(np.log10(sizes), deltas)
    print(f"\n  Spearman rho = {rho:.3f}, p = {p:.4f}")
    print(f"  (negative rho = larger models benefit LESS from RAG)")

    if p < 0.05:
        print(f"  *** SIGNIFICANT correlation ***")
    else:
        print(f"  Not significant at p < 0.05")


def chi_squared_errors(summary, category):
    """
    Chi-squared test: does the distribution of error types differ
    between pretrained and informed modes?
    """
    print(f"\n{'='*60}")
    print(f"TEST 4: Chi-Squared Test — Error Distribution ({category})")
    print(f"{'='*60}")
    print(f"H0: Error type distribution is the same pre/post RAG\n")

    error_types = ["correct", "wrong_answer", "format_error", "parse_failure"]
    pre_counts = defaultdict(int)
    inf_counts = defaultdict(int)

    for r in summary:
        if r.get("Quiz Category") != category:
            continue
        for et in error_types:
            pre_counts[et] += r.get("Pretrained Errors", {}).get(et, 0)
            inf_counts[et] += r.get("Informed Errors", {}).get(et, 0)

    print(f"  {'Error Type':<20s} {'Pretrained':>12s} {'Informed':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for et in error_types:
        print(f"  {et:<20s} {pre_counts[et]:>12d} {inf_counts[et]:>12d}")

    # Build contingency table (only types with counts > 0)
    observed = []
    used_types = []
    for et in error_types:
        if pre_counts[et] + inf_counts[et] > 0:
            observed.append([pre_counts[et], inf_counts[et]])
            used_types.append(et)

    if len(observed) >= 2:
        observed = np.array(observed)
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        print(f"\n  Chi-squared = {chi2:.2f}, df = {dof}, p = {p:.4f}")
        if p < 0.05:
            print(f"  *** SIGNIFICANT — error distribution changes with RAG ***")
        else:
            print(f"  Not significant")


def bootstrap_ci(detailed, category, n_bootstrap=10000):
    """
    Bootstrap 95% confidence intervals on RAG accuracy improvement.
    """
    print(f"\n{'='*60}")
    print(f"TEST 5: Bootstrap 95% CI — RAG Improvement ({category})")
    print(f"{'='*60}")
    print(f"Method: Resample question-level outcomes {n_bootstrap} times\n")

    # Collect per-question paired outcomes
    pairs = []  # (pre_correct, inf_correct)
    for r in detailed:
        if r.get("Quiz Category") != category:
            continue
        for q in r.get("Question Details", []):
            pre = 1 if q.get("pretrained_correct", False) else 0
            inf = 1 if q.get("informed_correct", False) else 0
            pairs.append((pre, inf))

    if len(pairs) < 20:
        print(f"  Insufficient data (n={len(pairs)})")
        return

    pairs = np.array(pairs)
    pre_acc = pairs[:, 0].mean() * 100
    inf_acc = pairs[:, 1].mean() * 100
    observed_delta = inf_acc - pre_acc

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(pairs), size=len(pairs), replace=True)
        boot_pre = pairs[idx, 0].mean() * 100
        boot_inf = pairs[idx, 1].mean() * 100
        boot_deltas.append(boot_inf - boot_pre)

    boot_deltas = np.array(boot_deltas)
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)
    boot_se = np.std(boot_deltas)

    print(f"  Pooled pretrained accuracy: {pre_acc:.1f}%")
    print(f"  Pooled informed accuracy:   {inf_acc:.1f}%")
    print(f"  Observed delta:             {observed_delta:+.2f}pp")
    print(f"  Bootstrap SE:               {boot_se:.2f}")
    print(f"  95% CI:                     [{ci_lo:+.2f}, {ci_hi:+.2f}]pp")

    if ci_lo > 0:
        print(f"\n  *** CI excludes zero — RAG improvement is significant ***")
    elif ci_hi < 0:
        print(f"\n  *** CI excludes zero — RAG significantly hurts ***")
    else:
        print(f"\n  CI includes zero — effect not significant")


def similarity_correlation(detailed, category):
    """
    Spearman correlation: retrieval similarity score vs RAG accuracy.
    """
    print(f"\n{'='*60}")
    print(f"TEST 6: Spearman Correlation — Retrieval Quality vs Accuracy ({category})")
    print(f"{'='*60}")
    print(f"H0: No relationship between similarity score and RAG accuracy\n")

    scores = []
    correct = []
    for r in detailed:
        if r.get("Quiz Category") != category:
            continue
        for q in r.get("Question Details", []):
            sim = q.get("best_similarity_score", 0)
            if sim > 0:  # Only questions with retrieval
                scores.append(sim)
                correct.append(1 if q.get("informed_correct", False) else 0)

    if len(scores) < 20:
        print(f"  Insufficient data with retrieval scores (n={len(scores)})")
        return

    # Bin for readability
    bins = [(0.5, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 1.0)]
    print(f"  {'Similarity Bin':<18s} {'N':>5s} {'Accuracy':>10s}")
    print(f"  {'-'*18} {'-'*5} {'-'*10}")
    for lo, hi in bins:
        in_bin = [(s, c) for s, c in zip(scores, correct) if lo <= s < hi]
        if len(in_bin) >= 5:
            acc = sum(c for _, c in in_bin) / len(in_bin) * 100
            print(f"  [{lo:.2f}, {hi:.2f})       {len(in_bin):>5d} {acc:>9.1f}%")

    rho, p = stats.spearmanr(scores, correct)
    print(f"\n  Spearman rho = {rho:.3f}, p = {p:.2e}")
    print(f"  n = {len(scores)} question×model pairs with retrieval")

    if p < 0.001:
        print(f"  *** HIGHLY SIGNIFICANT — higher similarity = higher accuracy ***")
    elif p < 0.05:
        print(f"  * SIGNIFICANT *")


def descriptive_summary(summary):
    """Overall descriptive statistics."""
    print(f"\n{'='*60}")
    print(f"DESCRIPTIVE STATISTICS")
    print(f"{'='*60}\n")

    for cat in ["C. elegans (Corpus)", "C. elegans", "General Knowledge", "Science"]:
        subset = [r for r in summary if r.get("Quiz Category") == cat]
        if not subset:
            continue

        pre_accs = [r.get("Pretrained Accuracy (%)", 0) for r in subset if r.get("Pretrained Accuracy (%)", 0) > 0]
        inf_accs = [r.get("Informed Accuracy (%)", 0) for r in subset if r.get("Informed Accuracy (%)", 0) > 0]
        deltas = [r.get("Informed vs Pretrained Delta (pp)", 0) for r in subset if r.get("Pretrained Accuracy (%)", 0) > 0]

        if not pre_accs:
            continue

        print(f"  {cat}")
        print(f"    n models = {len(pre_accs)}")
        print(f"    Pretrained: mean={np.mean(pre_accs):.1f}%, SD={np.std(pre_accs, ddof=1):.1f}, range=[{min(pre_accs):.0f}%, {max(pre_accs):.0f}%]")
        if inf_accs:
            print(f"    Informed:   mean={np.mean(inf_accs):.1f}%, SD={np.std(inf_accs, ddof=1):.1f}, range=[{min(inf_accs):.0f}%, {max(inf_accs):.0f}%]")
            print(f"    Delta:      mean={np.mean(deltas):+.1f}pp, SD={np.std(deltas, ddof=1):.1f}, range=[{min(deltas):+.0f}, {max(deltas):+.0f}]pp")
        print()


def main():
    score_dir = "openworm_ai/quiz/scores/rag_full_logprob"
    summary, detailed = load_data(score_dir)

    print(f"Loaded {len(summary)} summary entries, {len(detailed)} detailed entries")
    print(f"Models: {len(set(r['LLM'] for r in summary))}")
    print(f"Categories: {sorted(set(r.get('Quiz Category','') for r in summary))}")

    # Descriptive stats first
    descriptive_summary(summary)

    # Run all tests for key categories
    for cat in ["C. elegans (Corpus)", "C. elegans"]:
        mcnemar_test(detailed, cat)
        wilcoxon_test(summary, cat)
        spearman_size_vs_rag(summary, cat)
        chi_squared_errors(summary, cat)
        bootstrap_ci(detailed, cat)
        similarity_correlation(detailed, cat)


if __name__ == "__main__":
    main()
