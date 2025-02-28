import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats

# Define model parameters (LLM parameter sizes in billions)
llm_parameters = {
    "GPT3.5": 20,
    "Phi4": 14,
    "Gemma2": 9,
    "Gemma": 7,
    "Qwen": 4,
    "Llama3.2": 1,
    "GPT4o": 1760,
    "Gemini": 500,
    "Claude 3.5 Sonnet": 175
}

# Define quiz categories and corresponding file paths
file_paths = {
    "General Knowledge": "openworm_ai/quiz/scores/general/llm_scores_general_24-02-25.json",
    "Science": "openworm_ai/quiz/scores/science/llm_scores_science_24-02-25.json",
    "C. Elegans": "openworm_ai/quiz/scores/celegans/llm_scores_celegans_24-02-25.json"
}

# Folder to save figures
figures_folder = "openworm_ai/quiz/figures"
os.makedirs(figures_folder, exist_ok=True)  # Ensure the folder exists

# Process each quiz category
for category, file_path in file_paths.items():
    save_path = os.path.join(figures_folder, f"llm_accuracy_vs_parameters_grid_{category.replace(' ', '_').lower()}.png")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found - {file_path}. Skipping this category.")
        continue

    # Load JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract relevant data
    category_results = []
    for result in data.get("Results", []):  # Use .get() to avoid KeyError
        for key in llm_parameters:
            if key.lower() in result["LLM"].lower():
                category_results.append({
                    "Model": key,
                    "Accuracy (%)": result["Accuracy (%)"],
                    "Parameters (B)": llm_parameters[key]
                })
                break

    # Skip if no data
    if not category_results:
        print(f"⚠️ No valid results found in {file_path}. Skipping...")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(category_results)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Scatter plot with model labels
    sns.scatterplot(data=df, x="Parameters (B)", y="Accuracy (%)", s=100, color="blue", edgecolor="black")

    # Add model labels to each point
    for i, row in df.iterrows():
        plt.text(row["Parameters (B)"], row["Accuracy (%)"], row["Model"], fontsize=10, ha="right", va="bottom")

    # Fit a regression line (log-log scale)
    log_x = np.log10(df["Parameters (B)"])
    log_y = df["Accuracy (%)"]

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_x, log_y)
    regression_x = np.linspace(min(log_x), max(log_x), 100)
    regression_y = slope * regression_x + intercept

    plt.plot(10 ** regression_x, regression_y, color="red", linestyle="dashed", label=f"Trend (r={r_value:.2f})")

    # Log scale for x-axis (model parameters)
    plt.xscale("log")

    # Title and labels
    plt.title(f"LLM Accuracy vs. Model Parameters - {category}")
    plt.xlabel("Model Parameters (B)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)  # Ensure consistent scale
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save figure
    plt.legend()
    plt.savefig(save_path)
    print(f"✅ Saved plot: {save_path}")
    plt.show()
