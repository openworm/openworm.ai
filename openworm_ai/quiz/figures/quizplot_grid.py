import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ruff: noqa: F401
from openworm_ai.utils.llms import (
    LLM_OLLAMA_LLAMA32_1B,
    LLM_OLLAMA_LLAMA32_3B,
    LLM_GPT4o,
    LLM_GEMINI_2F,
    LLM_CLAUDE37,
    LLM_GPT35,
    LLM_OLLAMA_PHI4,
    LLM_OLLAMA_GEMMA2,
    LLM_OLLAMA_GEMMA,
    LLM_OLLAMA_QWEN,
    LLM_OLLAMA_TINYLLAMA,
    ask_question_get_response,
)

# Define model parameters (LLM parameter sizes in billions)
llm_parameters = {
    LLM_GPT4o: 1760,
    LLM_GPT35: 175,
    "GPT3.5": 20,
    "Phi4": 14,
    "Gemma2": 9,
    "Gemma": 7,
    "Qwen": 4,
    "Llama3.2": 1,
    "TinyLlama": 1.1,
    "GPT4o": 1760,
    "Gemini": 500,
    "Claude 3.5 Sonnet": 175,
}

# Define model distributors for coloring
model_distributors = {
    LLM_GPT4o: "OpenAI",
    LLM_GPT35: "OpenAI",
    "GPT3.5": "OpenAI",
    "GPT4o": "OpenAI",
    "Phi4": "Microsoft",
    "Gemma2": "Google",
    "Gemma": "Google",
    "Gemini": "Google",
    "Claude 3.5 Sonnet": "Anthropic",
    "Qwen": "Alibaba",
    "Llama3.2": "Meta",
    "TinyLlama": "Open Source",
}

# Define quiz categories and corresponding file paths
file_paths = {
    # "General Knowledge": "openworm_ai/quiz/scores/general/llm_scores_general_24-02-25.json",
    # "Science": "openworm_ai/quiz/scores/science/llm_scores_science_24-02-25.json",
    # "C. Elegans": "openworm_ai/quiz/scores/celegans/llm_scores_celegans_24-02-25.json",
    "RAG": "openworm_ai/quiz/scores/rag/llm_scores_rag_16-03-25_2.json"
}

# Folder to save figures
figures_folder = "openworm_ai/quiz/figures"
os.makedirs(figures_folder, exist_ok=True)  # Ensure the folder exists

# Define colors per distributor
distributor_colors = {
    "OpenAI": "blue",
    "Google": "red",
    "Anthropic": "green",
    "Microsoft": "purple",
    "Alibaba": "orange",
    "Meta": "cyan",
    "Open Source": "yellow",
}

# Process each quiz category
for category, file_path in file_paths.items():
    save_path = os.path.join(
        figures_folder,
        f"llm_accuracy_vs_parameters_{category.replace(' ', '_').lower()}.png",
    )

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
        print(6)
        for key in llm_parameters:
            print("---" + key)
            if key.lower() in result["LLM"].lower():
                print(44)
                category_results.append(
                    {
                        "Model": key,
                        "Accuracy (%)": result["Accuracy (%)"],
                        "Parameters (B)": llm_parameters[key],
                        "Distributor": model_distributors.get(key, "Unknown"),
                    }
                )
                break

    # Skip if no data
    if not category_results:
        print(f"⚠️  No valid results found in {file_path}. Skipping...")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(category_results)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Scatter plot with model labels, colored by distributor
    for distributor, color in distributor_colors.items():
        subset = df[df["Distributor"] == distributor]
        plt.scatter(
            subset["Parameters (B)"],
            subset["Accuracy (%)"],
            s=100,
            color=color,
            label=distributor,
            edgecolor="black",
        )

    # Add model labels to each point
    for i, row in df.iterrows():
        plt.text(
            row["Parameters (B)"],
            row["Accuracy (%)"],
            row["Model"],
            fontsize=10,
            ha="right",
            va="bottom",
        )

    # Log scale for x-axis (model parameters)
    plt.xscale("log")

    # Title and labels
    plt.title(f"LLM Accuracy vs. Model Parameters - {category}")
    plt.xlabel("Model Parameters (B)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)  # Ensure consistent scale
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save figure
    plt.legend()
    plt.savefig(save_path)
    print(f"✅ Saved plot: {save_path}")
    if "-nogui" not in sys.argv:
        plt.show()
