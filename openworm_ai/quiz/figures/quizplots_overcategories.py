import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Create an empty DataFrame to store results for all categories
df_all = pd.DataFrame()

# Process each quiz category
for category, file_path in file_paths.items():
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found - {file_path}. Skipping this category.")
        continue

    # Load JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract results
    category_results = []
    for result in data.get("Results", []):  # Use .get() to avoid KeyError
        for key in llm_parameters:
            if key.lower() in result["LLM"].lower():
                category_results.append({
                    "Model": key,
                    "Task Complexity": category,
                    "Accuracy (%)": result["Accuracy (%)"],
                    "Parameters (B)": llm_parameters[key]
                })
                break

    # Append results to the main DataFrame
    if category_results:
        df_category = pd.DataFrame(category_results)
        df_all = pd.concat([df_all, df_category], ignore_index=True)

# Ensure the task complexity is treated as an ordered category
df_all["Task Complexity"] = pd.Categorical(df_all["Task Complexity"],
                                           categories=["General Knowledge", "Science", "C. Elegans"],
                                           ordered=True)

# ✅ Plot: Accuracy vs. Task Complexity
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_all, x="Task Complexity", y="Accuracy (%)", hue="Model", marker="o", linewidth=2)

# Improve readability
plt.title("LLM Performance vs. Task Complexity")
plt.xlabel("Task Complexity (Increasing Difficulty →)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)  # Consistent scale
plt.xticks(rotation=20)  # Rotate x-axis labels for clarity
plt.grid(True)
plt.legend(title="Model")

# Save the figure
plot_path = os.path.join(figures_folder, "llm_performance_vs_task_complexity.png")
plt.savefig(plot_path)
print(f"✅ Saved plot: {plot_path}")

# Show the plot
plt.show()
