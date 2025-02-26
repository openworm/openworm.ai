import json
import matplotlib.pyplot as plt
import numpy as np

# Define model parameters
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

# File paths for quiz results
file_paths = {
    "General Knowledge": "openworm_ai/quiz/scores/general/llm_scores_general_24-02-25.json",
    "Science": "openworm_ai/quiz/scores/science/llm_scores_science_24-02-25.json",
    "C. Elegans": "openworm_ai/quiz/scores/celegans/llm_scores_celegans_24-02-25.json"
}

# Folder to save figures
figures_folder = "openworm_ai/quiz/figures"

for category, file_path in file_paths.items():
    save_path = f"{figures_folder}/llm_accuracy_vs_parameters_{category.replace(' ', '_').lower()}.png"
    
    # Load JSON data
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Extract relevant data
    filtered_results = []
    for result in data["Results"]:
        for key in llm_parameters:
            if key.lower() in result["LLM"].lower():
                filtered_results.append({
                    "LLM": key,
                    "Accuracy": result["Accuracy (%)"],
                    "Parameters": llm_parameters[key]
                })
                break
    
    # Sort data by parameters
    filtered_results.sort(key=lambda x: x["Parameters"])
    
    # Extract x (models), y (accuracy), and parameters
    models = [entry["LLM"] for entry in filtered_results]
    y_accuracy = np.array([entry["Accuracy"] for entry in filtered_results])
    y_parameters = np.array([entry["Parameters"] for entry in filtered_results])
    
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create a second y-axis
    
    # Plot bar graph for accuracy
    ax1.bar(models, y_accuracy, color='blue', alpha=0.6, label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)", color='blue')
    ax1.set_xlabel("Model")
    ax1.set_ylim(0, 100)
    
    # Plot scatter for parameters
    ax2.scatter(models, y_parameters, color='red', marker='o', label="Parameters (B)")
    ax2.set_ylabel("Number of Parameters (B)", color='red')
    ax2.set_yscale('log')  # Log scale for better visualization of parameter differences
    
    # Title and legend
    plt.title(f"LLM Accuracy vs. Model Parameters - {category}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid()
    
    # Save figure
    plt.savefig(save_path)
    plt.show()

