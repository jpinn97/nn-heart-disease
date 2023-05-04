import json
import os
import pandas as pd
import matplotlib.pyplot as plt

my_dir = "my_dir/bayesian_optimization"

trial_dirs = [
    os.path.join(my_dir, d)
    for d in os.listdir(my_dir)
    if os.path.isdir(os.path.join(my_dir, d))
]

data = []

for trial_dir in trial_dirs:
    trial_json_path = os.path.join(trial_dir, "trial.json")

    if os.path.exists(trial_json_path):
        with open(trial_json_path, "r") as f:
            trial_json = json.load(f)

        hp_values = trial_json["hyperparameters"]["values"]

        if "val_loss" in trial_json["metrics"]["metrics"]:
            best_val_loss = trial_json["metrics"]["metrics"]["val_loss"][
                "observations"
            ][0]["value"][0]
            data.append({**hp_values, "val_loss": best_val_loss})

# Create a pandas DataFrame using the data list
df = pd.DataFrame(data)

# Sort the DataFrame based on the "val_loss" column
df_sorted = df.sort_values(by="val_loss")
print(df_sorted.head(100))  # Top 100 models

# Visualize the relationship between different hyperparameters and the "val_loss" using scatter plots
plt.scatter(df["units_0"], df["lr"], c=df["val_loss"], cmap="viridis")
plt.xlabel("Number of Layers")
plt.ylabel("Learning Rate")
plt.colorbar(label="Validation Loss")
plt.show()

# Calculate the correlation between different hyperparameters and the "val_loss"
correlations = df.corr()
print(correlations["val_loss"])
