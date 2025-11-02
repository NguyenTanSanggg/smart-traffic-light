import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

folder_path = "./data"
smoothing = 0.99
output_folder = os.path.join(folder_path, "plots")

os.makedirs(output_folder, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

colors = plt.cm.tab10.colors

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)

    value_col = df.columns[-1]
    values = df[value_col]

    smoothed = values.ewm(alpha=1 - smoothing).mean()

    color = colors[i % len(colors)]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed, color=color, linewidth=2)
    plt.title(os.path.basename(file), fontsize=13)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(file))[0] + ".png"
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll smoothed scalar plots saved successfully!")
