
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("logs.pkl", "rb") as f:
     data = pickle.load(f)

print("Available runs:", list(data.keys()))

losses = ["gen_total_loss", "disc_loss"]
custom_names = {"gen_total_loss": "Generator Total Loss", "disc_loss": "Discriminator Loss",}

# Match the keys that actually exist in logs.pkl
disc_keys = sorted([k for k in data if k.startswith("disc_rep_")], key=lambda k: int(k.rsplit("_", 1)[1]))

def plot_group(keys, title):
    if not keys:
        print(f"No runs found for: {title}")
        return

    for loss in losses:
        plt.close()
        plt.rcParams.update({'font.size': 12})
        plt.rcParams["figure.figsize"] = [6,6]
        plt.rcParams['axes.facecolor']='ivory'
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        for name in keys:
            df = data[name]
            if loss not in df.columns or "step" not in df.columns:
                print(f"Missing required column in {name}")
                continue

            # Ensure values are numeric and valid
            values = df[loss].to_numpy(dtype=float)
            steps = df["step"].to_numpy(dtype=float)

            valid = (np.isfinite(values) & np.isfinite(steps) & (values >= 0))
            values = values[valid]
            steps = steps[valid]

            if len(values) == 0:
                continue

            cumulative_mean = np.cumsum(values) / np.arange(1, len(values) + 1)            
            y = np.sqrt(cumulative_mean)      # Preserve your original transformation

            plt.plot(steps, y, 'o', markersize=2, label=name)

        plt.title(custom_names[loss], fontweight="bold")
        plt.xlabel("Training Step")
        plt.ylabel(r"$\sqrt{\mathrm{Cumulative\ Mean\ Loss}}$")
        plt.legend()        
        plt.tight_layout()
        plt.savefig(f"{title}_{loss}.png", dpi=300)
        plt.close()

plot_group(disc_keys, "Discriminator_Rep_Runs")

