
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("logs.pkl", "rb") as f:
    data = pickle.load(f)
    
for name, df in data.items():
    print(name, type(df))

    if df is None:
        print(f"WARNING: {name} contains None!")

print("Available runs:", list(data.keys()))

losses = ["gen_total_loss", "disc_loss"]
custom_names = {
    "gen_total_loss": "Generator Total Loss",
    "disc_loss": "Discriminator Loss",
}

# Match the keys that actually exist in logs.pkl
disc_keys = sorted(
    [k for k in data if k.startswith("disc_rep_")],
    key=lambda k: int(k.rsplit("_", 1)[1])
)

def plot_group(keys, title):
    if not keys:
        print(f"No runs found for: {title}")
        return

    for loss in losses:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_facecolor("ivory")

        plotted = False

        for name in keys:
            df = data[name]

            if loss not in df.columns or "step" not in df.columns:
                print(f"Missing required column in {name}")
                continue

            # Ensure values are numeric and valid
            values = df[loss].to_numpy(dtype=float)
            steps = df["step"].to_numpy(dtype=float)

            valid = (
                np.isfinite(values)
                & np.isfinite(steps)
                & (values >= 0)
            )
            values = values[valid]
            steps = steps[valid]

            if len(values) == 0:
                continue

            cumulative_mean = np.cumsum(values) / np.arange(
                1, len(values) + 1
            )

            # Preserve your original transformation
            y = np.sqrt(cumulative_mean)

            ax.plot(
                steps, y,
                linewidth=1.2,
                label=name
            )
            plotted = True

        ax.set_title(custom_names[loss], fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel(r"$\sqrt{\mathrm{Cumulative\ Mean\ Loss}}$")
        ax.grid(True, alpha=0.25)

        if plotted:
            ax.legend()
        else:
            print(f"No valid data to plot for {loss}")
        
        fig.tight_layout()
        fig.savefig(f"{title}_{loss}.png", dpi=300)
        plt.close(fig)

plot_group(disc_keys, "Discriminator_Rep_Runs")

