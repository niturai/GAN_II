import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv

with open("logs.pkl", "rb") as f:                                         # Opens the file logs.pkl in read-binary mode (rb).
    data = pickle.load(f)                                                 # load dictionary of DataFrames

print(type(data))                                                         # the type of the object
print(data)                                                               # print the data


losses = ["gen_total_loss", "disc_loss"]                                  # a list of column names to plot from the DataFrames.
custom_name = ["Generator Total Loss", "Discriminator Loss"]

tele_keys = [k for k in data.keys() if "tele" in k]                       # separates data into telescope runs.
discrep_keys = [k for k in data.keys() if "discrep" in k]                 # separates data into discrep runs.

def plot_group(keys, title):
    for i, loss in enumerate(losses):                                     # iterate over each loss type
        plt.rcParams.update({'font.size': 12})
        plt.rcParams["figure.figsize"] = [6,6]
        plt.rcParams['axes.facecolor']='ivory'
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        for name in keys:                                                 # iterate over each key (run)
            df = data[name]

            if loss not in df.columns:                                    # check column existence
                print(f"Column {loss} not found in {name}")
                continue

            cumulative_mean = df[loss].expanding().mean()
            plt.plot(df["step"], np.sqrt(cumulative_mean), '.', label=name)

        plt.title(custom_name[i], fontweight='bold')
        plt.xlabel("Training Step")
        plt.ylabel(r"$\sqrt{\mathrm{Cumulative\ Mean\ of\ Losses}}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{title}_{loss}.png')
        plt.close()
        
plot_group(tele_keys, "Telescope Runs Losses")                            # Plot telescope runs

plot_group(discrep_keys, "Discrep Runs Losses")                           # Plot discrep runs
