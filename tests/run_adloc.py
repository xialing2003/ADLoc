# %%
import argparse
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

data_path = "results"
result_path = f"results/adloc"
if not os.path.exists(f"{result_path}"):
    os.makedirs(f"{result_path}")

# %%
epochs = 1
batch = 100
double_difference = False
base_cmd = f"../run.py --config {data_path}/config.json --stations {data_path}/stations.json --events {data_path}/events.csv --picks {data_path}/picks.csv --result_path {result_path} --batch_size {batch}"
if double_difference:
    base_cmd += " --double_difference"
os.system(f"python {base_cmd} --device=cpu --epochs={epochs}")

# %%
events_true = pd.read_csv(f"{data_path}/events.csv")
events_invert = pd.read_csv(f"{result_path}/adloc_events.csv")
stations = pd.read_json(f"{data_path}/stations.json", orient="index")

# %%
plt.figure()
plt.scatter(stations["longitude"], stations["latitude"], marker="^", s=10)
plt.scatter(events_true["longitude"], events_true["latitude"], s=1, label="true")
plt.scatter(events_invert["longitude"], events_invert["latitude"], s=1, label="invert")
plt.legend()
plt.savefig(f"{result_path}/events_xy.png", dpi=300)

plt.figure()
plt.scatter(stations["longitude"], stations["depth_km"], marker="^", s=10)
plt.scatter(events_true["longitude"], events_true["depth_km"], s=1, label="true")
plt.scatter(events_invert["longitude"], events_invert["depth_km"], s=1, label="invert")
plt.gca().invert_yaxis()
plt.legend()
plt.savefig(f"{result_path}/events_xz.png", dpi=300)

plt.figure()
plt.scatter(stations["latitude"], stations["depth_km"], marker="^", s=10)
plt.scatter(events_true["latitude"], events_true["depth_km"], s=1, label="true")
plt.scatter(events_invert["latitude"], events_invert["depth_km"], s=1, label="invert")
plt.gca().invert_yaxis()
plt.legend()
plt.savefig(f"{result_path}/events_yz.png", dpi=300)

# %%
