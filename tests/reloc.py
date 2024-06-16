# %%
import matplotlib
import numpy as np
import scipy
import h5py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../adloc'))
sys.path.append(project_path)
from sacloc3d import init_eikonal3d, ADLoc
from _ransac import RANSACRegressor

# from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

np.random.seed(0)
# matplotlib.use("Agg")

# %%
if __name__ == "__main__":

    # %%
    import json
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd

    # data_path = "../tests/results"
    data_path = "./checkerboard"
    events = pd.read_csv(os.path.join(data_path, "events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    picks = pd.read_csv(os.path.join(data_path, "picks_err_0.1.csv"), parse_dates=["phase_time"])
    with open(os.path.join(data_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    eikonal = init_eikonal3d(config, stations, data_path)

    mapping_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["vel"] = {mapping_int[k]: v for k, v in config["vel"].items()}

    # %%
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous
    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    
    num_event = len(events)
    event_input = np.ones((num_event,4))
    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, num_event=num_event, eikonal=eikonal, events=event_input)

    tru_loc = pd.DataFrame(columns=['x', 'y', 'z'])
    for event_index in range(num_event):
        
        picks_event = picks[picks["idx_eve"] == event_index]

        X = picks_event.merge(stations[["x_km", "y_km", "z_km", "station_id"]], on="station_id")
        t0 = X["phase_time"].min()
        X.rename(columns={"phase_type": "type", "phase_time": "t_s"}, inplace=True)
        X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
        X = X[["x_km", "y_km", "z_km", "t_s", "type", "idx_sta"]]
        mapping_int = {"P": 0, "S": 1}
        X["type"] = X["type"].apply(lambda x: mapping_int[x.upper()])

        tt = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
        estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values, event_index=event_index)
        # print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values}")
        # print(f"Init event loc: {estimator.events[event_index]}")

        estimator.fit(X[["idx_sta", "type"]].values, y=X["t_s"].values, event_index=event_index)
        estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values, event_index=event_index)
        tt = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
        # print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values}")
        # print(f"Estimated_loc_fit: {estimator.events[event_index]}")
    
    ev = estimator.events
    index = np.arange(num_event)

    fold = "picks_0.1"
    plt.figure()
    plt.scatter(events['x_km'], events['y_km'], c=index, cmap='hsv', marker='o')
    plt.scatter(ev[:,0], ev[:,1], c=index, cmap='hsv', marker='x', s=100)
    plt.xlabel('x_km'); plt.ylabel('y_km')
    plt.savefig(data_path + "/png/error_" + fold + "_xy.png")
    plt.figure()
    plt.scatter(events['x_km'], events['z_km'], c=index, cmap='hsv', marker='o')
    plt.scatter(ev[:,0], ev[:,2], c=index, cmap='hsv', marker='x', s=100)
    plt.xlabel('x_km'); plt.ylabel('z_km')
    plt.savefig(data_path + "/png/error_" + fold + "_xz.png")
    plt.figure()
    plt.scatter(events['y_km'], events['z_km'], c=index, cmap='hsv', marker='o')
    plt.scatter(ev[:,1], ev[:,2], c=index, cmap='hsv', marker='x', s=100)
    plt.xlabel('y_km'); plt.ylabel('z_km')
    plt.savefig(data_path + "/png/error_" + fold + "_yz.png")