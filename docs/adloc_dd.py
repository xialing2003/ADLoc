# %%
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pyproj import Proj
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from adloc.adloc import TravelTimeDD
from adloc.data import PhaseDatasetDD
from adloc.eikonal2d import init_eikonal2d
from adloc.inversion import optimize_dd

torch.manual_seed(0)
np.random.seed(0)

# %%
if __name__ == "__main__":

    # %%
    # data_path = "../tests/results/"
    # catalog_path = "../tests/results/"
    data_path = "test_data/ridgecrest"
    catalog_path = "results/ridgecrest"
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    stations["depth_km"] = -stations["elevation_m"] / 1000
    result_path = "results/synthetic/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = "figures/synthetic/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    ## Automatic region; you can also specify a region
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    xmin = stations["x_km"].min()
    xmax = stations["x_km"].max()
    ymin = stations["y_km"].min()
    ymax = stations["y_km"].max()
    x0 = (xmin + xmax) / 2
    y0 = (ymin + ymax) / 2
    ## set up the config; you can also specify the region manually
    config = {}
    config["xlim_km"] = (2 * xmin - x0, 2 * xmax - x0)
    config["ylim_km"] = (2 * ymin - y0, 2 * ymax - y0)
    config["zlim_km"] = (stations["z_km"].min(), 20)
    zmin = config["zlim_km"][0]
    zmax = config["zlim_km"][1]

    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        # (config["zlim_km"][0], config["zlim_km"][1] + 1),  # z
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}

    # %%
    ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3
    vel = {"Z": zz, "P": vp, "S": vs}
    config["eikonal"] = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    config["eikonal"] = init_eikonal2d(config["eikonal"])

    # %%
    plt.figure()
    plt.scatter(stations["x_km"], stations["y_km"], c=stations["depth_km"], cmap="viridis_r", s=100, marker="^")
    plt.colorbar(label="Depth (km)")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.xlim(config["xlim_km"])
    plt.ylim(config["ylim_km"])
    plt.title("Stations")
    plt.savefig(os.path.join(figure_path, "stations.png"), bbox_inches="tight", dpi=300)

    # %%
    # picks = pd.read_csv(os.path.join(data_path, "picks.csv"), parse_dates=["phase_time"])
    # events = pd.read_csv(os.path.join(data_path, "events.csv"), parse_dates=["time"])
    # # picks = pd.read_csv(os.path.join(catalog_path, "adloc_picks.csv"), parse_dates=["phase_time"])
    # # events = pd.read_csv(os.path.join(catalog_path, "adloc_events.csv"), parse_dates=["time"])

    # events[["x_km", "y_km"]] = events.apply(
    #     lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    # )
    # events["z_km"] = events["depth_km"]
    # picks = picks[picks["event_index"] != -1]
    # picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)

    # # %%
    # stations["station_term"] = 0.0
    # # stations["station_term_p"] = 0.0
    # # stations["station_term_s"] = 0.0
    # stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    # events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous

    # picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    # picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # # %%
    # picks["phase_time"] = picks.apply(
    #     lambda x: (x["phase_time"] - events.loc[x["idx_eve"], "time"]).total_seconds(), axis=1
    # )

    # # %% backup old events
    # events_old = events.copy()
    # events_old[["x_km", "y_km"]] = events_old.apply(
    #     lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    # )
    # events_old["z_km"] = events["depth_km"]

    # %%
    # utils.init_distributed_mode(args)
    # print(args)

    # %%
    data_path = "results/ridgecrest"
    picks = pd.read_csv(os.path.join(data_path, "adloc_picks.csv"), parse_dates=["phase_time"])
    events = pd.read_csv(os.path.join(data_path, "adloc_events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(data_path, "adloc_stations.csv"))
    # pairs = np.load(os.path.join(data_path, "adloc_dt.npz"))
    dtypes = pickle.load(open(os.path.join(data_path, "adloc_dtypes.pkl"), "rb"))
    pairs = np.memmap(os.path.join(data_path, "adloc_dt.dat"), mode="r", dtype=dtypes)
    print(dtypes)
    print(len(pairs["dd_time"]))

    # %%

    # events[["x_km", "y_km", "z_km"]] = events[["x_km", "y_km", "z_km"]].values * 0
    phase_dataset = PhaseDatasetDD(pairs, picks, events, stations)
    # if args.distributed:
    #     sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
    # else:
    #     sampler = torch.utils.data.SequentialSampler(phase_dataset)
    # data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)
    data_loader = DataLoader(phase_dataset, batch_size=None, num_workers=0, collate_fn=None)

    # for x in data_loader:
    #     print(x)
    #     break

    # %%
    num_event = len(events)
    num_station = len(stations)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    event_loc_init = np.zeros((num_event, 3))
    event_loc_init[:, 2] = np.mean(config["zlim_km"])
    travel_time = TravelTimeDD(
        num_event,
        num_station,
        station_loc,
        event_loc=event_loc_init,  # Initial location
        # event_loc=event_loc,  # Initial location
        # event_time=event_time,
        # eikonal=config["eikonal"],
    )

    print(f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(phase_dataset)} batches")
    optimize_dd(data_loader, travel_time, config)

    # %%
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()

    events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
    events["x_km"] = invert_event_loc[:, 0]
    events["y_km"] = invert_event_loc[:, 1]
    events["z_km"] = invert_event_loc[:, 2]
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events.to_csv(
        f"{result_path}/adloc_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
    )

    # %%
    fig, ax = plt.subplots(3, 1, figsize=(6, 15), squeeze=False)
    ax[0, 0].scatter(event_loc_init[:, 0], event_loc_init[:, 1], s=100, marker="o", label="Init")
    ax[0, 0].scatter(events["x_km"], events["y_km"], s=100, marker="o", label="Inverted")
    ax[0, 0].scatter(events_old["x_km"], events_old["y_km"], s=100, marker="x", label="True")
    ax[0, 0].scatter(stations["x_km"], stations["y_km"], c=stations["depth_km"], cmap="viridis_r", s=100, marker="^")
    ax[0, 0].set_xlabel("X (km)")
    ax[0, 0].set_ylabel("Y (km)")
    ax[0, 0].legend()

    ax[1, 0].scatter(event_loc_init[:, 0], event_loc_init[:, 2], s=100, marker="o", label="Init")
    ax[1, 0].scatter(events["x_km"], events["z_km"], s=100, marker="o", label="Inverted")
    ax[1, 0].scatter(events_old["x_km"], events_old["z_km"], s=100, marker="x", label="True")
    ax[1, 0].set_xlabel("X (km)")
    ax[1, 0].set_ylabel("Z (km)")
    ax[1, 0].legend()

    ax[2, 0].scatter(event_loc_init[:, 1], event_loc_init[:, 2], s=100, marker="o", label="Init")
    ax[2, 0].scatter(events["y_km"], events["z_km"], s=100, marker="o", label="Inverted")
    ax[2, 0].scatter(events_old["y_km"], events_old["z_km"], s=100, marker="x", label="True")
    ax[2, 0].set_xlabel("Y (km)")
    ax[2, 0].set_ylabel("Z (km)")
    ax[2, 0].legend()

    plt.savefig(os.path.join(figure_path, "events.png"), bbox_inches="tight", dpi=300)

# %%
