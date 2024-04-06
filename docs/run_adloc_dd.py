# %%
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
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


def plotting(events_new, stations, config, figure_path, events_old, iter=0):

    vmin = min(events_new["z_km"].min(), events_old["depth_km"].min())
    vmax = max(events_new["z_km"].max(), events_old["depth_km"].max())
    xmin = min(stations["x_km"].min(), events_old["x_km"].min())
    xmax = max(stations["x_km"].max(), events_old["x_km"].max())
    ymin = min(stations["y_km"].min(), events_old["y_km"].min())
    ymax = max(stations["y_km"].max(), events_old["y_km"].max())
    zmin, zmax = config["zlim_km"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    im = ax[0, 0].scatter(
        events_old["x_km"],
        events_old["y_km"],
        c=events_old["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 0])
    cbar.set_label("Depth (km)")
    ax[0, 0].set_title(f"ADLoc: {len(events_old)} events")

    im = ax[0, 1].scatter(
        events_new["x_km"],
        events_new["y_km"],
        c=events_new["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 1])
    cbar.set_label("Depth (km)")
    ax[0, 1].set_title(f"ADLoc DD: {len(events_new)} events")

    # im = ax[1, 0].scatter(
    #     events_new["x_km"],
    #     events_new["z_km"],
    #     c=events_new["z_km"],
    #     cmap="viridis_r",
    #     s=1,
    #     marker="o",
    #     vmin=vmin,
    #     vmax=vmax,
    # )
    # ax[1, 0].set_xlim([xmin, xmax])
    # ax[1, 0].set_ylim([zmax, zmin])
    # cbar = fig.colorbar(im, ax=ax[1, 0])
    # cbar.set_label("Depth (km)")

    im = ax[1, 0].scatter(
        events_old["y_km"],
        events_old["z_km"],
        c=events_old["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[1, 0].set_xlim([ymin, ymax])
    ax[1, 0].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 0])
    cbar.set_label("Depth (km)")

    im = ax[1, 1].scatter(
        events_new["y_km"],
        events_new["z_km"],
        c=events_new["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[1, 1].set_xlim([ymin, ymax])
    ax[1, 1].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 1])
    cbar.set_label("Depth (km)")
    plt.savefig(os.path.join(figure_path, f"location_{iter}.png"), bbox_inches="tight", dpi=300)


# %%
if __name__ == "__main__":

    # %%
    # data_path = "../tests/results/"
    # catalog_path = "../tests/results/"
    region = "synthetic"
    data_path = f"test_data/{region}"
    result_path = f"results/{region}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"figures/{region}/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    if os.path.exists(os.path.join(result_path, "adloc_dt.dat")):
        # promot to ask if you want to overwrite the existing results
        if input("Regenerate the double-difference pair file (adloc_dt.dat)? (y/n): ") == "y":
            os.system(
                f"python generate_pairs.py --stations {data_path}/stations.csv --events {data_path}/events.csv --picks {data_path}/picks.csv --result_path {result_path}"
            )
    else:
        os.system(
            f"python generate_pairs.py --stations {data_path}/stations.csv --events {data_path}/events.csv --picks {data_path}/picks.csv --result_path {result_path}"
        )

    # %%
    picks = pd.read_csv(os.path.join(result_path, "adloc_picks.csv"), parse_dates=["phase_time"])
    events = pd.read_csv(os.path.join(result_path, "adloc_events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(result_path, "adloc_stations.csv"))
    dtypes = pickle.load(open(os.path.join(result_path, "adloc_dtypes.pkl"), "rb"))
    pairs = np.memmap(os.path.join(result_path, "adloc_dt.dat"), mode="r", dtype=dtypes)
    events_old = events.copy()

    # %%
    ## Automatic region; you can also specify a region
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

    ## Project the coordinates
    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
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
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}

    # %%
    config["eikonal"] = None
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
    plotting(events, stations, config, figure_path, events_old, iter=0)

    # %%
    # utils.init_distributed_mode(args)
    # print(args)

    # %%
    phase_dataset = PhaseDatasetDD(pairs, picks, events, stations)
    # if args.distributed:
    #     sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
    # else:
    #     sampler = torch.utils.data.SequentialSampler(phase_dataset)
    # data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)
    data_loader = DataLoader(phase_dataset, batch_size=None, num_workers=0, collate_fn=None)

    # %%
    num_event = len(events)
    num_station = len(stations)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    # event_loc_init = np.zeros((num_event, 3))
    # event_loc_init[:, 2] = np.mean(config["zlim_km"])
    event_loc_init = events[["x_km", "y_km", "z_km"]].values.copy() + np.random.randn(num_event, 3) * 2.0
    travel_time = TravelTimeDD(
        num_event,
        num_station,
        station_loc,
        event_loc=event_loc_init,  # Initial location
        # event_loc=event_loc,  # Initial location
        # event_time=event_time,
        eikonal=config["eikonal"],
    )

    print(f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(phase_dataset)} batches")
    # optimize_dd(data_loader, travel_time, config)

    # %%
    # init loss
    loss = 0
    for meta in data_loader:
        station_index = meta["idx_sta"]
        event_index = meta["idx_eve"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]

        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
        )["loss"]

    # if args.distributed:
    #     dist.barrier()
    #     dist.all_reduce(loss)
    print(f"Init loss: {loss}")

    ## invert loss
    ######################################################################################################
    optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    EPOCHS = 100
    for i in range(EPOCHS):
        loss = 0
        optimizer.zero_grad()
        for meta in tqdm(data_loader, desc=f"Epoch {i}"):
            station_index = meta["idx_sta"]
            event_index = meta["idx_eve"]
            phase_time = meta["phase_time"]
            phase_type = meta["phase_type"]
            phase_weight = meta["phase_weight"]

            loss_ = travel_time(
                station_index,
                event_index,
                phase_type,
                phase_time,
                phase_weight,
            )["loss"]

            # optimizer.zero_grad()
            loss_.backward()
            # optimizer.step()
            loss += loss_

        optimizer.step()
        travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
        print(f"Loss: {loss}")

        invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = travel_time.event_time.weight.clone().detach().numpy()

        # events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
        events["x_km"] = invert_event_loc[:, 0]
        events["y_km"] = invert_event_loc[:, 1]
        events["z_km"] = invert_event_loc[:, 2]
        # events[["longitude", "latitude"]] = events.apply(
        #     lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        # )
        # events["depth_km"] = events["z_km"]
        # events.to_csv(
        #     f"{result_path}/adloc_dd_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
        # )

        if i % 10 == 0:
            plotting(events, stations, config, figure_path, events_old, iter=i + 1)

    ######################################################################################################
    optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = 0
        for meta in tqdm(data_loader, desc=f"BFGS"):
            station_index = meta["idx_sta"]
            event_index = meta["idx_eve"]
            phase_time = meta["phase_time"]
            phase_type = meta["phase_type"]
            phase_weight = meta["phase_weight"]

            loss_ = travel_time(
                station_index,
                event_index,
                phase_type,
                phase_time,
                phase_weight,
            )["loss"]
            loss_.backward()
            loss += loss_

        print(f"Loss: {loss}")
        travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
        return loss

    optimizer.step(closure)
    ######################################################################################################

    # travel_time.event_loc.weight.grad.data -= travel_time.event_loc.weight.grad.data.mean()
    # travel_time.event_time.weight.grad.data -= travel_time.event_time.weight.grad.data.mean()

    ## updated loss
    loss = 0
    for meta in data_loader:
        station_index = meta["idx_sta"]
        event_index = meta["idx_eve"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]

        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
        )["loss"]

    # if args.distributed:
    #     dist.barrier()
    #     dist.all_reduce(loss)

    print(f"Invert loss: {loss}")

    # set variable range
    # travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])

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
        f"{result_path}/adloc_dd_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
    )

    # %%
    plotting(events, stations, config, figure_path, events_old, iter=1)
