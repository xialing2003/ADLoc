# %% Download test data
# !if [ -f demo.tar ]; then rm demo.tar; fi
# !if [ -d test_data ]; then rm -rf test_data; fi
# !wget -q https://github.com/AI4EPS/GaMMA/releases/download/test_data/demo.tar
# !tar -xf demo.tar

# %%
import json
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj

from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location

# %%
if __name__ == "__main__":
    # %%
    data_path = "test_data/ridgecrest/"
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    stations["depth_km"] = -stations["elevation_m"] / 1000
    figure_path = "figures/ridgecrest/"
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

    # %% config for location
    config["min_picks"] = 4
    config["min_picks_ratio"] = 0.2
    config["max_residual_s"] = 1.0
    config["min_score"] = 0.9
    config["min_s_picks"] = 2
    config["min_p_picks"] = 2

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
    data_path = "results/ridgecrest/"
    picks = pd.read_csv(os.path.join(data_path, "gamma_picks.csv"), parse_dates=["phase_time"])
    events = pd.read_csv(os.path.join(data_path, "gamma_events.csv"), parse_dates=["time"])
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)

    # %%
    stations["station_term"] = 0.0
    # stations["station_term_p"] = 0.0
    # stations["station_term_s"] = 0.0
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %% backup old events
    events_old = events.copy()
    events_old[["x_km", "y_km"]] = events_old.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events_old["z_km"] = events["depth_km"]

    # %%
    def plotting(stations, figure_path, config, picks, events_old, locations, station_term, iter=0):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax[0, 0].hist(locations["adloc_score"], bins=30, edgecolor="white")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_title("ADLoc score")
        ax[0, 1].hist(locations["num_picks"], bins=30, edgecolor="white")
        ax[0, 1].set_title("Number of picks")
        ax[1, 0].hist(locations["adloc_residual_s"], bins=30, edgecolor="white")
        ax[1, 0].set_title("Event residual (s)")
        ax[1, 1].hist(picks[picks["mask"] == 1.0]["residual_s"], bins=30, edgecolor="white")
        ax[1, 1].set_title("Pick residual (s)")
        plt.savefig(os.path.join(figure_path, f"hist_{iter}.png"), bbox_inches="tight", dpi=300)

        vmin = min(locations["z_km"].min(), events_old["depth_km"].min())
        vmax = max(locations["z_km"].max(), events_old["depth_km"].max())
        xmin, xmax = stations["x_km"].min(), stations["x_km"].max()
        ymin, ymax = stations["y_km"].min(), stations["y_km"].max()
        zmin, zmax = config["zlim_km"]

        fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
        # fig, ax = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [2, 1]})
        im = ax[0, 0].scatter(
            locations["x_km"],
            locations["y_km"],
            c=locations["z_km"],
            cmap="viridis_r",
            s=1,
            marker="o",
            vmin=vmin,
            vmax=vmax,
        )
        ax[0, 0].set_xlim([xmin, xmax])
        ax[0, 0].set_ylim([ymin, ymax])
        cbar = fig.colorbar(im, ax=ax[0, 0])
        cbar.set_label("Depth (km)")
        ax[0, 0].set_title(f"ADLoc: {len(locations)} events")

        im = ax[0, 1].scatter(
            stations["x_km"],
            stations["y_km"],
            c=stations["station_term"],
            cmap="viridis_r",
            s=100,
            marker="^",
            alpha=0.5,
        )
        ax[0, 1].set_xlim([xmin, xmax])
        ax[0, 1].set_ylim([ymin, ymax])
        cbar = fig.colorbar(im, ax=ax[0, 1])
        cbar.set_label("Residual (s)")
        ax[0, 1].set_title(f"Station term: {np.mean(np.abs(stations['station_term'].values)):.4f} s")

        ## Separate P and S station term
        # im = ax[0, 1].scatter(
        #     stations["x_km"],
        #     stations["y_km"],
        #     c=stations["station_term_p"],
        #     cmap="viridis_r",
        #     s=100,
        #     marker="^",
        #     alpha=0.5,
        # )
        # ax[0, 1].set_xlim([xmin, xmax])
        # ax[0, 1].set_ylim([ymin, ymax])
        # cbar = fig.colorbar(im, ax=ax[0, 1])
        # cbar.set_label("Residual (s)")
        # ax[0, 1].set_title(f"Station term (P): {np.mean(np.abs(stations['station_term_p'].values)):.4f} s")

        # im = ax[0, 2].scatter(
        #     stations["x_km"],
        #     stations["y_km"],
        #     c=stations["station_term_s"],
        #     cmap="viridis_r",
        #     s=100,
        #     marker="^",
        #     alpha=0.5,
        # )
        # ax[0, 2].set_xlim([xmin, xmax])
        # ax[0, 2].set_ylim([ymin, ymax])
        # cbar = fig.colorbar(im, ax=ax[0, 2])
        # cbar.set_label("Residual (s)")
        # ax[0, 2].set_title(f"Station term (S): {np.mean(np.abs(stations['station_term_s'].values)):.4f} s")

        im = ax[1, 0].scatter(
            locations["x_km"],
            locations["z_km"],
            c=locations["z_km"],
            cmap="viridis_r",
            s=1,
            marker="o",
            vmin=vmin,
            vmax=vmax,
        )
        ax[1, 0].set_xlim([xmin, xmax])
        ax[1, 0].set_ylim([zmax, zmin])
        cbar = fig.colorbar(im, ax=ax[1, 0])
        cbar.set_label("Depth (km)")

        im = ax[1, 1].scatter(
            locations["y_km"],
            locations["z_km"],
            c=locations["z_km"],
            cmap="viridis_r",
            s=1,
            marker="o",
            vmin=vmin,
            vmax=vmax,
        )
        ax[1, 1].set_xlim([ymin, ymax])
        ax[1, 1].set_ylim([zmax, zmin])
        cbar = fig.colorbar(im, ax=ax[1, 1])
        cbar.set_label("Depth (km)")
        plt.savefig(os.path.join(figure_path, f"location_{iter}.png"), bbox_inches="tight", dpi=300)

    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=config["eikonal"])
    event_init = np.array([[np.mean(config["xlim_km"]), np.mean(config["ylim_km"]), np.mean(config["zlim_km"]), 0.0]])

    # %%
    NCPU = mp.cpu_count()
    MAX_SST_ITER = 10
    # MIN_SST_S = 0.01
    iter = 0
    while iter < MAX_SST_ITER:

        picks, locations = invert_location(picks, events, stations, config, estimator, iter=iter)
        station_term = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_s": "mean"}).reset_index()
        stations["station_term"] += stations["idx_sta"].map(station_term.set_index("idx_sta")["residual_s"]).fillna(0)

        ## Separate P and S station term
        # station_term = (
        #     picks[picks["mask"] == 1.0].groupby(["idx_sta", "phase_type"]).agg({"residual_s": "mean"}).reset_index()
        # )
        # stations["station_term_p"] = (
        #     stations["idx_sta"]
        #     .map(station_term[station_term["phase_type"] == 0].set_index("idx_sta")["residual_s"])
        #     .fillna(0)
        # )
        # stations["station_term_s"] = (
        #     stations["idx_sta"]
        #     .map(station_term[station_term["phase_type"] == 1].set_index("idx_sta")["residual_s"])
        #     .fillna(0)
        # )

        plotting(stations, figure_path, config, picks, events_old, locations, station_term, iter=iter)

        if iter == 0:
            MIN_SST_S = np.mean(np.abs(station_term["residual_s"])) / 10.0  # break at 10% of the initial station term
            print(f"MIN_SST (s): {MIN_SST_S}")
        if np.mean(np.abs(station_term["residual_s"])) < MIN_SST_S:
            print(f"Mean station term: {np.mean(np.abs(station_term['residual_s']))}")
            break
        iter += 1

# %%
