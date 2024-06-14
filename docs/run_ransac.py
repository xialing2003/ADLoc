# %% Download test data
# !if [ -f demo.tar ]; then rm demo.tar; fi
# !if [ -d test_data ]; then rm -rf test_data; fi
# !wget -q https://github.com/AI4EPS/datasets/releases/download/test_data/test_data.tar
# !tar -xf test_data.tar

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
    # data_path = "test_data/ridgecrest/"
    # region = "synthetic"
    region = "ridgecrest"
    data_path = f"test_data/{region}/"
    config = json.load(open(os.path.join(data_path, "config.json")))
    # picks = pd.read_csv(os.path.join(data_path, "gamma_picks.csv"), parse_dates=["phase_time"])
    # events = pd.read_csv(os.path.join(data_path, "gamma_events.csv"), parse_dates=["time"])
    picks = pd.read_csv(os.path.join(data_path, "phasenet_plus_picks.csv"), parse_dates=["phase_time"])
    events = None
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    stations["depth_km"] = -stations["elevation_m"] / 1000
    if "station_term" not in stations.columns:
        stations["station_term"] = 0.0
    result_path = f"results/{region}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"figures/{region}/"
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

    ## set up the config; you can also specify the region manually
    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):
        # xmin = stations["x_km"].min() - 50
        # xmax = stations["x_km"].max() + 50
        # ymin = stations["y_km"].min() - 50
        # ymax = stations["y_km"].max() + 50
        # zmin = stations["z_km"].min()
        # zmax = 20
        # x0 = (xmin + xmax) / 2
        # y0 = (ymin + ymax) / 2
        # config["xlim_km"] = (2 * xmin - x0, 2 * xmax - x0)
        # config["ylim_km"] = (2 * ymin - y0, 2 * ymax - y0)
        # config["zlim_km"] = (zmin, zmax)

        # project minlatitude, maxlatitude, minlongitude, maxlongitude to ymin, ymax, xmin, xmax
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin = stations["z_km"].min()
        zmax = 20
        config = {}
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}

    # %%
    config["eikonal"] = None
    # ## Eikonal for 1D velocity model
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
    config["min_picks"] = 8
    config["min_picks_ratio"] = 0.2
    config["max_residual_s"] = 1.0
    config["min_score"] = 0.9
    config["min_s_picks"] = 2
    config["min_p_picks"] = 2

    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        # (config["zlim_km"][0], config["zlim_km"][1] + 1),  # z
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

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
    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)

    # %%
    # stations["station_term"] = 0.0
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    if events is not None:
        events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous
        picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    else:
        picks["idx_eve"] = picks["event_index"]

    # %% backup old events
    if events is not None:
        events_old = events.copy()
        events_old[["x_km", "y_km"]] = events_old.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
        )
        events_old["z_km"] = events["depth_km"]
    else:
        events_old = None

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

        xmin, xmax = config["xlim_km"]
        ymin, ymax = config["ylim_km"]
        zmin, zmax = config["zlim_km"]
        vmin, vmax = config["zlim_km"]
        alpha = 0.8
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
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
            alpha=alpha,
        )
        # set ratio 1:1
        ax[0, 0].set_aspect("equal", "box")
        ax[0, 0].set_xlim([xmin, xmax])
        ax[0, 0].set_ylim([ymin, ymax])
        ax[0, 0].set_xlabel("X (km)")
        ax[0, 0].set_ylabel("Y (km)")
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
            alpha=alpha,
        )
        ax[0, 1].set_aspect("equal", "box")
        ax[0, 1].set_xlim([xmin, xmax])
        ax[0, 1].set_ylim([ymin, ymax])
        ax[0, 1].set_xlabel("X (km)")
        ax[0, 1].set_ylabel("Y (km)")
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
            alpha=alpha,
        )
        # ax[1, 0].set_aspect("equal", "box")
        ax[1, 0].set_xlim([xmin, xmax])
        ax[1, 0].set_ylim([zmax, zmin])
        ax[1, 0].set_xlabel("X (km)")
        # ax[1, 0].set_ylabel("Depth (km)")
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
            alpha=alpha,
        )
        # ax[1, 1].set_aspect("equal", "box")
        ax[1, 1].set_xlim([ymin, ymax])
        ax[1, 1].set_ylim([zmax, zmin])
        ax[1, 1].set_xlabel("Y (km)")
        # ax[1, 1].set_ylabel("Depth (km)")
        cbar = fig.colorbar(im, ax=ax[1, 1])
        cbar.set_label("Depth (km)")
        plt.savefig(os.path.join(figure_path, f"location_{iter}.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=config["eikonal"])
    event_init = np.array([[np.mean(config["xlim_km"]), np.mean(config["ylim_km"]), np.mean(config["zlim_km"]), 0.0]])

    # %%
    NCPU = mp.cpu_count()
    MAX_SST_ITER = 10
    # MIN_SST_S = 0.01
    iter = 0
    locations = None
    while iter < MAX_SST_ITER:
        picks, locations = invert_location(picks, stations, config, estimator, events_init=locations, iter=iter)
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
    picks.rename({"mask": "adloc_mask", "residual_s": "adloc_residual_s"}, axis=1, inplace=True)
    picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
    picks.drop(["idx_eve", "idx_sta"], axis=1, inplace=True, errors="ignore")
    locations[["longitude", "latitude"]] = locations.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    locations["depth_km"] = locations["z_km"]
    locations.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    stations.drop(["idx_sta", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    # stations.rename({"station_term": "adloc_station_term_s"}, axis=1, inplace=True)
    picks.sort_values(["phase_time"], inplace=True)
    locations.sort_values(["time"], inplace=True)
    picks.to_csv(os.path.join(result_path, "adloc_picks.csv"), index=False)
    locations.to_csv(os.path.join(result_path, "adloc_events.csv"), index=False)
    stations.to_csv(os.path.join(result_path, "adloc_stations.csv"), index=False)

# %%
