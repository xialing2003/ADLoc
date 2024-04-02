# %%
import multiprocessing as mp
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# %%
def convert_dd(
    pairs,
    picks_by_event,
    event_index1,
    event_index2,
    station_index,
    phase_type,
    phase_score,
    dd_time,
    min_obs=8,
    max_obs=20,
    i=0,
    lock=nullcontext(),
):

    station_index_ = []
    event_index1_ = []
    event_index2_ = []
    phase_type_ = []
    phase_score_ = []
    dd_time_ = []
    for idx1, idx2 in tqdm(pairs, desc=f"CPU {i}", position=i):
        picks1 = picks_by_event.get_group(idx1)
        picks2 = picks_by_event.get_group(idx2)

        common = picks1.merge(picks2, on=["idx_sta", "phase_type"], how="inner")
        if len(common) < min_obs:
            continue
        common["phase_score"] = (common["phase_score_x"] + common["phase_score_y"]) / 2.0
        common.sort_values("phase_score", ascending=False, inplace=True)
        common = common.head(max_obs)
        event_index1_.extend(common["idx_eve_x"].values)
        event_index2_.extend(common["idx_eve_y"].values)
        station_index_.extend(common["idx_sta"].values)
        phase_type_.extend(common["phase_type"].values)
        phase_score_.extend(common["phase_score"].values)
        dd_time_.extend(np.round(common["travel_time_x"].values - common["travel_time_y"].values, 5))

    with lock:
        event_index1.extend(event_index1_)
        event_index2.extend(event_index2_)
        station_index.extend(station_index_)
        phase_type.extend(phase_type_)
        phase_score.extend(phase_score_)
        dd_time.extend(dd_time_)


# %%
if __name__ == "__main__":

    # %%
    MAX_PAIR_DIST = 10  # km
    MAX_NEIGHBORS = 500
    MIN_NEIGHBORS = 8
    MIN_OBS = 8
    MAX_OBS = 20

    # %%
    # data_path = "../tests/results/"
    # catalog_path = "../tests/results/"
    data_path = "test_data/ridgecrest"
    catalog_path = "results/ridgecrest"
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    # %%
    # picks = pd.read_csv(os.path.join(data_path, "picks.csv"), parse_dates=["phase_time"])
    # events = pd.read_csv(os.path.join(data_path, "events.csv"), parse_dates=["time"])
    picks = pd.read_csv(os.path.join(catalog_path, "adloc_picks.csv"), parse_dates=["phase_time"])
    events = pd.read_csv(os.path.join(catalog_path, "adloc_events.csv"), parse_dates=["time"])

    picks = picks[picks["event_index"] != -1]

    # %%
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["depth_km"] = -stations["elevation_m"] / 1000
    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    picks["travel_time"] = picks.apply(
        lambda x: (x["phase_time"] - events.loc[x["idx_eve"], "time"]).total_seconds(), axis=1
    )

    # %%
    picks_by_event = picks.groupby("idx_eve")

    # Option 1:
    neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    pairs = set()
    neigh_ind = neigh.radius_neighbors(sort_results=True)[1]
    for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
        if len(neighs) < MIN_NEIGHBORS:
            continue
        for j in neighs[:MAX_NEIGHBORS]:
            if i > j:
                pairs.add((j, i))
            else:
                pairs.add((i, j))
    pairs = list(pairs)

    # Option 2:
    # neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    # neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    # pairs = set()
    # neigh_ind = neigh.radius_neighbors()[1]
    # for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
    #     if len(neighs) < MIN_NEIGHBORS:
    #         continue
    #     neighs = neighs[np.argsort(events.loc[neighs, "num_picks"])]  ## TODO: check if useful
    #     for j in neighs[:MAX_NEIGHBORS]:
    #         if i > j:
    #             pairs.add((j, i))
    #         else:
    #             pairs.add((i, j))
    # pairs = list(pairs)

    # %%
    NCPU = mp.cpu_count()
    with mp.Manager() as manager:
        event_index1 = manager.list()
        event_index2 = manager.list()
        station_index = manager.list()
        phase_type = manager.list()
        phase_score = manager.list()
        dd_time = manager.list()
        lock = manager.Lock()

        pool = mp.Pool(NCPU)
        pool.starmap(
            convert_dd,
            [
                (
                    pairs[i::NCPU],
                    picks_by_event,
                    event_index1,
                    event_index2,
                    station_index,
                    phase_type,
                    phase_score,
                    dd_time,
                    MIN_OBS,
                    MAX_OBS,
                    i,
                    lock,
                )
                for i in range(NCPU)
            ],
        )
        pool.close()
        pool.join()

        print("Collecting results")
        event_index1 = np.array(event_index1, dtype=int)
        event_index2 = np.array(event_index2, dtype=int)
        station_index = np.array(station_index, dtype=int)
        phase_type = np.array(phase_type, dtype=str)
        phase_score = np.array(phase_score, dtype=float)
        dd_time = np.array(dd_time, dtype=float)
        print(f"Saving to disk: {len(event_index1)} pairs")
        np.savez_compressed(
            os.path.join(catalog_path, "adloc_dt.npz"),
            event_index1=event_index1,
            event_index2=event_index2,
            station_index=station_index,
            phase_type=phase_type,
            phase_score=phase_score,
            dd_time=dd_time,
        )

# %%
