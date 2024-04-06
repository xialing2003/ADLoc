# %%
import argparse
import multiprocessing as mp
import os
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pairs")
    parser.add_argument("--stations", type=str, default="test_data/synthetic/stations.csv")
    parser.add_argument("--events", type=str, default="test_data/synthetic/events.csv")
    parser.add_argument("--picks", type=str, default="test_data/synthetic/picks.csv")
    parser.add_argument("--result_path", type=str, default="results/synthetic")
    return parser.parse_args()


# %%
def convert_dd(
    pairs,
    picks_by_event,
    min_obs=8,
    max_obs=20,
    i=0,
):

    station_index = []
    event_index1 = []
    event_index2 = []
    phase_type = []
    phase_score = []
    dd_time = []
    for idx1, idx2 in tqdm(pairs, desc=f"CPU {i}", position=i):
        picks1 = picks_by_event.get_group(idx1)
        picks2 = picks_by_event.get_group(idx2)

        common = picks1.merge(picks2, on=["idx_sta", "phase_type"], how="inner")
        if len(common) < min_obs:
            continue
        common["phase_score"] = (common["phase_score_x"] + common["phase_score_y"]) / 2.0
        common.sort_values("phase_score", ascending=False, inplace=True)
        common = common.head(max_obs)
        event_index1.extend(common["idx_eve_x"].values)
        event_index2.extend(common["idx_eve_y"].values)
        station_index.extend(common["idx_sta"].values)
        phase_type.extend(common["phase_type"].values)
        phase_score.extend(common["phase_score"].values)
        dd_time.extend(np.round(common["travel_time_x"].values - common["travel_time_y"].values, 5))

    return {
        "event_index1": event_index1,
        "event_index2": event_index2,
        "station_index": station_index,
        "phase_type": phase_type,
        "phase_score": phase_score,
        "dd_time": dd_time,
    }


# %%
if __name__ == "__main__":

    args = parse_args()
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %%
    MAX_PAIR_DIST = 10  # km
    MAX_NEIGHBORS = 500
    MIN_NEIGHBORS = 8
    MIN_OBS = 8
    MAX_OBS = 20
    mapping_phase_type_int = {"P": 0, "S": 1}

    # %%
    stations = pd.read_csv(args.stations)
    picks = pd.read_csv(args.picks, parse_dates=["phase_time"])
    events = pd.read_csv(args.events, parse_dates=["time"])

    picks = picks[picks["event_index"] != -1]
    # check phase_type is P/S or 0/1
    if set(picks["phase_type"].unique()).issubset(set(mapping_phase_type_int.keys())):  # P/S
        picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)

    # %%
    if "idx_eve" in events.columns:
        events = events.drop("idx_eve", axis=1)
    if "idx_sta" in stations.columns:
        stations = stations.drop("idx_sta", axis=1)
    if "idx_eve" in picks.columns:
        picks = picks.drop("idx_eve", axis=1)
    if "idx_sta" in picks.columns:
        picks = picks.drop("idx_sta", axis=1)

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
    stations["z_km"] = stations["depth_km"]
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

        pool = mp.Pool(NCPU)
        results = pool.starmap(
            convert_dd,
            [
                (
                    pairs[i::NCPU],
                    picks_by_event,
                    MIN_OBS,
                    MAX_OBS,
                    i,
                )
                for i in range(NCPU)
            ],
        )
        pool.close()
        pool.join()

        print("Collecting results")
        event_index1 = np.concatenate([r["event_index1"] for r in results])
        event_index2 = np.concatenate([r["event_index2"] for r in results])
        station_index = np.concatenate([r["station_index"] for r in results])
        phase_type = np.concatenate([r["phase_type"] for r in results])
        phase_score = np.concatenate([r["phase_score"] for r in results])
        dd_time = np.concatenate([r["dd_time"] for r in results])
        print(f"Saving to disk: {len(event_index1)} pairs")
        # np.savez_compressed(
        #     os.path.join(catalog_path, "adloc_dt.npz"),
        #     event_index1=event_index1,
        #     event_index2=event_index2,
        #     station_index=station_index,
        #     phase_type=phase_type,
        #     phase_score=phase_score,
        #     dd_time=dd_time,
        # )

        dtypes = np.dtype(
            [
                ("event_index1", np.int32),
                ("event_index2", np.int32),
                ("station_index", np.int32),
                ("phase_type", np.int32),
                ("phase_score", np.float32),
                ("dd_time", np.float32),
            ]
        )
        pairs_array = np.memmap(
            os.path.join(result_path, "adloc_dt.dat"),
            mode="w+",
            shape=(len(dd_time),),
            dtype=dtypes,
        )
        pairs_array["event_index1"] = event_index1
        pairs_array["event_index2"] = event_index2
        pairs_array["station_index"] = station_index
        pairs_array["phase_type"] = phase_type
        pairs_array["phase_score"] = phase_score
        pairs_array["dd_time"] = dd_time
        with open(os.path.join(result_path, "adloc_dtypes.pkl"), "wb") as f:
            pickle.dump(dtypes, f)

        events.to_csv(os.path.join(result_path, "adloc_events.csv"), index=False)
        stations.to_csv(os.path.join(result_path, "adloc_stations.csv"), index=False)
        picks.to_csv(os.path.join(result_path, "adloc_picks.csv"), index=False)

# %%
