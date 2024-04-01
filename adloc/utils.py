import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._ransac import RANSACRegressor


def invert(
    event_index,
    picks_by_event,
    estimator,
    events,
    stations,
    config,
    mask_idx,
    mask,
    error_idx,
    error_s,
    lock=nullcontext(),
):

    def is_model_valid(estimator, X, y):
        score = estimator.score(X, y)
        return score > config["min_score"]

    def is_data_valid(X, y):
        """
        X: idx_sta, type, score
        y: t_s
        """
        n0 = np.sum(X[:, 1] == 0)  # P
        n1 = np.sum(X[:, 1] == 1)  # S
        return n0 >= config["min_p_picks"] and n1 >= config["min_s_picks"]  # At least min P and S picks

    MIN_PICKS = config["min_picks"]
    MIN_PICKS_RATIO = config["min_picks_ratio"]
    MAX_RESIDUAL = config["max_residual_s"]

    X = picks_by_event.merge(
        stations[["x_km", "y_km", "z_km", "station_id", "station_term"]],
        # stations[["x_km", "y_km", "z_km", "station_id", "station_term_p", "station_term_s"]], ## Separate P and S station term
        on="station_id",
    )
    event_init = np.array([[np.median(X["x_km"]), np.median(X["y_km"]), np.mean(config["zlim_km"]), 0.0]])
    xstd = np.std(X["x_km"])
    ystd = np.std(X["y_km"])
    rstd = np.sqrt(xstd**2 + ystd**2)

    t0 = X["phase_time"].min()
    X.rename(columns={"phase_type": "type", "phase_score": "score", "phase_time": "t_s"}, inplace=True)
    X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X["t_s"] = X["t_s"] - X["station_term"]
    # X["t_s"] = X.apply(
    #     lambda x: x["t_s"] - x["station_term_p"] if x["type"] == 0 else x["t_s"] - x["station_term_s"], axis=1
    # ) ## Separate P and S station term
    X = X[
        [
            "idx_sta",
            "type",
            "score",
            "t_s",
        ]
    ]
    # X["type"] = X["type"].apply(lambda x: mapping_phase_type_int[x])

    estimator.set_params(**{"events": event_init})
    # ## Location using ADLoc
    # estimator.fit(X[["idx_sta", "type", "score"]].values, y=X["t_s"].values)
    # mask = np.ones(len(picks_by_event)).astype(bool)

    ## Location using RANSAC
    num_picks = len(picks_by_event)
    reg = RANSACRegressor(
        estimator=estimator,
        random_state=0,
        min_samples=max(MIN_PICKS, int(MIN_PICKS_RATIO * num_picks)),
        residual_threshold=MAX_RESIDUAL * (1.0 - np.exp(-rstd / 60.0)),  # not sure which one is better
        is_model_valid=is_model_valid,
        is_data_valid=is_data_valid,
    )

    try:
        reg.fit(X[["idx_sta", "type", "score"]].values, X["t_s"].values)
    except Exception as e:
        return f"No valid model for event_index {event_index}."
    estimator = reg.estimator_
    inlier_mask = reg.inlier_mask_

    ## Predict travel time
    tt = estimator.predict(X[["idx_sta", "type"]].values)
    score = estimator.score(X[["idx_sta", "type", "score"]].values[inlier_mask], y=X["t_s"].values[inlier_mask])

    if (np.sum(inlier_mask) > MIN_PICKS) and (score > config["min_score"]):
        mean_residual_s = np.sum(np.abs(X["t_s"].values - tt) * inlier_mask) / np.sum(inlier_mask)
        x, y, z, t = estimator.events[0]
        events.append(
            [event_index, x, y, z, t0 + pd.Timedelta(t, unit="s"), score, mean_residual_s, np.sum(inlier_mask)]
        )
    else:
        inlier_mask = np.zeros(len(inlier_mask), dtype=bool)

    with lock:
        error_idx.extend(picks_by_event.index.values)
        error_s.extend(X["t_s"].values - tt)
        mask_idx.extend(picks_by_event.index.values)
        mask.extend(inlier_mask.astype(int))


def invert_location(picks, events, stations, config, estimator, iter=0):

    if "ncpu" in config:
        NCPU = config["ncpu"]
    else:
        NCPU = mp.cpu_count() - 1

    with mp.Manager() as manager:
        mask_idx = manager.list()
        mask = manager.list()
        error_idx = manager.list()
        error_s = manager.list()
        locations = manager.list()
        lock = manager.Lock()
        pbar = tqdm(total=len(picks.groupby("idx_eve")), desc=f"Iter {iter}")
        threads = []
        with mp.get_context("spawn").Pool(NCPU) as pool:
            for event_index, picks_by_event in picks.groupby("idx_eve"):
                thread = pool.apply_async(
                    invert,
                    args=(
                        event_index,
                        picks_by_event,
                        estimator,
                        locations,
                        stations,
                        config,
                        mask_idx,
                        mask,
                        error_idx,
                        error_s,
                        lock,
                    ),
                    callback=lambda x: pbar.update(),
                )
                threads.append(thread)
            for thread in threads:
                out = thread.get()
                if out is not None:
                    print(out)
        mask_idx = list(mask_idx)
        mask = list(mask)
        error_idx = list(error_idx)
        error_s = list(error_s)
        locations = list(locations)
        pbar.close()

    picks.loc[mask_idx, "mask"] = mask
    picks.loc[error_idx, "residual_s"] = error_s

    locations = pd.DataFrame(
        locations, columns=["idx_eve", "x_km", "y_km", "z_km", "time", "adloc_score", "adloc_residual_s", "num_picks"]
    )
    locations = locations.merge(events[["event_index", "idx_eve"]], on="idx_eve")

    print(f"ADLoc using {len(picks[picks['mask'] == 1])} picks outof {len(picks)} picks")

    return picks, locations
