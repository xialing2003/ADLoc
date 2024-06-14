import numpy as np
import h5py
from tqdm import tqdm
import sys
import os
np.random.seed(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../adloc'))
sys.path.append(project_path)
from tests.reloc import init_eikonal3d, ADLoc

if __name__ == "__main__":

    import json
    import pandas as pd

    data_path = "../examples/checkerboard"
    events = pd.read_csv(os.path.join(data_path, "events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    picks = pd.read_csv(os.path.join(data_path, "picks.csv"), parse_dates=["phase_time"])
    with open(os.path.join(data_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    mapping_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_int[k]: v for k, v in config["vel"].items()}

    eikonal = init_eikonal3d(config, stations, data_path)

    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous

    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    
    num_event = len(events)
    # est_loc = pd.DataFrame(columns=['x', 'y', 'z']); tru_loc = pd.DataFrame(columns=['x', 'y', 'z'])
    
    cal_p = np.zeros((100,50)); cal_s = np.zeros((100,50))
    for event_index in range(num_event):
        
        event_loc = events.loc[event_index]
        picks_event = picks[picks["idx_eve"] == event_index]

        X = picks_event.merge(stations[["x_km", "y_km", "z_km", "station_id"]], on="station_id")
        X.rename(columns={"phase_type": "type", "phase_time": "t_s"}, inplace=True)
        t0 = X["t_s"].min()
        X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
        X = X[["x_km", "y_km", "z_km", "t_s", "type", "idx_sta"]]
        mapping_int = {"P": 0, "S": 1}
        X["type"] = X["type"].apply(lambda x: mapping_int[x.upper()])

        event_input = np.ones((num_event,4))
        event_input[event_index, :3]=event_loc[['x_km', 'y_km', 'z_km']].values
        del_t = (events.time[event_index]-t0).total_seconds()
        event_input[event_index, 3] = del_t
        # print(del_t)

        estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, num_event=num_event, eikonal=eikonal, events=event_input)
        tt = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
        estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values, event_index=event_index)

        
        tt -= del_t
        
        cal_p[event_index] = tt[::2]
        cal_s[event_index] = tt[1::2]
    
    with h5py.File(data_path + '/cal_p.h5', 'w') as hf:
        hf.create_dataset('data', data=cal_p)
    with h5py.File(data_path + 'cal_s.h5', 'w') as hf:
        hf.create_dataset('data', data=cal_s)