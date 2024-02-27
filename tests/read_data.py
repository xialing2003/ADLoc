# %%
import pandas as pd

# %%
picks = pd.read_csv("results/picks.csv", parse_dates=["phase_time"])

# %%
stations = pd.read_csv("results/stations.csv")
stations["index"] = stations.index # reindex in case the index does not start from 0 or is not continuous 

# %%
events = pd.read_csv("results/events.csv", parse_dates=["time"])
events["index"] = events.index # reindex in case the index does not start from 0 or is not continuous
# %%
picks = picks.merge(stations[["station_id", "index"]], on="station_id", suffixes=("", "_station"))
# %%
picks = picks.merge(events[["event_index", "index"]], on="event_index", suffixes=("", "_event"))
# %%
for index_station, picks_station in picks.groupby("index_station"):
    station_loc = stations.loc[index_station]
    event_loc = picks_station["index_event"]
    print(f"Station {station_loc['station_id']} at ({station_loc['latitude']}, {station_loc['longitude']})")
    for _, pick in picks_station.iterrows():
        event_loc = events.loc[pick["index_event"]]
        print(f"Event {event_loc['event_index']} at ({event_loc['latitude']}, {event_loc['longitude']})")
    raise
# %%
