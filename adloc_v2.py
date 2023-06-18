# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pyproj import Proj
from torch import nn
import torch.optim as optim
from tqdm.auto import tqdm
import utils
from torch.utils.data import Dataset, DataLoader


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


class PhaseDataset:
    def __init__(self, picks, events, stations):
        self.picks = picks
        self.events = events
        self.stations = stations

    def __len__(self):
        return len(self.events)

    def __getitem__(self, i):
        phase_time = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_time"].values
        phase_score = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_score"].values
        phase_type = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]][
            "phase_type"
        ].values.tolist()
        event_index = np.array([i] * len(self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]))
        station_index = self.stations.loc[
            self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["station_id"], "index"
        ].values

        station_index = torch.tensor(station_index, dtype=torch.long)
        event_index = torch.tensor(event_index, dtype=torch.long)
        phase_weight = torch.tensor(phase_score, dtype=torch.float32)
        phase_time = torch.tensor(phase_time[:, np.newaxis], dtype=torch.float32)

        return {
            "event_index": event_index,
            "station_index": station_index,
            "phase_time": phase_time,
            "phase_weight": phase_weight,
            "phase_type": phase_type,
        }


class TravelTime(nn.Module):
    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt=None,
        event_loc=None,
        event_time=None,
        reg=0.001,
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_dt = nn.Embedding(num_station, 1)
        if station_dt is not None:
            self.station_dt.weight = torch.nn.Parameter(torch.tensor(station_dt, dtype=dtype))
        else:
            self.station_dt.weight = torch.nn.Parameter(torch.zeros(num_station, 1, dtype=dtype))
        self.register_buffer("station_loc", torch.tensor(station_loc, dtype=dtype))
        self.velocity = velocity
        self.reg = reg
        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype))
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(torch.tensor(event_time, dtype=dtype))

    def calc_time(self, event_loc, station_loc, phase_type):
        dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
        velocity = torch.ones(len(dist), 1)
        velocity[phase_type == "p"] = self.velocity["P"]
        velocity[phase_type == "s"] = self.velocity["S"]
        tt = dist / velocity

        return tt

    def forward(
        self, station_index, event_index=None, phase_type=None, phase_time=None, phase_weight=None, use_pair=False
    ):
        station_loc = self.station_loc[station_index]
        station_dt = self.station_dt(station_index)

        event_loc = self.event_loc(event_index)
        event_time = self.event_time(event_index)

        tt = self.calc_time(event_loc, station_loc, phase_type)
        t = event_time + tt + station_dt

        if use_pair:
            t = t[0] - t[1]

        if phase_time is None:
            loss = None
        else:
            # loss = torch.mean(phase_weight * (t - phase_time) ** 2)
            loss = torch.mean(F.huber_loss(t, phase_time, reduction="none") * phase_weight)
            loss += self.reg * torch.mean(
                torch.abs(station_dt)
            )  ## prevent the trade-off between station_dt and event_time

        return {"phase_time": t, "loss": loss}


def main(args):

    # %%
    data_path = Path("test_data")
    config = {
        "center": (-117.504, 35.705),
        "xlim_degree": [-118.004, -117.004],
        "ylim_degree": [35.205, 36.205],
        "degree2km": 111.19492474777779,
        "starttime": datetime(2019, 7, 4, 17, 0),
        "endtime": datetime(2019, 7, 5, 0, 0),
    }

    # %%
    stations = pd.read_csv(data_path / "stations.csv", delimiter="\t")
    picks = pd.read_csv(data_path / "picks_gamma.csv", delimiter="\t", parse_dates=["phase_time"])
    events = pd.read_csv(data_path / "catalog_gamma.csv", delimiter="\t", parse_dates=["time"])
    
    events = events[events["event_index"] < 1]
    picks = picks[picks["event_index"] < 1]

    # %%
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)
    starttime = events["time"].min()
    events["time"] = (events["time"] - starttime).dt.total_seconds()
    picks["phase_time"] = (picks["phase_time"] - starttime).dt.total_seconds()
    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth(m)"].apply(lambda x: x / 1e3)

    # %%
    num_event = len(events)
    num_station = len(stations)
    vp = 6.0
    vs = vp / 1.73

    stations.reset_index(inplace=True, drop=True)
    stations["index"] = stations.index.values
    stations.set_index("station", inplace=True)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    station_dt = None

    events.reset_index(inplace=True, drop=True)
    events["index"] = events.index.values
    event_loc = events[["x_km", "y_km", "z_km"]].values
    event_time = events["time"].values[:, np.newaxis]

    # %%
    plt.figure()
    plt.scatter(stations["x_km"], stations["y_km"], s=10, marker="^")
    plt.scatter(events["x_km"], events["y_km"], s=1)
    plt.axis("scaled")

    # %%

    # event_index = []
    # station_index = []
    # phase_score = []
    # phase_time = []
    # phase_type = []

    # for i in range(len(events)):
    #     phase_time.append(picks[picks["event_index"] == events.loc[i, "event_index"]]["phase_time"].values)
    #     phase_score.append(picks[picks["event_index"] == events.loc[i, "event_index"]]["phase_score"].values)
    #     phase_type.extend(picks[picks["event_index"] == events.loc[i, "event_index"]]["phase_type"].values.tolist())
    #     event_index.extend([i] * len(picks[picks["event_index"] == events.loc[i, "event_index"]]))
    #     station_index.append(stations.loc[
    #         picks[picks["event_index"] == events.loc[i, "event_index"]]["station_id"], "index"
    #     ].values)

    # phase_time = np.concatenate(phase_time)
    # phase_score = np.concatenate(phase_score)
    # event_index = np.array(event_index)
    # station_index = np.concatenate(station_index)

    # # %%
    # station_index = torch.tensor(station_index, dtype=torch.long)
    # event_index = torch.tensor(event_index, dtype=torch.long)
    # phase_weight = torch.tensor(phase_score, dtype=torch.float32)
    # phase_time = torch.tensor(phase_time[:, np.newaxis], dtype=torch.float32)

    utils.init_distributed_mode(args)
    print(args)

    phase_dataset = PhaseDataset(picks, events, stations)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(phase_dataset)

    data_loader = DataLoader(
        phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None
    )


    travel_time = TravelTime(num_event, num_station, station_loc, velocity={"P": vp, "S": vs})
    init_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    init_event_time = travel_time.event_time.weight.clone().detach().numpy()

    optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    # optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    
    epoch = 1
    for i in range(epoch):

        optimizer.zero_grad()

        for meta in tqdm(data_loader, desc=f"Epoch {i}"):

            station_index = meta["station_index"]
            event_index = meta["event_index"]
            phase_time = meta["phase_time"]
            phase_type = meta["phase_type"]
            phase_weight = meta["phase_weight"]

            def closure():
                loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
                loss.backward()
                return loss
            optimizer.step(closure)

            # loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
            # loss.backward()
        
        # optimizer.step(closure)
        # optimizer.step()

    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
    invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()

    plt.figure()
    # plt.scatter(station_loc[:,0], station_loc[:,1], c=tp[idx_event,:])
    plt.plot(event_loc[:, 0], event_loc[:, 1], "x", markersize=1, color="blue", label="True locations")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(init_event_loc[:, 0], init_event_loc[:, 1], "x", markersize=1, color="green", label="Initial locations")
    plt.plot(
        invert_event_loc[:, 0], invert_event_loc[:, 1], "x", markersize=1, color="red", label="Inverted locations"
    )
    plt.scatter(station_loc[:, 0], station_loc[:, 1], c=station_dt, marker="o", alpha=0.6)
    plt.scatter(station_loc[:, 0] + 1, station_loc[:, 1] + 1, c=invert_station_dt, marker="o", alpha=0.6)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis("scaled")
    plt.legend()
    plt.savefig("absolute_location.png", dpi=300)

    raise

    # %%
    # travel_time = TravelTime(
    #     num_event,
    #     num_station,
    #     station_loc,
    #     station_dt=station_dt,
    #     event_loc=event_loc,
    #     event_time=event_time,
    #     reg=0,
    #     velocity={"P": vp, "S": vs},
    # )

    # tt = travel_time(station_index, event_index, phase_type)["phase_time"]
    # print("True location: ", F.mse_loss(tt, phase_time))

    # %%
    # travel_time = TravelTime(num_event, num_station, station_loc, velocity={"P": vp, "S": vs})
    # tt = travel_time(station_index, event_index, phase_type)["phase_time"]
    # print("Initial loss", F.mse_loss(tt, phase_time))
    # init_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    # init_event_time = travel_time.event_time.weight.clone().detach().numpy()

    # %%
    # print(f"{station_index.shape = }, {event_index.shape = }, {phase_weight.shape = }, {phase_time.shape = }")
    # print(f"{len(phase_type) = }")

    # %%
    # optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe")

    # def closure():
    #     optimizer.zero_grad()
    #     loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
    #     loss.backward()
    #     return loss

    # optimizer.step(closure)

    # %%
    optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)

    batch_size = 1000
    epoch = 1000
    for i in tqdm(range(epoch)):
        optimizer.zero_grad()
        loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
        loss.backward()
        optimizer.step()

    tt = travel_time(station_index, event_index, phase_type)["phase_time"]
    print("Optimized loss", F.mse_loss(tt, phase_time))
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
    invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()

    # %%
    plt.figure()
    # plt.scatter(station_loc[:,0], station_loc[:,1], c=tp[idx_event,:])
    plt.plot(event_loc[:, 0], event_loc[:, 1], "x", markersize=1, color="blue", label="True locations")
    plt.plot(init_event_loc[:, 0], init_event_loc[:, 1], "x", markersize=1, color="green", label="Initial locations")
    plt.plot(
        invert_event_loc[:, 0], invert_event_loc[:, 1], "x", markersize=1, color="red", label="Inverted locations"
    )
    plt.scatter(station_loc[:, 0], station_loc[:, 1], c=station_dt, marker="o", alpha=0.6)
    plt.scatter(station_loc[:, 0] + 1, station_loc[:, 1] + 1, c=invert_station_dt, marker="o", alpha=0.6)
    # plt.xlim([0, xmax])
    # plt.ylim([0, xmax])
    plt.axis("scaled")
    plt.legend()
    plt.savefig("absolute_location.png", dpi=300)
    # plt.show()
    # %%


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
