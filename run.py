from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import utils
from adloc import PhaseDataset, TravelTime, initialize_eikonal

torch.manual_seed(0)
np.random.seed(0)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("-dd", "--double_difference", action="store_true", help="Use double difference")
    parser.add_argument("--eikonal", action="store_true", help="Use eikonal")
    parser.add_argument("--dd_weight", default=1.0, type=float, help="weight for double difference")
    parser.add_argument("--min_pair_dist", default=20.0, type=float, help="minimum distance between pairs")
    parser.add_argument("--max_time_res", default=0.5, type=float, help="maximum time residual")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="lbfgs", type=str, help="optimizer")
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


def main(args):
    # %%
    data_path = Path("tests")
    figure_path = Path("figures")
    figure_path.mkdir(exist_ok=True)

    ##TODO: clean up config
    config = {
        "center": (-117.504, 35.705),
        "xlim_degree": [-118.004, -117.004],
        "ylim_degree": [35.205, 36.205],
        "degree2km": 111.19492474777779,
        "starttime": datetime(2019, 7, 4, 17, 0),
        "endtime": datetime(2019, 7, 5, 0, 0),
    }

    ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 1.0
    vel = {"z": zz, "p": vp, "s": vs}
    config["x(km)"] = (
        (np.array(config["xlim_degree"]) - np.array(config["center"][0]))
        * config["degree2km"]
        * np.cos(np.deg2rad(config["center"][1]))
    )
    config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
    config["z(km)"] = (0, 20)
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    if args.eikonal:
        eikonal = initialize_eikonal(config["eikonal"])
    else:
        eikonal = None

    # %% TODO: Convert to args
    stations = pd.read_csv(data_path / "stations.csv")
    picks = pd.read_csv(data_path / "picks.csv", parse_dates=["phase_time"])
    events = pd.read_csv(data_path / "events.csv", parse_dates=["time"])

    # %%
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["depth_km"]
    # starttime = events["time"].min()
    # events["time"] = (events["time"] - starttime).dt.total_seconds()
    # picks["phase_time"] = (picks["phase_time"] - starttime).dt.total_seconds()
    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    # %%
    num_event = len(events)
    num_station = len(stations)
    vp = 6.0
    vs = vp / 1.73

    stations.reset_index(inplace=True, drop=True)
    stations["index"] = stations.index.values
    stations.set_index("station_id", inplace=True)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    station_dt = None

    events.reset_index(inplace=True, drop=True)
    events["index"] = events.index.values
    event_loc = events[["x_km", "y_km", "z_km"]].values
    event_time = events["time"].values

    event_index_map = {x: i for i, x in enumerate(events["event_index"])}
    picks = picks[picks["event_index"] != -1]
    picks["index"] = picks["event_index"].apply(lambda x: event_index_map[x])
    picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - event_time[x["index"]]).total_seconds(), axis=1)

    # %%
    utils.init_distributed_mode(args)
    print(args)

    phase_dataset = PhaseDataset(picks, events, stations, double_difference=False, config=args)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(phase_dataset)
    data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)

    if args.double_difference:
        phase_dataset_dd = PhaseDataset(picks, events, stations, double_difference=True, config=args)
        if args.distributed:
            sampler_dd = torch.utils.data.distributed.DistributedSampler(phase_dataset_dd, shuffle=False)
        else:
            sampler_dd = torch.utils.data.SequentialSampler(phase_dataset_dd)

        data_loader_dd = DataLoader(
            phase_dataset_dd, batch_size=None, sampler=sampler_dd, num_workers=args.workers, collate_fn=None
        )

    # %%

    travel_time = TravelTime(
        num_event,
        num_station,
        station_loc,
        # event_loc=event_loc,
        # event_time=event_time,
        velocity={"P": vp, "S": vs},
        eikonal=eikonal,
    )
    init_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    init_event_time = travel_time.event_time.weight.clone().detach().numpy()

    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    elif args.opt.lower() == "adam":
        optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    elif args.opt.lower() == "sgd":
        optimizer = optim.SGD(params=travel_time.parameters(), lr=10.0)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        prev_loss = 0
        for i in range(args.epochs):

            def closure():
                optimizer.zero_grad()
                for meta in data_loader:
                    station_index = meta["station_index"]
                    event_index = meta["event_index"]
                    phase_time = meta["phase_time"]
                    phase_type = meta["phase_type"]
                    phase_weight = meta["phase_weight"]
                    loss = travel_time(
                        station_index,
                        event_index,
                        phase_type,
                        phase_time,
                        phase_weight,
                        double_difference=False,
                    )["loss"]
                    if args.distributed:
                        dist.barrier()
                        dist.all_reduce(loss)
                    loss.backward()

                    if args.double_difference:
                        for meta in data_loader_dd:
                            station_index = meta["station_index"]
                            event_index = meta["event_index"]
                            phase_time = meta["phase_time"]
                            phase_type = meta["phase_type"]
                            phase_weight = meta["phase_weight"]

                            loss_dd = travel_time(
                                station_index,
                                event_index,
                                phase_type,
                                phase_time,
                                phase_weight,
                                double_difference=True,
                            )["loss"]
                            if args.distributed:
                                dist.barrier()
                                dist.all_reduce(loss_dd)

                            (loss_dd * args.dd_weight).backward()

                return loss

            optimizer.step(closure)

            loss = 0
            loss_dd = 0
            for meta in data_loader:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]
                loss += travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                    double_difference=False,
                )["loss"]
            if args.double_difference:
                for meta in data_loader_dd:
                    station_index = meta["station_index"]
                    event_index = meta["event_index"]
                    phase_time = meta["phase_time"]
                    phase_type = meta["phase_type"]
                    phase_weight = meta["phase_weight"]

                    loss_dd = travel_time(
                        station_index,
                        event_index,
                        phase_type,
                        phase_time,
                        phase_weight,
                        double_difference=True,
                    )["loss"]
            if abs((loss + loss_dd) - prev_loss) < 1e-6:
                break
            prev_loss = loss + loss_dd
            print(f"Loss: {loss+loss_dd}:  {loss} + {loss_dd}")

            # set variable range
            travel_time.event_loc.weight.data[:, 2] += (
                torch.randn_like(travel_time.event_loc.weight.data[:, 2]) * (args.epochs - i) / args.epochs
            )
            travel_time.event_loc.weight.data[:, 2].clamp_(min=config["z(km)"][0], max=config["z(km)"][1])

    else:
        for i in range(args.epochs):
            optimizer.zero_grad()

            prev_loss = 0
            loss = 0
            loss_dd = 0
            for meta in data_loader:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]

                loss = travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                    double_difference=False,
                )["loss"]
                loss.backward()

            if args.double_difference:
                for meta in data_loader_dd:
                    station_index = meta["station_index"]
                    event_index = meta["event_index"]
                    phase_time = meta["phase_time"]
                    phase_type = meta["phase_type"]
                    phase_weight = meta["phase_weight"]

                    loss_dd = travel_time(
                        station_index,
                        event_index,
                        phase_type,
                        phase_time,
                        phase_weight,
                        double_difference=True,
                    )["loss"]
                    (loss_dd * args.dd_weight).backward()

            optimizer.step()

            if abs((loss + loss_dd) - prev_loss) < 1e-3:
                print(f"Loss: {loss+loss_dd}:  {loss} + {loss_dd}")
                break
            prev_loss = loss + loss_dd

            if i % 100 == 0:
                print(f"Loss: {loss+loss_dd}:  {loss} + {loss_dd}")

            # set variable range
            travel_time.event_loc.weight.data[:, 2] += (
                torch.randn_like(travel_time.event_loc.weight.data[:, 2]) * (args.epochs - i) / args.epochs
            )
            travel_time.event_loc.weight.data[:, 2].clamp_(min=config["z(km)"][0], max=config["z(km)"][1])

    # %%
    station_dt = travel_time.station_dt.weight.clone().detach().numpy()
    print(f"station_dt: max = {np.max(station_dt)}, min = {np.min(station_dt)}, mean = {np.mean(station_dt)}")
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
    invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()

    # %%
    plt.figure()
    # plt.scatter(station_loc[:,0], station_loc[:,1], c=tp[idx_event,:])
    plt.plot(event_loc[:, 0], event_loc[:, 1], "x", markersize=1, color="blue", label="True locations")
    plt.scatter(station_loc[:, 0], station_loc[:, 1], c=station_dt[:, 0], marker="o", linewidths=0, alpha=0.6)
    plt.scatter(station_loc[:, 0], station_loc[:, 1] + 2, c=station_dt[:, 1], marker="o", linewidths=0, alpha=0.6)
    plt.axis("scaled")
    plt.colorbar()
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(init_event_loc[:, 0], init_event_loc[:, 1], "x", markersize=1, color="green", label="Initial locations")
    plt.plot(invert_event_loc[:, 0], invert_event_loc[:, 1], "x", markersize=1, color="red", label="Inverted locations")
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.legend()
    plt.savefig(figure_path / "invert_location_xy.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(event_loc[:, 0], event_loc[:, 2], "x", markersize=1, color="blue", label="True locations")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(init_event_loc[:, 0], init_event_loc[:, 2], "x", markersize=1, color="green", label="Initial locations")
    plt.plot(invert_event_loc[:, 0], invert_event_loc[:, 2], "x", markersize=1, color="red", label="Inverted locations")
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.legend()
    plt.savefig(figure_path / "invert_location_xz.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(event_loc[:, 1], event_loc[:, 2], "x", markersize=1, color="blue", label="True locations")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(init_event_loc[:, 1], init_event_loc[:, 2], "x", markersize=1, color="green", label="Initial locations")
    plt.plot(invert_event_loc[:, 1], invert_event_loc[:, 2], "x", markersize=1, color="red", label="Inverted locations")
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.legend()
    plt.savefig(figure_path / "invert_location_yz.png", dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    main(args)
