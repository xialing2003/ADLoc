# %%
import shelve
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from .eikonal2d import _interp, eikonal_solve, grad_traveltime, traveltime


class CalcTravelTime(Function):
    @staticmethod
    def forward(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h):
        tt = _interp(timetable, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        tt = torch.from_numpy(tt)
        return tt

    @staticmethod
    def setup_context(ctx, inputs, output):
        r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h = inputs
        ctx.save_for_backward(r, z)
        ctx.timetable = timetable
        ctx.timetable_grad_r = timetable_grad_r
        ctx.timetable_grad_z = timetable_grad_z
        ctx.rgrid0 = rgrid0
        ctx.zgrid0 = zgrid0
        ctx.nr = nr
        ctx.nz = nz
        ctx.h = h

    @staticmethod
    def backward(ctx, grad_output):
        timetable_grad_r = ctx.timetable_grad_r
        timetable_grad_z = ctx.timetable_grad_z
        rgrid0 = ctx.rgrid0
        zgrid0 = ctx.zgrid0
        nr = ctx.nr
        nz = ctx.nz
        h = ctx.h
        r, z = ctx.saved_tensors

        grad_r = _interp(timetable_grad_r, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        grad_z = _interp(timetable_grad_z, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)

        grad_r = torch.from_numpy(grad_r) * grad_output
        grad_z = torch.from_numpy(grad_z) * grad_output

        return grad_r, grad_z, None, None, None, None, None, None, None, None


# %%
class TravelTime(nn.Module):
    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt=None,
        event_loc=None,
        event_time=None,
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        eikonal=None,
        zlim=[0, 30],
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_loc = nn.Embedding(num_station, 3)
        self.station_dt = nn.Embedding(num_station, 1)  # same statioin term for P and S
        self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)
        if station_dt is not None:
            self.station_dt.weight = torch.nn.Parameter(torch.tensor(station_dt, dtype=dtype), requires_grad=False)
        else:
            self.station_dt.weight = torch.nn.Parameter(torch.zeros(num_station, 1, dtype=dtype), requires_grad=False)

        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(
                torch.tensor(event_loc, dtype=dtype).contiguous(), requires_grad=True
            )
        else:
            self.event_loc.weight = torch.nn.Parameter(
                torch.zeros(num_event, 3, dtype=dtype).contiguous(), requires_grad=True
            )
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(
                torch.tensor(event_time, dtype=dtype).contiguous(), requires_grad=True
            )
        else:
            self.event_time.weight = torch.nn.Parameter(
                torch.zeros(num_event, 1, dtype=dtype).contiguous(), requires_grad=True
            )

        self.velocity = [velocity["P"], velocity["S"]]
        self.eikonal = eikonal
        self.zlim = zlim

    def calc_time(self, event_loc, station_loc, phase_type):
        if self.eikonal is None:
            dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
            tt = dist / self.velocity[phase_type]
            tt = tt.float()
        else:
            r = torch.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)  ## nb, 2 (pair), 3
            z = event_loc[:, 2] - station_loc[:, 2]

            timetable = self.eikonal["up"] if phase_type == 0 else self.eikonal["us"]
            timetable_grad = self.eikonal["grad_up"] if phase_type == 0 else self.eikonal["grad_us"]
            timetable_grad_r = timetable_grad[0]
            timetable_grad_z = timetable_grad[1]
            rgrid0 = self.eikonal["rgrid"][0]
            zgrid0 = self.eikonal["zgrid"][0]
            nr = self.eikonal["nr"]
            nz = self.eikonal["nz"]
            h = self.eikonal["h"]
            tt = CalcTravelTime.apply(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h)

            tt = tt.float().unsqueeze(-1)

        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_time=None,
        phase_weight=None,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)
        resisudal = torch.zeros(len(phase_type), dtype=torch.float32)
        for type in [0, 1]:  # phase_type: 0 for P, 1 for S
            if len(phase_type[phase_type == type]) == 0:
                continue
            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            station_loc_ = self.station_loc(station_index_)  # (nb, 3)
            station_dt_ = self.station_dt(station_index_)  # (nb, 1)

            event_loc_ = self.event_loc(event_index_)  # (nb, 3)
            event_time_ = self.event_time(event_index_)  # (nb, 1)

            tt_ = self.calc_time(event_loc_, station_loc_, type)  # (nb, 1)

            t_ = event_time_ + tt_ + station_dt_  # (nb, 1)
            t_ = t_.squeeze(1)  # (nb, )

            pred_time[phase_type == type] = t_  # (nb, )

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]
                resisudal[phase_type == type] = phase_time_ - t_
                loss += torch.sum(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)

        return {"phase_time": pred_time, "residual_s": resisudal, "loss": loss}


# %%
class TravelTimeDD(nn.Module):
    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt=None,
        event_loc=None,
        event_time=None,
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        eikonal=None,
        zlim=[0, 30],
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_loc = nn.Embedding(num_station, 3)
        self.station_dt = nn.Embedding(num_station, 1)  # same statioin term for P and S
        self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)
        if station_dt is not None:
            self.station_dt.weight = torch.nn.Parameter(torch.tensor(station_dt, dtype=dtype), requires_grad=False)
        else:
            self.station_dt.weight = torch.nn.Parameter(torch.zeros(num_station, 1, dtype=dtype), requires_grad=False)

        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(
                torch.tensor(event_loc, dtype=dtype).contiguous(), requires_grad=True
            )
        else:
            self.event_loc.weight = torch.nn.Parameter(
                torch.zeros(num_event, 3, dtype=dtype).contiguous(), requires_grad=True
            )
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(
                torch.tensor(event_time, dtype=dtype).contiguous(), requires_grad=True
            )
        else:
            self.event_time.weight = torch.nn.Parameter(
                torch.zeros(num_event, 1, dtype=dtype).contiguous(), requires_grad=True
            )

        self.velocity = [velocity["P"], velocity["S"]]
        self.eikonal = eikonal
        self.zlim = zlim

    def calc_time(self, event_loc, station_loc, phase_type):
        if self.eikonal is None:
            dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
            tt = dist / self.velocity[phase_type]
            tt = tt.float()
        else:
            nb1, ne1, nc1 = event_loc.shape  # batch, event, xyz
            nb2, ne2, nc2 = station_loc.shape
            assert ne1 % ne2 == 0
            assert nb1 == nb2
            station_loc = torch.repeat_interleave(station_loc, ne1 // ne2, dim=1)
            event_loc = event_loc.view(nb1 * ne1, nc1)
            station_loc = station_loc.view(nb1 * ne1, nc2)

            r = torch.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)  ## nb, 2 (pair), 3
            z = event_loc[:, 2] - station_loc[:, 2]
            # z = torch.clamp(z, min=0, max=30)

            timetable = self.eikonal["up"] if phase_type == 0 else self.eikonal["us"]
            timetable_grad = self.eikonal["grad_up"] if phase_type == 0 else self.eikonal["grad_us"]
            timetable_grad_r = timetable_grad[0]
            timetable_grad_z = timetable_grad[1]
            rgrid0 = self.eikonal["rgrid"][0]
            zgrid0 = self.eikonal["zgrid"][0]
            nr = self.eikonal["nr"]
            nz = self.eikonal["nz"]
            h = self.eikonal["h"]
            tt = CalcTravelTime.apply(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h)
            tt = tt.float().view(nb1, ne1, 1)

        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_time=None,
        phase_weight=None,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)
        for type in [0, 1]:  # phase_type: 0 for P, 1 for S
            if len(phase_type[phase_type == type]) == 0:
                continue
            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            station_loc_ = self.station_loc(station_index_)  # (nb, 3)
            # station_dt_ = self.station_dt(station_index_)[:, [type]]  # (nb, 1)
            station_dt_ = self.station_dt(station_index_)  # (nb, 1)

            event_loc_ = self.event_loc(event_index_)  # (nb, 2, 3)
            event_time_ = self.event_time(event_index_)  # (nb, 2, 1)

            station_loc_ = station_loc_.unsqueeze(1)  # (nb, 1, 3)
            station_dt_ = station_dt_.unsqueeze(1)  # (nb, 1, 1)

            tt_ = self.calc_time(event_loc_, station_loc_, type)  # (nb, 2)

            t_ = event_time_ + tt_ + station_dt_  # (nb, 2, 1)

            t_ = t_[:, 0] - t_[:, 1]  # (nb, 1)
            t_ = t_.squeeze(1)  # (nb, )

            pred_time[phase_type == type] = t_  # (nb, )

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]
                loss += torch.sum(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)

        return {"phase_time": pred_time, "loss": loss}


class Test(nn.Module):
    def __init__(self, timetable, rgrid0, zgrid0, nr, nz, h):
        super().__init__()
        self.timetable = timetable
        self.rgrid0 = rgrid0
        self.zgrid0 = zgrid0
        self.nr = nr
        self.nz = nz
        self.h = h

    def forward(self, r, z):
        tt = CalcTravelTime.apply(r, z, self.timetable, self.rgrid0, self.zgrid0, self.nr, self.nz, self.h)
        return tt


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    starttime = time.time()
    rgrid0 = 0
    zgrid0 = 0
    nr0 = 20
    nz0 = 20
    h = 1
    r = rgrid0 + h * np.arange(0, nr0)
    z = zgrid0 + h * np.arange(0, nz0)
    r, z = np.meshgrid(r, z, indexing="ij")
    timetable = np.sqrt(r**2 + z**2)
    grad_r, grad_z = np.gradient(timetable, h, edge_order=2)
    # timetable = torch.from_numpy(timetable.flatten())
    timetable = timetable.flatten()

    nr = 1000
    nz = 1000
    r = torch.linspace(0, 20, nr)
    z = torch.linspace(0, 20, nz)
    r, z = torch.meshgrid(r, z, indexing="ij")
    r = r.flatten()
    z = z.flatten()

    test = Test(timetable, rgrid0, zgrid0, nr0, nz0, h)
    r.requires_grad = True
    z.requires_grad = True
    tt = test(r, z)
    tt.backward(torch.ones_like(tt))

    endtime = time.time()
    print(f"Time elapsed: {endtime - starttime} seconds.")
    tt = tt.detach().numpy()

    fig, ax = plt.subplots(3, 2)
    im = ax[0, 0].imshow(tt.reshape(nr, nz))
    fig.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(timetable.reshape(nr0, nz0))
    fig.colorbar(im, ax=ax[0, 1])
    im = ax[1, 0].imshow(r.grad.reshape(nr, nz))
    fig.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(grad_r.reshape(nr0, nz0))
    fig.colorbar(im, ax=ax[1, 1])
    im = ax[2, 0].imshow(z.grad.reshape(nr, nz))
    fig.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(grad_z.reshape(nr0, nz0))
    fig.colorbar(im, ax=ax[2, 1])
    plt.show()


# %%
