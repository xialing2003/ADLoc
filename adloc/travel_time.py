# %%
import numpy as np
import torch
from numba import njit
from torch.autograd import Function
from torch import nn
import time


@njit
def _get_index(ir, iz, nr, nz, order="C"):
    if order == "C":
        return ir * nz + iz
    elif order == "F":
        return iz * nr + ir
    else:
        raise ValueError("order must be either C or F")


def test_get_index():
    vr, vz = np.meshgrid(np.arange(10), np.arange(20), indexing="ij")
    vr = vr.flatten()
    vz = vz.flatten()
    nr = 10
    nz = 20
    for ir in range(nr):
        for iz in range(nz):
            assert vr[_get_index(ir, iz, nr, nz)] == ir
            assert vz[_get_index(ir, iz, nr, nz)] == iz


@njit
def _interp(time_table, r, z, rgrid0, zgrid0, nr, nz, h):
    ir0 = np.floor((r - rgrid0) / h).clip(0, nr - 2).astype(np.int64)
    iz0 = np.floor((z - zgrid0) / h).clip(0, nz - 2).astype(np.int64)
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x1 = ir0 * h + rgrid0
    x2 = ir1 * h + rgrid0
    y1 = iz0 * h + zgrid0
    y2 = iz1 * h + zgrid0

    Q11 = time_table[_get_index(ir0, iz0, nr, nz)]
    Q12 = time_table[_get_index(ir0, iz1, nr, nz)]
    Q21 = time_table[_get_index(ir1, iz0, nr, nz)]
    Q22 = time_table[_get_index(ir1, iz1, nr, nz)]

    t = (
        1
        / (x2 - x1)
        / (y2 - y1)
        * (
            Q11 * (x2 - r) * (y2 - z)
            + Q21 * (r - x1) * (y2 - z)
            + Q12 * (x2 - r) * (z - y1)
            + Q22 * (r - x1) * (z - y1)
        )
    )

    return t


class TravelTime(Function):
    @staticmethod
    def forward(r, z, timetable, rgrid0, zgrid0, nr, nz, h):
        tt = _interp(timetable.numpy(), r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        tt = torch.from_numpy(tt)
        return tt

    @staticmethod
    def setup_context(ctx, inputs, output):
        r, z, timetable, rgrid0, zgrid0, nr, nz, h = inputs
        ctx.timetable = timetable
        ctx.rgrid0 = rgrid0
        ctx.zgrid0 = zgrid0
        ctx.nr = nr
        ctx.nz = nz
        ctx.h = h

    @staticmethod
    def backward(ctx, grad_output):
        timetable = ctx.timetable
        rgrid0 = ctx.rgrid0
        zgrid0 = ctx.zgrid0
        nr = ctx.nr
        nz = ctx.nz
        h = ctx.h

        grad_r = grad_z = grad_timetable = grad_rgrid0 = grad_zgrid0 = grad_nr = grad_nz = grad_h = None

        timetable = timetable.numpy().reshape(nr, nz)
        grad_time_r, grad_time_z = np.gradient(timetable, h, edge_order=2)
        grad_time_r = grad_time_r.flatten()
        grad_time_z = grad_time_z.flatten()
        grad_r = _interp(grad_time_r, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        grad_z = _interp(grad_time_z, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        grad_r = torch.from_numpy(grad_r)
        grad_z = torch.from_numpy(grad_z)

        return grad_r, grad_z, grad_timetable, grad_rgrid0, grad_zgrid0, grad_nr, grad_nz, grad_h


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
        tt = TravelTime.apply(r, z, self.timetable, self.rgrid0, self.zgrid0, self.nr, self.nz, self.h)
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
    timetalbe = np.sqrt(r**2 + z**2)
    timetable = torch.from_numpy(timetalbe.flatten())
    grad_r, grad_z = np.gradient(timetalbe, h, edge_order=2)

    nr = 10000
    nz = 10000
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
