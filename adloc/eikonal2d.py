import itertools
import shelve
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numba import njit
from numba.typed import List

np.random.seed(0)

###################################### Eikonal Solver ######################################

# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit
def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min([a, b]) + f * h
    else:
        return (a + b + np.sqrt(2.0 * f * f * h * h - (a - b) ** 2)) / 2.0


@njit
def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    for i in I:
        for j in J:
            if i == 0:
                uxmin = u[i + 1, j]
            elif i == m - 1:
                uxmin = u[i - 1, j]
            else:
                uxmin = min([u[i - 1, j], u[i + 1, j]])

            if j == 0:
                uymin = u[i, j + 1]
            elif j == n - 1:
                uymin = u[i, j - 1]
            else:
                uymin = min([u[i, j - 1], u[i, j + 1]])

            u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

            u[i, j] = min([u_new, u[i, j]])

    return u


@njit
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n = u.shape
    I = np.arange(m)
    iI = I[::-1]
    J = np.arange(n)
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, v, h):
    print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, v, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iter {i}, error = {err:.3f}")
        if err < 1e-6:
            break
    print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
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
    vr = vr.ravel()
    vz = vz.ravel()
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
    r0 = ir0 * h + rgrid0
    r1 = ir1 * h + rgrid0
    z0 = iz0 * h + zgrid0
    z1 = iz1 * h + zgrid0

    Q00 = time_table[_get_index(ir0, iz0, nr, nz)]
    Q01 = time_table[_get_index(ir0, iz1, nr, nz)]
    Q10 = time_table[_get_index(ir1, iz0, nr, nz)]
    Q11 = time_table[_get_index(ir1, iz1, nr, nz)]
    # Q00 = time_table[ir0, iz0]
    # Q01 = time_table[ir0, iz1]
    # Q10 = time_table[ir1, iz0]
    # Q11 = time_table[ir1, iz1]

    t = (
        1.0
        / (r1 - r0)
        / (z1 - z0)
        * (
            Q00 * (r1 - r) * (z1 - z)
            + Q10 * (r - r0) * (z1 - z)
            + Q01 * (r1 - r) * (z - z0)
            + Q11 * (r - r0) * (z - z0)
        )
    )

    return t


def traveltime(event_loc, station_loc, phase_type, eikonal):
    r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
    z = event_loc[:, 2] - station_loc[:, 2]

    rgrid0 = eikonal["rgrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nr = eikonal["nr"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    p_index = phase_type == "p"
    s_index = phase_type == "s"
    tt = np.zeros(len(phase_type), dtype=np.float32)
    tt[p_index] = _interp(eikonal["up"], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    tt[s_index] = _interp(eikonal["us"], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    return tt


def grad_traveltime(event_loc, station_loc, phase_type, eikonal):
    r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
    z = event_loc[:, 2] - station_loc[:, 2]

    rgrid0 = eikonal["rgrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nr = eikonal["nr"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    p_index = phase_type == "p"
    s_index = phase_type == "s"
    dt_dr = np.zeros(len(phase_type))
    dt_dz = np.zeros(len(phase_type))
    dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    dr_dxy = (event_loc[:, :2] - station_loc[:, :2]) / (r[:, np.newaxis] + 1e-6)
    dt_dxy = dt_dr[:, np.newaxis] * dr_dxy

    grad = np.column_stack((dt_dxy, dt_dz[:, np.newaxis]))

    return grad


if __name__ == "__main__":

    nr = 21
    nz = 21
    vel = {"p": 6.0, "s": 6.0 / 1.73}
    vp = np.ones((nr, nz)) * vel["p"]
    vs = np.ones((nr, nz)) * vel["s"]
    h = 10.0

    up = 1000 * np.ones((nr, nz))
    # up[nr//2, nz//2] = 0.0
    up[0, 0] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = [x.ravel() for x in grad_up]

    us = 1000 * np.ones((nr, nz))
    # us[nr//2, nz//2] = 0.0
    us[0, 0] = 0.1

    us = eikonal_solve(us, vs, h)
    grad_us = np.gradient(us, h, edge_order=2)
    us = us.ravel()
    grad_us = [x.ravel() for x in grad_us]

    num_event = 10
    event_loc = np.random.rand(num_event, 3) * np.array([nr * h / np.sqrt(2), nr * h / np.sqrt(2), nz * h])
    print(f"{event_loc = }")
    # event_loc = np.round(event_loc, 0)
    # station_loc = np.random.rand(1, 3) * np.array([nr*h/np.sqrt(2), nr*h/np.sqrt(2), 0])
    station_loc = np.array([0, 0, 0])
    print(f"{station_loc = }")
    station_loc = np.tile(station_loc, (num_event, 1))
    # phase_type = np.random.choice(["p", "s"], num_event, replace=True)
    # print(f"{list(phase_type) = }")
    phase_type = np.array(["p"] * (num_event // 2) + ["s"] * (num_event - num_event // 2))
    v = np.array([vel[x] for x in phase_type])
    t = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / v
    grad_t = (
        (event_loc - station_loc) / np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True) / v[:, np.newaxis]
    )
    print(f"True traveltime: {t = }")
    print(f"True grad traveltime: {grad_t = }")

    # tp = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["p"]
    # print(f"{tp = }")
    # ts = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["s"]
    # print(f"{ts = }")

    config = {
        "up": up,
        "us": us,
        "grad_up": grad_up,
        "grad_us": grad_us,
        "rgrid": np.arange(nr) * h,
        "zgrid": np.arange(nz) * h,
        "nr": nr,
        "nz": nz,
        "h": h,
    }
    t = traveltime(event_loc, station_loc, phase_type, config)
    grad_t = grad_traveltime(event_loc, station_loc, phase_type, config)
    print(f"Computed traveltime: {t = }")
    print(f"Computed grad traveltime: {grad_t = }")

    up = up.reshape((nr, nz))
    plt.figure()
    plt.pcolormesh(up[:, :])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig("slice_tp_2d.png")

    us = us.reshape((nr, nz))
    plt.figure()
    plt.pcolormesh(us[:, :])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig("slice_ts_2d.png")

    grad_up = [x.reshape((nr, nz)) for x in grad_up]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cax0 = ax[0].pcolormesh(grad_up[0][:, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_tp_x")
    cax1 = ax[1].pcolormesh(grad_up[1][:, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_tp_z")
    plt.savefig("slice_grad_tp_2d.png")

    grad_us = [x.reshape((nr, nz)) for x in grad_us]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cax0 = ax[0].pcolormesh(grad_us[0][:, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_ts_x")
    cax1 = ax[1].pcolormesh(grad_us[1][:, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_ts_z")
    plt.savefig("slice_grad_ts_2d.png")
