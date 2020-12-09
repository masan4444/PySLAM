# %%
import numpy as np
from fxpmath import Fxp
from fxpmath import sum as sumf
import matplotlib.pyplot as plt

import coordinate as coord
import utility as util
import numpyFxp as npf

%load_ext autoreload


# %%
%autoreload


def info(n_word, n_frac):
    print(Fxp(None, True, n_word, n_frac).info(verbose=3))


# class MyFxp(Fxp):
#     def __init__(self, *args, **kwords):
#         super(MyFxp, self).__init__(*args, **kwords)
#         self.ndim = self().ndim
#         self.shape = self().shape


# %%
%autoreload


def closest_points_index_fxp(sensor_r, robot_r, reference_scan):
    scan = coord.robot2map_fxp(sensor_r, robot_r)
    euclidean_distance = np.linalg.norm((scan[..., np.newaxis, :, :] - reference_scan[..., np.newaxis, :])(), axis=-1)
    return np.argmin(euclidean_distance, axis=-2)


def cost_function_fxp(scan, robot_r, reference_scan):
    current_scan = coord.robot2map_fxp(scan, robot_r)
    cost = npf.square_sum(current_scan - reference_scan) / Fxp(reference_scan().shape[:-1])
    cost = npf.format(cost, n_word=16, n_int=1)
    # print('cost: {}', cost.dtype)
    return cost


def differential_fxp(scan, robot_r, reference_scan):
    robot_rotation = coord.rotation_matrix_fxp(robot_r, n_word=16, n_int=1)
    print(robot_rotation.dtype)
    rotated_scan = npf.matmul(scan, robot_rotation)
    rotated_scan = npf.format(rotated_scan, n_word=16, n_int=3)
    linear_trans_vector = robot_r[..., np.newaxis, 0:2] - reference_scan[..., 0:2]
    linear_trans_vector = npf.format(linear_trans_vector, n_word=16, n_int=3)

    # print(rotated_scan.info())
    # print(linear_trans_vector.dtype)

    dxy = rotated_scan + linear_trans_vector
    dt = npf.cross(rotated_scan, linear_trans_vector)
    # dxy = Fxp(dxy).like(Fxp(None, n_word=32, n_frac=28))
    # dt = Fxp(dt).like(Fxp(None, n_word=32, n_frac=28))
    d = npf.concatenate([dxy, dt[..., np.newaxis]], -1)
    d = npf.format(d, n_word=16, n_int=0)
    # print('d: {}', d.dtype)
    return npf.sum(d, axis=-2) / Fxp(reference_scan().shape[:-1])


def optimize_fxp(scan, robot_r, reference_scan, closest_index):
    _k = 0.1
    # _k = 0.1
    k = Fxp([_k, _k, _k / 10], n_word=16, n_int=1)

    cost_min = Fxp(9999, n_word=16)
    cost_old = cost_function_fxp(scan, robot_r, reference_scan[closest_index])
    print(cost_old)
    robot_r_best = robot_r
    count = 0
    while True:
        dd = differential_fxp(scan, robot_r, reference_scan[closest_index])
        print(dd)
        dd = npf.format(dd, n_word=16, n_int=3)
        robot_r = robot_r - k * dd
        robot_r = npf.format(robot_r, n_word=16, n_int=3)
        cost_new = cost_function_fxp(scan, robot_r, reference_scan[closest_index])
        count += 1
        # print(cost_old - cost_new)
        print(cost_new)

        if cost_min > cost_new:
            cost_min = cost_new
            robot_r_best = robot_r
#         if np.abs(cost_old - cost_new) < 0.0001:
        if np.abs((cost_old - cost_new)()) < 0.0000001:
            break

        cost_old = cost_new

    return robot_r_best, cost_min, count


def icp_fxp(scan, robot_r, reference_scan):
    cost_min = Fxp(9999, n_word=16)
    cost_old = cost_min
    robot_r_best = robot_r

    optimize_counts = []
    icp_count = 0

    while True:
        # print(icp_count)
        closest_index = closest_points_index_fxp(scan, robot_r, reference_scan)

        robot_r, cost_new, optimize_count = optimize_fxp(scan, robot_r, reference_scan, closest_index)
        if cost_min > cost_new:
            cost_min = cost_new
            robot_r_best = robot_r
#         if np.abs(cost_old - cost_new) < 0.0001:
        if np.abs((cost_old - cost_new)()) < 0.0000001:
            break
        cost_old = cost_new

        print(optimize_count)
        optimize_counts.append(optimize_count)
        icp_count += 1

    return robot_r_best, icp_count, np.array(optimize_counts)


def plot_scan(sub, scan, robot_r=None, s=1):
    sub.scatter(scan[..., 0], scan[..., 1], s=s)
    if robot_r is not None:
        sub.scatter(robot_r[..., 0], robot_r[..., 1], s=s)


def plot_scans(sub, scans):
    # for (scan, robot_r) in enumerate(zip(scans)):
    for scan in scans:
        plot_scan(sub, scan)


def padding(counts):
    counts = list(map(lambda x: x.tolist(), counts.tolist()))
    ll = max(map(len, counts))
    counts = list(map(lambda x: x + [0] * (ll - len(x)), counts))
    return np.array(counts)

%autoreload
odometries, sensor_rs = util.read_lsc_file(filepath="dataset/circle2.lsc")
odometries = Fxp(odometries, n_word=16, n_int=3)
sensor_rs = [Fxp(sensor_r, n_word=16, n_int=3) for sensor_r in sensor_rs]
# xy_odometries = np.round(scale * odometries[..., :2])
# odometries = np.concatenate([xy_odometries, odometries[..., np.newaxis, 2]], axis=-1).astype(np.int32)
# sensor_rs = [np.round(scale * sensor_r).astype(np.int32) for sensor_r in sensor_rs]


%autoreload
robot_rs = []
reference_scans = []
optimize_counts = []
start_i = 0
reference_scan = coord.robot2map_fxp(sensor_rs[start_i], odometries[start_i])
sub = plt.subplot()
sub.set_aspect('equal')
sub.set_xlabel('x')
sub.set_ylabel('y')
# plot_scan(sub, sensor_rs[start_i](), odometries[start_i](), s=1)


# for i, (odometory, sensor_r) in enumerate(zip(odometries[start_i + 1:], sensor_rs[start_i + 1:])):
#     if i > 10:
#         break
#     reference_scans.append(reference_scan)

#     robot_r, icp_count, optimize_count = icp_fxp(sensor_r, odometory, reference_scan)
#     robot_rs.append(robot_r)
#     optimize_counts.append(optimize_count)
#     # print(robot_r)
#     print(i, np.sum(optimize_count), optimize_count)
#     reference_scan = coord.robot2map_fxp(sensor_r, robot_r)

#     # if i % 100 == 0:
#     #     plot_scan(sub, reference_scan, robot_r, s=1)
#     plot_scan(sub, reference_scan(), robot_r(), s=1)
# reference_scans.append(reference_scan)

# np.array(reference_scans).shape


odometory = odometries[start_i]
sensor_r = coord.robot2map_fxp(sensor_rs[start_i], Fxp([0.1, 0.1, np.pi/8], n_word=16, n_int=3))
plot_scan(sub, sensor_r(), odometory(), s=1)
robot_r, icp_count, optimize_count = icp_fxp(sensor_r, odometory, reference_scan)
robot_rs.append(robot_r)
optimize_counts.append(optimize_count)
print(np.sum(optimize_count), optimize_count)
reference_scan = coord.robot2map_fxp(sensor_r, robot_r)
plot_scan(sub, reference_scan(), robot_r(), s=1)

# # reference_scans.append(reference_scan)
