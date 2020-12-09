import numpy as np
from fxpmath import Fxp
import numpyFxp as npf

SENSOR_OFFSET = np.pi  # rad
# SENSOR_OFFSET = 0  # rad

"""
データ形式

r := (x, y, t)のとき

    r = np.array([
        [x0, y0, t0],
        [x1, y1, t1],
        ,
        ,
    ])
    r.shape => (size, 3)

*注意事項
    matmulに注意せよ
"""


def rotation_matrix(robot_r):
    return np.array([
        [np.cos(robot_r[..., 2]), -np.sin(robot_r[..., 2])],
        [np.sin(robot_r[..., 2]), np.cos(robot_r[..., 2])]
    ]).transpose(tuple(range(2, robot_r.ndim + 1)) + (1, 0))


def sensor2robot(distance, angle):
    return (distance * np.array([ \
            np.cos(angle - SENSOR_OFFSET), \
            np.sin(angle - SENSOR_OFFSET)  \
        ])).T


def robot2map(r, robot_r):
    robot_rotation = rotation_matrix(robot_r)
    return (np.matmul(r, robot_rotation) + robot_r[..., np.newaxis, 0:2])


def sensor2map(distance, angle, robot_r):
    r = sensor2robot(distance, angle)
    return robot2map(r, robot_r)


# for fixed point
# def sensor2robot_fxp(distance, angle):
#     return (distance * np.array([ \
#             np.cos(angle - SENSOR_OFFSET), \
#             np.sin(angle - SENSOR_OFFSET)  \
#         ])).T
def rotation_matrix_fxp(robot_r, n_word=None, n_int=None):
    theta = robot_r[..., 2]()
    robot_rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]).transpose(tuple(range(2, robot_r().ndim + 1)) + (1, 0))
    return Fxp(robot_rotation, n_word=n_word, n_int=n_int)


def robot2map_fxp(r, robot_r):
    robot_rotation = rotation_matrix_fxp(robot_r, n_word=16, n_int=1)
    rotated = npf.matmul(r, robot_rotation)
    rotated = npf.format(rotated, n_word=16, n_int=3)
    return rotated + robot_r[..., np.newaxis, 0:2]


# def sensor2map_fxp(distance, angle, robot_r):
#     r = sensor2robot(distance, angle)
#     return robot2map(r, robot_r)
