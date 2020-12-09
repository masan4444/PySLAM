import numpy as np
import coordinate as coord

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


def _read_line_lsc(line):
    data = line.split()
    size = int(data[4])
    distances = []
    angles = []
    for i in range(size):
        angles.append(np.deg2rad(float(data[5 + 2 * i])))
        distances.append(float(data[6 + 2 * i]))
    return \
        np.array([
            float(data[-5]), \
            float(data[-4]), \
            float(data[-3])  \
        ]), \
        np.array(distances), \
        np.array(angles)


def read_line_lsc(line):
    (robot_r, distances, angles) = _read_line_lsc(line)
    sensor_r = coord.sensor2robot(distances, angles)
    return robot_r, sensor_r


def read_lsc_file(filepath):
    lines = open(filepath, "r").readlines()
    robot_rs = []
    sensor_rs = []
    for line in lines:
        robot_r, sensor_r = read_line_lsc(line)
        robot_rs.append(robot_r)
        sensor_rs.append(sensor_r)
    return np.array(robot_rs), sensor_rs
