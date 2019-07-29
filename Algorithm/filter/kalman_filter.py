# -*- coding: utf-8 -*-
'''


# @Time    : 2018/12/10 16:04
# @Author  : SJQ
# @File    : kalman_filter.py
'''
import pandas as pd
import datetime
import operator
import numpy as np
import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise


def gen_obseration():
    obseration_with_n = []
    obseration = []

    for i in range(9):
        point = (i + 2 + random.uniform(-0.5, 0.5), 2 + random.uniform(-0.5, 0.5))
        obseration_with_n.append(point)
        p = (i + 2, 2)
        obseration.append(p)
    for i in range(14):
        point = (10 + random.uniform(-0.5, 0.5), i + 3 + random.uniform(-0.5, 0.5))
        obseration_with_n.append(point)
        p = (10, i + 3)
        obseration.append(p)
    for i in range(8):
        point = (9 - i + random.uniform(-0.5, 0.5), 16 + random.uniform(-0.5, 0.5))
        obseration_with_n.append(point)
        p = (9 - i, 16)
        obseration.append(p)
    for i in range(13):
        point = (2 + random.uniform(-0.5, 0.5), 15 - i + random.uniform(-0.5, 0.5))
        obseration_with_n.append(point)
        p = (2, 15 - i)
        obseration.append(p)
    return obseration, obseration_with_n


def gen_R_and_Q(R_std, Q_std):
    R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=1.0, var=Q_std**2)
    Q = block_diag(q, q)
    return R, Q

def tracker():
    R_std = 1
    Q_std = 0.05
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                         [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker


if __name__ == '__main__':
    points, points_with_n = gen_obseration()
    circle_tracker = tracker()
    # zs = np.array(points)
    zs_with_n = np.array(points_with_n)
    # mu, _, _, _ = circle_tracker.batch_filter(zs)
    mu_with_n, _, _, _ = circle_tracker.batch_filter(zs_with_n)

    plt.cla()
    plt.clf()
    plt.xlim([0., 13.])
    plt.ylim([0., 18.])
    plt.grid(True)
    for point in points:
        plt.plot(point[0], point[1], color='red', marker='.')
        circle_tracker.predict()
        z = np.array(point).reshape(2, 1)
        circle_tracker.update(z)
        update_point = circle_tracker.x
        plt.plot(update_point[0], update_point[2], color='green', marker='.')
    plt.show()

    plt.cla()
    plt.clf()
    plt.figure(2)
    plt.xlim([0., 13.])
    plt.ylim([0., 18.])
    plt.grid(True)
    for point in points_with_n:
        plt.plot(point[0], point[1], color='red', marker='.')
    for z in mu_with_n:
        x = z[0, 0]
        y = z[2, 0]
        plt.plot(x, y, color='green', marker='.')
    plt.show()

    # use ukf


