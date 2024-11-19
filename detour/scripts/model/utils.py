from functools import reduce
import numpy as np
import bisect
import torch
from torch.autograd import Variable

def get_init_pose(index):
    # x y z angle
    init_pose_list = [
        [-12, 3, 0.2, 0], [-12, 12, 0.2, 0], [-12, -12, 0.2, 0],
        [0, 10, 0.2, 0], [-4, 0, 0.2, 0], [-1, -12, 0.2, 0],
        [12,11, 0.2, 0], [11, -2, 0.2, 0], [11, -11, 0.2, 0],
                      ]
    return init_pose_list[index]

def get_test_init_pose(index):
    # x y z angle
    # init_pose_list = [
    #                     [11, -8.8, 0.2, 0], [12, 5.6, 0.2, 0], [2.1, 9.4, 0.2, 0],
    #                     [0.4, -3, 0.2, 0], [-12.6, -12.4, 0.2, 0], [-12.1, 3, 0.2, 0],
    #                     [-4.7, 13.4, 0.2, 0], [2, -12.3, 0.2, 0], [-7.2, -3.39, 0.2, 0],
    #                   ]  # easy
    init_pose_list = [
                        [-18.3, 20, 0.2, 0], [3.6, 17.7, 0.2, 0], [21.7, 9, 0.2, 0],
                        [7.9, -6.3, 0.2, 0], [9.9, -20, 0.2, 0], [-18.8, -18.1, 0.2, 0],
                        [-7, -7.7, 0.2, 0], [-0.9, 6.1, 0.2, 0], [-0.5, -3.86, 0.2, 0],
                      ]  # hard
    return init_pose_list[index]


def get_filter_index(d_list):
    filter_index = []
    filter_flag = 0
    step = d_list.shape[0]
    num_env = d_list.shape[1]
    for i in range(num_env):
        for j in range(step):
            if d_list[j, i] == True:
                filter_flag += 1
            else:
                filter_flag = 0
            if filter_flag >= 2:
                filter_index.append(num_env*j + i)
    return filter_index


def get_group_terminal(terminal_list, index):
    group_terminal = False
    refer = [0, 6, 10, 15, 19, 24, 34, 44]
    r = bisect.bisect(refer, index)
    if reduce(lambda x, y: x * y, terminal_list[refer[r-1]:refer[r]]) == 1:
        group_terminal = True
    return group_terminal


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True) # num_env * frames * 1
    return log_density



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#algorithm_Parallel
    def __init__(self, epsilon=1e-4, shape=()):  # shape (1, 1, 84, 84)
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
