# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:08:53 2018

@author: Cliff Zhuo LI

"""

### Trilateral Localization

import pandas as pd
import numpy as np
import scipy as sp
import math
import datetime
import operator
import matplotlib.pyplot as plt

from Algorithm.Sql import ConnectDb

### define the position of anchor
conn = ConnectDb()
result = conn.select_table("DeviceManagement_device", '1', '1', 'id','`position`')
anchors_position_dict = {}
tmp_dic = dict(result)
for key in tmp_dic:
    anchors_position_dict[str(key)] = tuple([float(item) for item in tmp_dic[key].split(' ')])
print ("*****************anchor))", anchors_position_dict)
# approx_distance1 = 1.8
approx_distance1 = 2
# approx_distance1 = 1
approx_distance2 = 3.5
var_range1 = 0.3
var_range2 = 0.2
var_range_tri_pos = 0.2
var_range_2c_contain = 0.2
var_range_2_intersection = 1.0
var_range_0m_radius = 0.2
sigma_approx1 = 0.2
sigma_approx2 = 0.2
sigma_tri_pos = 0.1
sigma_2c_contain = 0.2
sigma_2_intersection = 1.0
sigma_0m_radius = 0.2
n_used_pos_anchors= 3


if_do_approx2_estimation = False
if_show_realtime_tracking = True
# define csv file directory
folder_name = 'left'
gw1_csv_dir = '/Users/sjq/Develop/Python/localization/measure/test-data/' + folder_name + '-anchor1.csv'
gw2_csv_dir = '/Users/sjq/Develop/Python/localization/measure/test-data/' + folder_name + '-anchor2.csv'
gw3_csv_dir = '/Users/sjq/Develop/Python/localization/measure/test-data/' + folder_name + '-anchor3.csv'
gw4_csv_dir = '/Users/sjq/Develop/Python/localization/measure/test-data/' + folder_name + '-anchor4.csv'
gw5_csv_dir = '/Users/sjq/Develop/Python/localization/measure/test-data/' + folder_name + '-anchor5.csv'


def gen_trunc_norm(up_bound=1, low_bound=-1, num_samples=1, mu=0, sigma=0.5):
    randn_num = np.random.normal(loc=mu, scale=sigma, size=num_samples)[0]
    if randn_num <= low_bound or randn_num >= up_bound:
        randn_num = gen_trunc_norm()
    return randn_num


def calculate_radius(d):
    anchor_height = 0.8
    if d <= anchor_height:
        rand_r = gen_trunc_norm(up_bound=var_range_0m_radius, low_bound=-var_range_0m_radius, mu=0, sigma=sigma_0m_radius)
        R = np.abs(rand_r)
    else:
        R = np.sqrt(d**2 - anchor_height**2)
    return R

def rssi_to_distantce(anchor_id, rssi):
    anchor_height = 1.2
    if anchor_id in ['3']:
        n = 2.5
        A = -50
    elif anchor_id in ['1', '5']:
        n = 2.6
        A = -44
    elif anchor_id in ['4']:
        n = 3.0
        A = -45
    elif anchor_id in ['2']:
        n = 2.1
        A = -51

    #txPower = -40
    exponent = (np.abs(rssi) - np.abs(A)) / (10 * n)
    d = math.pow(10, exponent)
    if d <= anchor_height:
        R = np.abs(gen_trunc_norm())
        #print("R_random =", R)
    else:
        R = np.sqrt(d**2 - anchor_height**2)
    return R

def get_distance_list(rssi_list):
    distance_list = []
    for ii in range(len(rssi_list)):
        rssi_tuple_temp = rssi_list[ii]
        anchor_id_temp = rssi_tuple_temp[0]
        rssi_temp = rssi_tuple_temp[1]
        est_distance = rssi_to_distantce(anchor_id_temp, rssi_temp)
        distance_tuple = (anchor_id_temp, est_distance)
        print("anchor_id_temp =", anchor_id_temp, " rssi_temp =", rssi_temp, " est_distance =", est_distance)
        distance_list.append(distance_tuple)
    return distance_list

### get the observation of distance
def data_smoothing(rssi_list, avg_rssi_prev):
    list_length = len(rssi_list)
    if list_length == 0:
        return avg_rssi_prev
    elif 0 < list_length < 4:
        return np.mean(rssi_list)
    elif 4 <= list_length < 7:
        rssi_list.sort()
        rssi_list.pop(-1)
        rssi_list.pop(0)
        return np.mean(rssi_list)
    elif 7 <= list_length < 11:
        rssi_list.sort()
        rssi_list.pop(-1)
        rssi_list.pop(-1)
        rssi_list.pop(0)
        rssi_list.pop(0)
        return np.mean(rssi_list)
    elif 11 <= list_length:
        rssi_list.sort()
        rssi_list.pop(-1)
        rssi_list.pop(-1)
        rssi_list.pop(-1)
        rssi_list.pop(0)
        rssi_list.pop(0)
        rssi_list.pop(0)
        return np.mean(rssi_list)

def get_observation(test_time_steps = 100):
    gw1_data = pd.read_csv(gw1_csv_dir)
    gw2_data = pd.read_csv(gw2_csv_dir)
    gw3_data = pd.read_csv(gw3_csv_dir)
    gw4_data = pd.read_csv(gw4_csv_dir)
    gw5_data = pd.read_csv(gw5_csv_dir)
    gw1_data['timestamp'] = pd.to_datetime(gw1_data['timestamp'], format='%d/%m/%Y %H:%M:%S')
    gw2_data['timestamp'] = pd.to_datetime(gw2_data['timestamp'], format='%d/%m/%Y %H:%M:%S')
    gw3_data['timestamp'] = pd.to_datetime(gw3_data['timestamp'], format='%d/%m/%Y %H:%M:%S')
    gw4_data['timestamp'] = pd.to_datetime(gw4_data['timestamp'], format='%d/%m/%Y %H:%M:%S')
    gw5_data['timestamp'] = pd.to_datetime(gw5_data['timestamp'], format='%d/%m/%Y %H:%M:%S')
    start_1 = gw1_data['timestamp'].min() # sort_values(by = "timestamp")['timestamp']
    start_2 = gw2_data['timestamp'].min()
    start_3 = gw3_data['timestamp'].min()
    start_4 = gw4_data['timestamp'].min()
    start_5 = gw5_data['timestamp'].min()
    start = max(start_1, start_2, start_3, start_4, start_5)
    #start = datetime.datetime(2018, 12, 14, 2, 22, 17)
    time_window = datetime.timedelta(seconds=2)
    window_shift = datetime.timedelta(seconds=1)
    gw1_avg_rssi_prev = -55
    gw2_avg_rssi_prev = -55
    gw3_avg_rssi_prev = -55
    gw4_avg_rssi_prev = -55
    gw5_avg_rssi_prev = -55
    observation_list = []
    for i in range(test_time_steps):
        #rssi_dict = {}
        #distance_dict = {}
        observation_dict = {}
        gw1_rssi_pd = gw1_data[ (start <= gw1_data['timestamp']) & (start + time_window > gw1_data['timestamp']) ]
        gw2_rssi_pd = gw2_data[ (start <= gw2_data['timestamp']) & (start + time_window > gw2_data['timestamp']) ]
        gw3_rssi_pd = gw3_data[ (start <= gw3_data['timestamp']) & (start + time_window > gw3_data['timestamp']) ]
        gw4_rssi_pd = gw4_data[ (start <= gw4_data['timestamp']) & (start + time_window > gw4_data['timestamp']) ]
        gw5_rssi_pd = gw5_data[ (start <= gw5_data['timestamp']) & (start + time_window > gw5_data['timestamp']) ]
        gw1_avg_rssi = data_smoothing(list(gw1_rssi_pd['rssi']), gw1_avg_rssi_prev)
        gw2_avg_rssi = data_smoothing(list(gw2_rssi_pd['rssi']), gw2_avg_rssi_prev)
        gw3_avg_rssi = data_smoothing(list(gw3_rssi_pd['rssi']), gw3_avg_rssi_prev)
        gw4_avg_rssi = data_smoothing(list(gw4_rssi_pd['rssi']), gw4_avg_rssi_prev)
        gw5_avg_rssi = data_smoothing(list(gw5_rssi_pd['rssi']), gw5_avg_rssi_prev)
        observation_dict['1'] = rssi_to_distantce('1', gw1_avg_rssi), gw1_avg_rssi
        observation_dict['2'] = rssi_to_distantce('2', gw2_avg_rssi), gw2_avg_rssi
        observation_dict['3'] = rssi_to_distantce('3', gw3_avg_rssi), gw3_avg_rssi
        observation_dict['4'] = rssi_to_distantce('4', gw4_avg_rssi), gw4_avg_rssi
        observation_dict['5'] = rssi_to_distantce('5', gw5_avg_rssi), gw5_avg_rssi
        #sorted_rssi = sorted(rssi_dict.items(), key=operator.itemgetter(1), reverse=True)
        #sorted_distance = sorted(distance_dict.items(), key=operator.itemgetter(1), reverse=False)
        sorted_observation = sorted(observation_dict.items(), key=operator.itemgetter(1), reverse=False)
        observation_list.append(sorted_observation)
        gw1_avg_rssi_prev = gw1_avg_rssi
        gw2_avg_rssi_prev = gw2_avg_rssi
        gw3_avg_rssi_prev = gw3_avg_rssi
        gw4_avg_rssi_prev = gw4_avg_rssi
        gw5_avg_rssi_prev = gw5_avg_rssi
        start += window_shift
    return observation_list

def show_dynamic_trajectory(distance_list, est_position_list):
    plt.clf()
    plt.cla()
    est_position = est_position_list[-1]
    x_hat, y_hat = est_position[0], est_position[1]
    x_hat_list = np.array(est_position_list)[:, 0].flatten()
    y_hat_list = np.array(est_position_list)[:, 1].flatten()
    topK_anchor_list = np.array(distance_list)[:, 0]

    # plot the position of anchors
    for ii in range(len(anchors_position_dict)):
        anchor_xi = list(anchors_position_dict.items())[ii][1][0]
        anchor_yi = list(anchors_position_dict.items())[ii][1][1]
        plt.plot(anchor_xi, anchor_yi, color='black', marker='*', markersize='15')
    # plot the lines between estimate position of tags and top-K anchors
    for jj in range(len(topK_anchor_list)):
        topK_anchor_xi = anchors_position_dict[topK_anchor_list[jj]][0]
        topK_anchor_yi = anchors_position_dict[topK_anchor_list[jj]][1]
        plt.plot([x_hat, topK_anchor_xi], [y_hat, topK_anchor_yi], "-k")

    plt.plot(x_hat, y_hat, color='blue', marker='.', markersize='15')
    plt.plot(x_hat_list, y_hat_list, "--r")
    point_list = est_position_list[0:-1]
    for i in range(len(point_list)):
        x = point_list[i][0]
        y = point_list[i][1]
        plt.plot(x, y, marker='.', color='blue', markersize='8')
        plt.text(x, y + 0.25, i, color='black')
    plt.xlim([-0.1, 12.6])
    plt.ylim([-0.1, 18.1])
    plt.grid(True)
    color = 'green'
    linewidth = 5.0
    plt.plot([7, 11], [14.5, 14.5], color='brown', linewidth=linewidth)
    plt.plot([0 ,12.5], [0,0], color=color, linewidth=linewidth)
    plt.plot([12.5 ,12.5], [0, 14.5], color=color, linewidth=linewidth)
    plt.plot([0, 0], [0, 18], color=color, linewidth=linewidth)
    plt.plot([0, 0], [0, 18], color=color, linewidth=linewidth)
    plt.plot([0, 7], [18, 18], color=color, linewidth=linewidth)
    plt.plot([7, 7], [18, 14.5], color=color, linewidth=linewidth)
    plt.plot([11, 12.5], [14.5, 14.5], color=color, linewidth=linewidth)
    rect1 = plt.Rectangle((4, 4.5), 4, 3)
    rect2 = plt.Rectangle((4, 10.5), 4, 3)
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.xticks(np.arange(0, 14, 2))
    plt.yticks(np.arange(0, 20, 2))
    plt.pause(0.3)

def get_unique_intersection(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    intersection = calc_intersection(x1, y1, r1, x2, y2, r2)
    print("================================ Step 4: get_unique_intersection =================================")
    flag = False
    p = None
    print("intersection =", intersection)
    if intersection is not None and len(intersection) != 0:
        flag = True
        d_c1_c3 = calc_distance_btw_2points(x1, y1, x3, y3)
        d_c2_c3 = calc_distance_btw_2points(x2, y2, x3, y3)
        print("d_c1_c3 = %.2f, d_c2_c3 = %.2f, r3 = %.2f" % (d_c1_c3, d_c2_c3, r3))
        print("d_c1_c3 < r3 =", d_c1_c3 < r3, ",   d_c2_c3 < r3 =", d_c2_c3 < r3)
        for ii in range(len(intersection)):
            point = intersection[ii]
            print("The", ii+1, "th intersection =", point)
            px = point[0]
            py = point[1]
            d_p12_c3 = calc_distance_btw_2points(px, py, x3, y3)
            if p is None:
                # find the 1st intersection
                flag = True
                p = point
                print("p = p1")
            elif p is not None:
                # find the 2nd intersection, and compare it with the 1st intersection in terms of
                # distance to the center of circle 3
                p1x = p[0]
                p1y = p[1]
                d_p12_c3_old = calc_distance_btw_2points(p1x, p1y, x3, y3)
                print("The 2nd intersection: d_p2c_old = %.2f, d_p12_c3 = %.2f" % (d_p12_c3_old, d_p12_c3))
                print("(d_p12_c3 < d_p12_c3_old) =", d_p12_c3 < d_p12_c3_old)
                #print("d_c1_c3 < r3 =", d_c1_c3 < r3, ",   d_c2_c3 < r3 =", d_c2_c3 < r3)
                if d_p12_c3_old < d_p12_c3:
                    print("(d_p12_c3 < d_p12_c3_old) =", d_p12_c3 < d_p12_c3_old)
                    print("p = p2, the new closer intersection")
                    p = point
                else:
                    print("(d_p12_c3 < d_p12_c3_old) =", d_p12_c3 < d_p12_c3_old)
                    print("p = p1, the first intersection still")
                    pass
# =============================================================================
#                 if d_c1_c3 < r3 and d_c2_c3 < r3:
#                     #print("d_c1_c3 < r3 =", d_c1_c3 < r3, ",   d_c2_c3 < r3 =", d_c2_c3 < r3)
#                     if d_p12_c3_old < d_p12_c3:
#                         print("(d_c1_c3 < r3 and d_c2_c3 < r3) and choose the farther intersection")
#                         pass
#                         print("(d_c1_c3 < r3 and d_c2_c3 < r3) and choose the farther intersection")
#                         print("p = p2")
#                         p = point
#                         p2x = p[0]
#                         p2y = p[1]
#                         d_p1_p2 = calc_distance_btw_2points(p1x, p1y, p2x, p2y)
#                         rand_distance = gen_trunc_norm(up_bound=var_range_2_intersection, \
#                                                         low_bound=-var_range_2_intersection, \
#                                                         mu=0, sigma=sigma_2_intersection)
#                         delta_distance = abs(rand_distance) + d_p1_p2/3
#                         # TODO: need more tests since its emperical values
#                         factor = delta_distance / d_p1_p2
#                         p2x_new = p2x - factor * (p2x - p1x)
#                         p2y_new = p2y - factor * (p2y - p1y)
#                         p[0] = p2x_new
#                         p[1] = p2y_new
#                     else:
#                         print("(d_c1_c3 < r3 and d_c2_c3 < r3) and neglect the closer intersection")
#                         pass
#                 else:
#                     #print("d_c1_c3 < r3 =", d_c1_c3 < r3, ",   d_c2_c3 < r3 =", d_c2_c3 < r3)
#                     if d_p12_c3_old > d_p12_c3:
#                         print("Not (d_c1_c3 < r3 and d_c2_c3 < r3) and choose the closer intersection")
#                         print("p = p2")
#                         p = point
#                     else:
#                         print("Not (d_c1_c3 < r3 and d_c2_c3 < r3) and neglect the farther intersection")
#                         pass
# =============================================================================
    print("if_intersection_exist =", flag, ",   unique_point =", p)
    return flag, p

def get_relative_position(x, y, r, theta):
    delta_x = 0.7 * r * math.cos(theta)
    delta_y = 0.7 * r * math.sin(theta)
    est_x = x + delta_x
    est_y = y + delta_y
    return est_x, est_y

def convert_to_positive_angle(angle):
    if angle < 0.0:
        angle = angle + 2 * math.pi
        #angle = angle
    return angle

def calc_steering_angle(theta1, theta2, theta12):
    if abs(theta12) > 180.0:
        if theta1 > 0:
            theta12 = (theta1 + 360.0) - theta2
        elif theta2 > 0:
            theta12 = theta1 - (theta2 + 360.0)
    return theta12

def trilateral_centroid_positioning(tri_data_list):
    """
    # format of input tri_data_list
    # tri_data_list = [(anchor_id1, (distance1, rssi1)),
    #                  (anchor_id2, (distance2, rssi2)),
    #                  (anchor_id3, (distance3, rssi3))]
    """
    print("============================ Step 1: trilateral_centroid_positioning =============================")
    #multiply_scalar_list = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1,9, 2.0]

    def get_tri_centroid(x1, y1, r1, r1_orig, x2, y2, r2, r2_orig, x3, y3, r3, r3_orig, iteration = 1):
        print("================================== Step 2: get_tri_centroid ======================================")
        print("\n")
        print("iteration = %d" % (iteration))
        print("====================== Calculate intersection between circle 1 and circle 2 ======================")
        flag1, p1 = get_unique_intersection(x1, y1, r1, x2, y2, r2, x3, y3, r3)
        print("flag1 =", flag1, ",   p1 =", p1)
        print("====================== Calculate intersection between circle 3 and circle 1 ======================")
        flag2, p2 = get_unique_intersection(x3, y3, r3, x1, y1, r1, x2, y2, r2)
        print("flag2 =", flag2, ",   p2 =", p2)
        print("====================== Calculate intersection between circle 2 and circle 3 ======================")
        flag3, p3 = get_unique_intersection(x2, y2, r2, x3, y3, r3, x1, y1, r1)
        print("flag3 =", flag3, ",   p3 =", p3)
        if (flag1 is True) and (flag2 is True) and (flag3 is True):
            print("================================== calculate the centroid ========================================")
            rand_x = gen_trunc_norm(up_bound=var_range_tri_pos, low_bound=-var_range_tri_pos, mu=0, sigma=sigma_tri_pos)
            rand_y = gen_trunc_norm(up_bound=var_range_tri_pos, low_bound=-var_range_tri_pos, mu=0, sigma=sigma_tri_pos)
            #centroid_x = (p1[0] + p2[0] + p3[0]) / 3
            #centroid_y = (p1[1] + p2[1] + p3[1]) / 3
            centroid_x = calc_centroid_of_3points(p1[0], p2[0], p3[0])
            centroid_y = calc_centroid_of_3points(p1[1], p2[1], p3[1])
            centroid_x = centroid_x + rand_x
            centroid_y = centroid_y + rand_y
            tri_centroid = [centroid_x, centroid_y]
            print("tri_centroid =", tri_centroid)
            if_get_centroid = True
            return if_get_centroid, tri_centroid
        else:
            print("================================= iterative radius scaling =======================================")
            if_get_centroid = False
            tri_centroid = None
            scaling_iter = 11
            multiply_factor_list = np.linspace(1.0, 1.7, scaling_iter)
            print("Original radius: r1 = %.2f, r2 = %.2f, r3 = %.2f" % (r1_orig, r2_orig, r3_orig))
            print("Before the %d-th scaling: r1 = %.2f, r2 = %.2f, r3 = %.2f" % (iteration, r1, r2, r3))
            print("Go to the %d-th recursion" % (iteration))
            iteration = iteration + 1
            multiply_factor = multiply_factor_list[iteration-1]
            r1_new = r1_orig * multiply_factor
            r2_new = r2_orig * multiply_factor
            r3_new = r3_orig * multiply_factor
            print("After the %d-th scaling: multiply_factor = %.2f, r1_new = %.2f, r2_new = %.2f, r3_new = %.2f" % (iteration, multiply_factor, r1_new, r2_new, r3_new))
            if iteration <= scaling_iter-1:
                if_get_centroid, tri_centroid = get_tri_centroid(x1, y1, r1_new, r1_orig, x2, y2, r2_new, r2_orig, x3, y3, r3_new, r3_orig, iteration = iteration)
                import copy
                print("Retrun from the", iteration-1, "th iteration,   if_get_centroid =", if_get_centroid, ",   tri_centroid =", copy.deepcopy(tri_centroid))
            return if_get_centroid, tri_centroid

    def get_approx1_centroid(x1, y1, r1, x2, y2, r2, x3, y3, r3):
        """
        # assume r3 < approx_distace1, and we try to estiamte the pos of tag near (x3, y3)
        # L13: y = k13x + b13, the line between points (x1, y1) and (x3, y3), where k13 = (y1 - y3) / (x1 - x3)
        # L23: y = k23x + b23, the line between points (y2, y2) and (x3, y3), where k23 = (y2 - y3) / (x2 - x3)
        # L03: y = k03x + b03, the line between points (xx0, yy0) and (x3, y3), where k03 = (y0 - y3) / (x0 - x3)
        # theta12 is the steering angle between L13 and L23, where tan(theta12) = (k23 - k13) / (1 + k13 * k23)
        """
        k13 = (y1 - y3) / (x1 - x3 + 1e-10)
        k23 = (y2 - y3) / (x2 - x3 + 1e-10)
        print("Before switching: k13 = %.2f,  k23 = %.2f" % (k13, k23))
        theta1 = convert_to_positive_angle(math.atan2(y1-y3, x1-x3))
        theta2 = convert_to_positive_angle(math.atan2(y2-y3, x2-x3))
        theta12 = theta1 - theta2
        theta12 = calc_steering_angle(theta1, theta2, theta12)
        print("After switching: theta1 = %.2f,  theta2 = %.2f,  theta12 = %.2f" % (math.degrees(theta1), math.degrees(theta2), math.degrees(theta12)))
        if theta12 > 0.0:
            print("switch (x1, y1) and (x2, y2)")
            temp_x, temp_y, temp_r = x1, y1, r1
            x1, y1, r1 = x2, y2, r2
            x2, y2, r2 = temp_x, temp_y, temp_r
            k13 = (y1 - y3) / (x1 - x3 + 1e-10)
            k23 = (y2 - y3) / (x2 - x3 + 1e-10)
            print("Before switching: k13 = %.2f,  k23 = %.2f" % (k13, k23))
            theta1 = convert_to_positive_angle(math.atan2(y1-y3, x1-x3))
            theta2 = convert_to_positive_angle(math.atan2(y2-y3, x2-x3))
            print("After switching: theta1 = %.2f,  theta2 = %.2f,  theta12 = %.2f" % (math.degrees(theta1), math.degrees(theta2), math.degrees(theta12)))
            theta12 = theta1 - theta2
            theta12 = calc_steering_angle(theta1, theta2, theta12)
        delta_theta12 = abs(theta12)
        d13 = calc_distance_btw_2points(x1, y1, x3, y3)
        d23 = calc_distance_btw_2points(x2, y2, x3, y3)
        divide_factor = 3
        if d13 >= max(r1, r3) and d23 >= max(r2, r3):
            print("d13 >= max(r1, r3) and d23 >= max(r2, r3)")
            """
            # delta_theta12: steering angle between L1 and L2, where tan(delta_theta12) = (k23 - k13) / (1 + k13k23)
            # delta_theta02: steering angle between L0 and L2, where tan(delta_theta02) = (k23 - k03) / (1 + k03k23)
            # delta_theta12 = 2 * delta_theta02
            """
            delta_theta02 = delta_theta12 / 2
            theta0 = theta2 - delta_theta02
            print("theta0 =", math.degrees(theta0))
            est_x, est_y = get_relative_position(x3, y3, r3, theta0)
        elif d13 >= max(r1, r3) and d23 < max(r2, r3):
            print("d13 >= max(r1, r3) and d23 < max(r2, r3)")
            delta_theta01 = delta_theta12 / divide_factor
            theta0 = theta1 - delta_theta01
            print("theta0 =", math.degrees(theta0))
            est_x, est_y = get_relative_position(x3, y3, r3, theta0)
        elif d13 < max(r1, r3) and d23 >= max(r2, r3):
            print("d13 < max(r1, r3) and d23 >= max(r2, r3)")
            delta_theta20 = delta_theta12 / divide_factor
            theta0 = theta2 + delta_theta20
            print("theta0 =", math.degrees(theta0))
            est_x, est_y = get_relative_position(x3, y3, r3, theta0)
        elif d13 < max(r1, r3) and d23 < max(r2, r3):
            print("d13 < max(r1, r3) and d23 < max(r2, r3)")
            delta_theta02 = delta_theta12 / 2
            theta0 = (theta2 - delta_theta02) - math.pi
            est_x, est_y = get_relative_position(x3, y3, r3, theta0)
        else:
            est_x, est_y = x3, y3
        return est_x, est_y

    def choose_approx2_anchor(x1, y1, r1, x2, y2, r2, x3, y3, r3):
        flag = False
        anchor_x, anchor_y, anchor_r = None, None, None
        r_min = min(r1, r2, r3)
        if r_min >= approx_distance1 and r_min <= approx_distance2:
            flag = True
            if r_min == r1:
                anchor_x, anchor_y, anchor_r = x1, y1, r1
            elif r_min == r2:
                anchor_x, anchor_y, anchor_r = x2, y2, r2
            elif r_min == r3:
                anchor_x, anchor_y, anchor_r = x3, y3, r3
        return flag, anchor_x, anchor_y, anchor_r

    def perform_approx2_estimation(x, y, r, tri_centroid):
        est_x, est_y = tri_centroid[0], tri_centroid[1]
        d_est_pos_anchor = calc_distance_btw_2points(x, y, est_x, est_y)
        factor = r / d_est_pos_anchor
        est_x_new = (est_x - x) * factor + x
        est_y_new = (est_y - y) * factor + y
        rand_x = gen_trunc_norm(up_bound=var_range2, low_bound=-var_range2, mu=0, sigma=sigma_approx2)
        rand_y = gen_trunc_norm(up_bound=var_range2, low_bound=-var_range2, mu=0, sigma=sigma_approx2)
        print("rand_x =", rand_x, ",   rand_y =", rand_y)
        est_x_new, est_y_new = est_x_new + rand_x, est_y_new + rand_y
        tri_centroid_new = [est_x_new, est_y_new]
        return tri_centroid_new

    def check_if_obtuse_triangle(x1, y1, x2, y2, x3, y3):
        d_c1_c2 = calc_distance_btw_2points(x1, y1, x2, y2)
        d_c1_c3 = calc_distance_btw_2points(x1, y1, x3, y3)
        d_c2_c3 = calc_distance_btw_2points(x2, y2, x3, y3)
        flag = True
        obtuse_triangle = math.pi / 2
        obtuse_triangle_threshold = 135.0
        if d_c1_c2**2 > (d_c1_c3**2 + d_c2_c3**2):
            print("d_c1_c2^2 > (d_c1_c3^2 + d_c2_c3^2)")
            k13 = (y1 - y3) / (x1 - x3 + 1e-10)
            k23 = (y2 - y3) / (x2 - x3 + 1e-10)
            theta132 = math.pi - math.atan(abs((k23 - k13) / (1 + k13 * k23)))
            obtuse_triangle = theta132
            print("theta132 =", math.degrees(theta132))
        elif d_c1_c3**2 > (d_c1_c2**2 + d_c2_c3**2):
            print("d_c1_c3^2 > (d_c1_c2^2 + d_c2_c3^2)")
            k12 = (y1 - y2) / (x1 - x2 + 1e-10)
            k32 = (y3 - y2) / (x3 - x2 + 1e-10)
            theta321 = math.pi - math.atan(abs((k32 - k12) / (1 + k12 * k32)))
            obtuse_triangle = theta321
            print("theta321 =", math.degrees(theta321))
        elif d_c2_c3**2 > (d_c1_c2**2 + d_c1_c3**2):
            print("d_c2_c3^2 > (d_c1_c2^2 + d_c1_c3^2)")
            k21 = (y2 - y1) / (x2 - x1 + 1e-10)
            k31 = (y3 - y1) / (x3 - x1 + 1e-10)
            theta213 = math.pi - math.atan(abs((k31 - k21) / (1 + k21 * k31)))
            obtuse_triangle = theta213
            print("theta213 =", math.degrees(theta213))
        print("obtuse_triangle=", math.degrees(obtuse_triangle))
        if math.degrees(obtuse_triangle) > obtuse_triangle_threshold:
            flag = False
        print("if_valid_obtuse_triangle:", flag)
        return flag

    anchor1_id = tri_data_list[0][0]
    anchor2_id = tri_data_list[1][0]
    anchor3_id = tri_data_list[2][0]
    # r is radius
    r1 = tri_data_list[0][1][0]
    r2 = tri_data_list[1][1][0]
    r3 = tri_data_list[2][1][0]
    r1_orig, r2_orig, r3_orig = r1, r2, r3
    x1 = anchors_position_dict[anchor1_id][0]
    y1 = anchors_position_dict[anchor1_id][1]
    x2 = anchors_position_dict[anchor2_id][0]
    y2 = anchors_position_dict[anchor2_id][1]
    x3 = anchors_position_dict[anchor3_id][0]
    y3 = anchors_position_dict[anchor3_id][1]

    r_min = min(r1, r2, r3)
    if r_min < approx_distance1:
        rand_x = gen_trunc_norm(up_bound=var_range1, low_bound=-var_range1, mu=0, sigma=sigma_approx1)
        rand_y = gen_trunc_norm(up_bound=var_range1, low_bound=-var_range1, mu=0, sigma=sigma_approx1)
        if r_min == r1:
            print("==================================== r_min(r1, r2, r3) == r1 =====================================")
            #centroid_x, centroid_y = (x1 + rand_x), (y1 + rand_y)
            centroid_x, centroid_y = get_approx1_centroid(x2, y2, r2, x3, y3, r3, x1, y1, r1)
        elif r_min == r2:
            print("==================================== r_min(r1, r2, r3) == r2 =====================================")
            #centroid_x, centroid_y = (x2 + rand_x), (y2 + rand_y)
            centroid_x, centroid_y = get_approx1_centroid(x3, y3, r3, x1, y1, r1, x2, y2, r2)
        elif r_min == r3:
            print("==================================== r_min(r1, r2, r3) == r3 =====================================")
            #centroid_x, centroid_y = (x3 + rand_x), (y3 + rand_y)
            centroid_x, centroid_y = get_approx1_centroid(x1, y1, r1, x2, y2, r2, x3, y3, r3)
        centroid_x, centroid_y = centroid_x + rand_x, centroid_y + rand_y
        tri_centroid = [centroid_x, centroid_y]
        if_get_centroid = True
    else:
        if_valid_obtuse_triangle = check_if_obtuse_triangle(x1, y1, x2, y2, x3, y3)
        if if_valid_obtuse_triangle is True:
            if_get_centroid, tri_centroid = get_tri_centroid(x1, y1, r1, r1_orig, x2, y2, r2, r2_orig, x3, y3, r3, r3_orig)
        else:
            if_get_centroid = False
            tri_centroid = None
    if if_get_centroid is True and if_do_approx2_estimation is True:
        flag, anchor_x, anchor_y, anchor_r = choose_approx2_anchor(x1, y1, r1, x2, y2, r2, x3, y3, r3)
        if flag is True:
            tri_centroid = perform_approx2_estimation(anchor_x, anchor_y, anchor_r, tri_centroid)
    return if_get_centroid, tri_centroid

def calc_distance_btw_2points(a1, b1, a2, b2):
    squared_diff_distance = (a1 - a2)**2 + (b1 - b2)**2
    distance = np.sqrt(squared_diff_distance)
    return distance

def calc_centroid_of_3points(p1, p2, p3):
    centroid_p = (p1 + p2 + p3) / 3
    return centroid_p

"""
# @param anchor_id1: the id of the 1st anchor
# @param r1: the radius of circle 1
# @param anchor_id2: the id of the 2nd anchor
# @param r2: the radius of circle 2
# @retrun intersection: a list of coordinates of the intersections of the two circles
"""
def calc_intersection(x1, y1, r1, x2, y2, r2):
    print("================================== Step 3: calc_intersection =====================================")
    d12 = calc_distance_btw_2points(x1, y1, x2, y2)

    """
    # x0 is the intersection
    """
    intersection = []

    if d12 > (r1 + r2):
        print("####################################### d12 > (r1 + r2) ##########################################")
        print("d12 = %.2f, (r1 + r2) = %.2f, r1 = %.2f, r2 = %.2f" % (d12, (r1+r2), r1, r2))
        """
        # d12 < (r1 + r2): two circles separate
        """
        return None

    elif d12 < np.abs(r1 - r2):
        print("####################################### d12 < |r1 - r2| ##########################################")
        """
        # d12 < |r1 - r2|: one circle contains the other
        """
        rand_n = gen_trunc_norm(up_bound=var_range_2c_contain, low_bound=-var_range_2c_contain, mu=0, sigma=sigma_2c_contain)
        if r1 > r2:
            r1_new = d12 + r2 - np.abs(rand_n)
            intersection = calc_intersection(x1, y1, r1_new, x2, y2, r2)
        elif r2 > r1:
            r2_new = d12 + r1 - np.abs(rand_n)
            intersection = calc_intersection(x1, y1, r1, x2, y2, r2_new)
        return intersection
        #return None

    elif x1 == x2 and y1 == y2:
        print("##################################### x1 == x2 and y1 == y2 ######################################")
        """
        # x1 == x2 and y1 == y2: two cincles are concentric
        """
        return None

    elif x1 != x2 and y1 == y2:
        print("##################################### x1 != x2 and y1 == y2 ######################################")
        """
        # when x1 != x2 and y1 == y2
        # r1^2 - (x1-x0)^2 = r2^2 - (x2-x0)^2 ==> x0 = ((r1^2-r2^2)-(x1^2-x2^2)) / 2(x2-x1)
        """
        x0 = ((x1**2 - x2**2) - (r1**2 - r2**2)) / (2*x1 - 2*x2)

        if d12 == np.abs(r1 - r2) or d12 == (r1 + r2):
            print("d12 == np.abs(r1 - r2) or d12 == (r1 + r2)")
            """
            # when there is only one intersection
            # internally tangent: d = |r1 - r2| and externally tangent: d = (r1 + r2)
            """
            y0 = y1
            intersection.append([x0, y0])
        else:
            print("d12 != np.abs(r1 - r2) or d12 != (r1 + r2)")
            """
            # when there are two intersections
            """
            y0 = y1
            delta_y = np.sqrt(r1**2 - (x0 - x1)**2)
            y0a = y0 + delta_y
            y0b = y0 - delta_y
            intersection.append([x0, y0a])
            intersection.append([x0, y0b])
        return intersection

    elif x1 == x2 and y1 != y2:
        print("##################################### x1 == x2 and y1 != y2 ######################################")
        """
        # when x1 == x2 and y1 != y2
        # r1^2 - (y0-y1)^2 = r2^2 - (y2-y0)^2 ==> y0 = ((r1^2-r2^2)-(y1^2-y2^2)) / 2(y2-y1)
        """
        y0 = ((y1**2 - y2**2) - (r1**2 - r2**2)) / (2*y1 - 2*y2)

        if d12 == np.abs(r1 - r2) or d12 == (r1 + r2):
            print("d12 == np.abs(r1 - r2) or d12 == (r1 + r2)")
            """
            # when there is only one intersection
            # internally tangent: d = |r1 - r2| and externally tangent: d = (r1 + r2)
            """
            x0 = x1
            intersection.append([x0, y0])
        else:
            print("d12 != np.abs(r1 - r2) or d12 != (r1 + r2)")
            """
            # when there are two intersections
            """
            x0 = x1
            delta_x = np.sqrt(r1**2 - (y0 - y1)**2)
            x0a = x0 - delta_x
            x0b = x0 + delta_x
            intersection.append([x0a, y0])
            intersection.append([x0b, y0])
        return intersection

    elif x1 != x2 and y1 != y2:
        print("##################################### x1 != x2 and y1 != y2 ######################################")
        """
        # when x1 != x2 and y1 != y2
        #
        # 1. L0': y = k0'x + b0', the line between (x1, y1) and (x2, y2)
        #    i.e., L0': (y-y1)/(y2-y1) = (x-x1)/(x2-x1) (x1≠x2, y1≠y2)
        #    where the slope k0' = (y2-y1) / (x2-x1)
        #
        # 2. L0: y = k0x + b0, the perpendicular line passing through the intersection(s)
        #           _
        #          |  k0 = -1/k0' = -(x2-x1)/(y2-y1),
        #    where |
        #          |_ b0 = [(r1^2-r2^2)-(x1^2-x2^2)-(y1^2-y2^2)]/2(y2-y1)
        #                                          _
        #                                         |  y0 = k0x0 + b0
        #    in which b0 is obtained by solving: -|
        #                                         |_ r1^2 - [(x0-x1)^2+(y0-y1)^2] = r2^2 - [(x0-x2)^2-(y0-y2)^2]
        #
        # 3. then, we solve the following quadratic equations,
        #      _
        #     |  (x-x1)^2 + (y-y1)^2 = r1^2
        #    -|
        #     |_  y = k0x + b0
        #
        #    simplifying it, we get: (k0^2-1)x^2 - (2x1+2k0y1-2k0b0)x - (x1^2+y1^2-2b0y1+b0^2-r1^2) = 0
        #
        #                       -b +- sqrt(b^2-4ac)
        #    and solve it  x = ---------------------
        #     _                         2a
        #    |  a = k0^2 + 1
        #    |  b = -(2x1 + 2k0y1 - 2k0b0)
        #    |_ c = x1^2 + y1^2 - 2b0y1 + b0^2 - r1^2
        #
        """
        k0 = (x1 - x2) / (y2 - y1)
        b0 = ((r1**2 - r2**2) - (x1**2 - x2**2) - (y1**2 - y2**2)) / (2*y2 - 2*y1)

        a = k0 * k0 + 1
        b = -(2 * x1 + 2 * k0 * y1 - 2 * k0 * b0)
        c = x1**2 + y1**2 - 2 * b0 * y1 + b0**2 - r1**2
        #discriminant = b * b - 4 * a * c

        if d12 == np.abs(r1 - r2) or d12 == (r1 + r2):
            x0 = - b / (2 * a)
            y0 = k0 * x0 + b0
            intersection.append([x0, y0])
        else:
            x0a = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
            y0a = k0 * x0a + b0
            intersection.append([x0a, y0a])
            x0b = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            y0b = k0 * x0b + b0
            intersection.append([x0b, y0b])
        return intersection

def plot_anchors_distance(topK_observation_list):
    def plot_circle(cx, cy, r):
        theta = np.arange(0, 2*np.pi, 0.01)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        #plt.plot(cx, cy, color='black', marker='*', markersize='15')
        plt.plot(x, y, color = 'red', linestyle='--')

    def plot_anchors():
        for key in list(anchors_position_dict.keys()):
            anchor_x = anchors_position_dict[key][0]
            anchor_y = anchors_position_dict[key][1]
            plt.plot(anchor_x, anchor_y, color='black', marker='*', markersize='15')

    plt.cla()
    plt.clf()
    plot_anchors()
    plt.xlim([-0.1, 12.6])
    plt.ylim([-0.1, 18.1])
    plt.grid(True)

    for item in topK_observation_list:
        anchor_id = item[0]
        cx = anchors_position_dict[anchor_id][0]
        cy = anchors_position_dict[anchor_id][1]
        r = item[1][0]
        plot_circle(cx, cy, r)

def plot_tri_centroid(tri_centroid):
    px = tri_centroid[0]
    py = tri_centroid[1]
    plt.plot(px, py, color='blue', marker='.', markersize='25')


if __name__ == '__main__':
    observed_data = get_observation()
    total_timestep = len(observed_data)
    X_hat_prev = np.array([0, 0])
    X_hat_list = []
    hist_data_list = []
    temp_time = 0

    # top3_observation_list=[('1', (1.27, -45.23)), ('2', (5.13, -68.85)), ('3', (6.09, -71.38))]
    top3_observation_list=[('1', (1.27, -45.23)), ('2', (5.13, -68.85)), ('5', (6.62, -69.69))]
    # top3_observation_list = [('1', (1.27, -45.23)), ('3', (6.09, -71.38)), ('5', (6.62, -69.69))]
    # top3_observation_list=[('2', (5.13, -68.85)), ('3', (6.09, -71.38)), ('5', (6.62, -69.69))]
    plot_anchors_distance(top3_observation_list)

    if_get_centroid, tri_centroid = trilateral_centroid_positioning(top3_observation_list)

    if if_get_centroid is True:
        print(tri_centroid)

        color = 'green'
        linewidth = 5.0
        plt.plot([7, 11], [14.5, 14.5], color='brown', linewidth=linewidth)
        plt.plot([0 ,12.5], [0,0], color=color, linewidth=linewidth)
        plt.plot([12.5 ,12.5], [0, 14.5], color=color, linewidth=linewidth)
        plt.plot([0, 0], [0, 18], color=color, linewidth=linewidth)
        plt.plot([0, 0], [0, 18], color=color, linewidth=linewidth)
        plt.plot([0, 7], [18, 18], color=color, linewidth=linewidth)
        plt.plot([7, 7], [18, 14.5], color=color, linewidth=linewidth)
        plt.plot([11, 12.5], [14.5, 14.5], color=color, linewidth=linewidth)
        plot_tri_centroid(tri_centroid)
        rect1 = plt.Rectangle((4, 4), 4, 3)
        rect2 = plt.Rectangle((4, 10), 4, 3)
        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
        plt.xticks(np.arange(0, 14, 2))
        plt.yticks(np.arange(0, 20, 2))

        plt.show()