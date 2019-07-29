"""
model is generated, using the model to estimate distance
"""
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import operator
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import optimize

from Algorithm.train_distance_estimator import distance_model
from Algorithm.trilateral_localization import trilateral_centroid_positioning, show_dynamic_trajectory
from Algorithm.filter.kalman_filter import tracker, gen_R_and_Q
from Algorithm.Sql import ConnectDb

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class IndoorLocalization(object):
    def __init__(self):
        self.y, self.x_, self.a_ = distance_model()
        self.model_path = r"E:\save_model_folder_name\ble_distance_estimator_model_20190618_11-08-33\ble_distance_estimator_model_20190618_11-08-33.ckpt"
        # self.model_path = r"E:\save_model_folder_name\ble_distance_estimator_model_20190403_11-42-48\ble_distance_estimator_model_20190403_11-42-48.ckpt"
        # self.model_path = r"E:\Code\Python\indoor_location\algorithm\ble_localization\localization\measure\saved_trained_model\ble_distance_estimator_model_20190103_20-34-00\ble_distance_estimator_model_20190103_20-34-00.ckpt"
        self.saver = tf.train.Saver()
        # 标签和网关距离预估
        self.step_size = 10
        self.anchor_id_scaling_factor = 10

        self.sigma_0m_radius = 0.2
        self.curve_fit_sample_num = 4
        self.var_range_0m_radius = 0.2

        self.position_list = []
        self.tri_centroid_list = []
        self.curve_fit_sample_x_list = []
        self.curve_fit_sample_y_list = []
        self.use_fitting = True
        self.w_prev = 0.5
        # for test
        self.test_list = []
        self.tri_centroid_list = []
        self.optimize_list = []

        self.initialization_step = 10
        time_win_interval = 5
        self.no_rssi_last = -50
        self.anchor_height = 1.9
        # self.movement_tracker = tracker()

    def dic_to_pd(self, gateway_dic):
        gw_data = []
        conn = ConnectDb()
        result = conn.select_table("DeviceManagement_device", '1', '1', 'mac', 'id')

        i_dic = dict(result) # 注意大小写
        print ('---i_dic----')
        print (i_dic)
        for info in gateway_dic:
            tmp_data = pd.DataFrame(gateway_dic[info])
            tmp_data["timestamp"] = pd.to_datetime(tmp_data["timestamp"], format='%Y/%m/%d %H:%M:%S')
            tmp_data["anchor"] = i_dic[info]
            print ("---info---")
            print (info)
            gw_data.append(tmp_data)
        start = max([each['timestamp'].min() for each in gw_data])
        end = min([each['timestamp'].max() for each in gw_data])
        time_window = datetime.timedelta(seconds=2)
        window_shift = datetime.timedelta(seconds=1)
        self.anchor_num = len(gw_data)
        return gw_data, start, end, time_window, window_shift

    def data_process(self):
        pass

    def distance_estimator(self):
        pass

    # def trilateral_localization(self):
    #     pass

    def data_smoothing(self, rssi_list, step_size):
        list_length = len(rssi_list)
        sorted_list = sorted(rssi_list)
        if self.step_size * 2 - 4 <= list_length < self.step_size * 2 - 1:
            rssi_list.remove(sorted_list[0])
            rssi_list.remove(sorted_list[-1])
        elif list_length >= self.step_size * 2 - 1:
            rssi_list.remove(sorted_list[0])
            rssi_list.remove(sorted_list[1])
            rssi_list.remove(sorted_list[-1])
            rssi_list.remove(sorted_list[-2])
        temp_len = len(rssi_list)
        rssi_batch = []
        if temp_len >= step_size:
            start_index = temp_len - step_size
            end_index = temp_len
            while start_index >= 0:
                rssi_batch.append(rssi_list[start_index:end_index])
                start_index -= 1
                end_index -=1
        else:
            if len(rssi_list) == 0:
                rssi_last = self.no_rssi_last
            else:
                rssi_last = rssi_list[-1]
            for i in range(step_size - temp_len):
                rssi_list.append(rssi_last)
            rssi_batch.append(rssi_list)
        return rssi_batch, np.mean(rssi_batch)

    def feature_scaling(self, input_df, scaling_meathod):
        rssi_min = -80
        rssi_max = -30
        rssi_mean = (rssi_min + rssi_max) / 2
        if scaling_meathod == 'z-score':
            # return (input_df - input_df.mean()) / input_df.std()
            return (input_df - rssi_mean) / input_df.std()
        elif scaling_meathod == 'min-max':
            # return (input_df - input_df.min()) / (input_df.max() - input_df.min())
            return (input_df - rssi_min) / (rssi_max - rssi_min)

    def gen_trunc_norm(self, up_bound=1, low_bound=-1, num_samples=1, mu=0, sigma=0.5):
        randn_num = np.random.normal(loc=mu, scale=sigma, size=num_samples)[0]
        if randn_num <= low_bound or randn_num >= up_bound:
            randn_num = self.gen_trunc_norm()
        return randn_num

    def calculate_radius(self, d):
        anchor_height = self.anchor_height
        if d <= anchor_height:
            print ("**d<=anchor_height", d)
            rand_r = self.gen_trunc_norm(up_bound=self.var_range_0m_radius, low_bound=-self.var_range_0m_radius, mu=0,
                                    sigma=self.sigma_0m_radius)
            R = np.abs(rand_r)
        else:
            R = np.sqrt(d ** 2 - anchor_height ** 2)
        return R

    def point_distance(self, a1, b1, a2, b2):
        return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)

    def f_2(self, x, a, b, c):
        return a * x * x + b * x + c

    def point_quadratic_distance(self, x0, y0, a0, a1, a2):
        p0 = 2 * a2 * a2
        p1 = 3 * a1 * a2
        p2 = a1 * a1 + 2 * a0 * a2 - 2 * a2 * y0 + 1
        p3 = a0 * a1 - a1 * y0 - x0
        coeff = [p0, p1, p2, p3]
        return np.roots(coeff)

    def fit_point(self, sample_x, sample_y):
        xt = sample_x[-1]
        yt = sample_y[-1]
        xt_1 = sample_x[-2]
        yt_1 = sample_y[-2]
        a2, a1, a0 = optimize.curve_fit(self.f_2, sample_x, sample_y)[0]
        roots = self.point_quadratic_distance(xt, yt, a0, a1, a2)
        root_list = []
        X = None
        Y = None
        if len(roots) > 1:
            isreal = np.isreal(roots)
            for i in range(len(isreal)):
                if isreal[i]:
                    x = roots[i]
                    y = a2 * x * x + a1 * x + a0
                    plt.plot(x, y, color='red', marker='.', markersize='10')
                    d1 = self.point_distance(x, y, xt, yt)  # distance to current timestamp observation
                    d2 = self.point_distance(x, y, xt_1, yt_1)  # distance to last timestamp observation
                    root_list.append(((x, y), d1, d2))
            sort_d1 = sorted(root_list, key=operator.itemgetter(1))
            sort_d2 = sorted(root_list, key=operator.itemgetter(2))
            if sort_d1[0][0] == sort_d2[0][0]:
                X = sort_d1[0][0][0]
                Y = sort_d1[0][0][1]
            else:
                X = (sort_d1[0][0][0] + sort_d2[0][0][0]) / 2
                Y = (sort_d1[0][0][1] + sort_d2[0][0][1]) / 2
        else:
            X = roots[0]
            Y = a2 * X * X + a1 * X + a0

        weight = 0.1
        fitting_x = weight * xt + (1 - weight) * X
        fitting_y = weight * yt + (1 - weight) * Y

        return float(fitting_x), float(fitting_y)

    def predict_distance(self, sess, gw_data, start, time_window):
        observation_list = []
        for i in range(self.anchor_num):
            anchor_id = gw_data[i]['anchor'][0]
            rssi_pd = gw_data[i][(start <= gw_data[i]['timestamp']) & (start + time_window > gw_data[i]['timestamp'])]
            smooth_rssi_list, rssi_avg = self.data_smoothing(list(rssi_pd['rssi']), self.step_size)
            rssi_bat = np.reshape(smooth_rssi_list, [len(smooth_rssi_list), self.step_size, 1])
            anchor_id_list = [int(anchor_id) / self.anchor_id_scaling_factor for i in range(len(smooth_rssi_list))]
            anchor_id_bat = np.reshape(anchor_id_list, [len(smooth_rssi_list), 1, 1])
            rssi_bat = self.feature_scaling(rssi_bat, 'min-max')
            pred_distance_list = sess.run(self.y, feed_dict={self.x_: rssi_bat, self.a_: anchor_id_bat})

            # if anchor_id == 5 and pred_result > 4:
            #     pred_result *= 0.9
            radius = self.calculate_radius(np.mean(pred_distance_list))  # 计算
            if anchor_id == 5:
                radius = -0.2253 * radius + 4.6496
            elif anchor_id == 4:
                radius = 0.5357 * radius + 1.8409

            observation = (str(anchor_id), (radius, rssi_avg))
            observation_list.append(observation)
        return observation_list

    def trilateral_localization(self, sorted_observation, iteration):
        # from test import count_xy
        # obs_combination = [[sorted_observation[i] for i in [0, 1, 2]], [sorted_observation[i] for i in [0, 1, 3]],
        #                    [sorted_observation[i] for i in [0, 1, 4]], [sorted_observation[i] for i in [0, 2, 3]],
        #                    [sorted_observation[i] for i in [0, 2, 4]], [sorted_observation[i] for i in [0, 3, 4]],
        #                    [sorted_observation[i] for i in [1, 2, 3]], [sorted_observation[i] for i in [1, 2, 4]],
        #                    [sorted_observation[i] for i in [1, 3, 4]], [sorted_observation[i] for i in [2, 3, 4]]]
        obs_combination = [[sorted_observation[i] for i in [0, 1, 2]], [sorted_observation[i] for i in [0, 1, 3]],
                           [sorted_observation[i] for i in [1, 2, 3]], [sorted_observation[i] for i in [0, 2, 3]]]
        # obs_combination = [[sorted_observation[i] for i in [0, 1, 2]]]

        tri_centroid_list_temp = []
        for item in obs_combination:
            print ("****gatewayid***",item)
            get_centroid, centroid = trilateral_centroid_positioning(item)
            if get_centroid:
                tri_centroid_list_temp.append(centroid)
        print ("*** todo",tri_centroid_list_temp)
        if len(tri_centroid_list_temp) > 0:
            if iteration >= 1:
                p_prev = self.position_list[-1]
                if len(tri_centroid_list_temp) >= 3:
                    weighted_x_mean = (np.sum(tri_centroid_list_temp, axis=0)[0]) / len(tri_centroid_list_temp)
                    weighted_y_mean = (np.sum(tri_centroid_list_temp, axis=0)[1]) / len(tri_centroid_list_temp)
                    sigma_x = np.std(tri_centroid_list_temp, axis=0)[0]
                    sigma_y = np.std(tri_centroid_list_temp, axis=0)[1]
                else:

                    weighted_x_mean = (p_prev[0] + np.sum(tri_centroid_list_temp, axis=0)[0]) / (
                            len(tri_centroid_list_temp) + 1)
                    weighted_y_mean = (p_prev[1] + np.sum(tri_centroid_list_temp, axis=0)[1]) / (
                            len(tri_centroid_list_temp) + 1)
                    temp = tri_centroid_list_temp + [p_prev]
                    sigma_x = np.std([item[0] for item in temp])[0]
                    sigma_y = np.std([item[1] for item in temp])[0]

                thr_x = 1.8 + 0.3 * sigma_x
                thr_y = 1.8 + 0.3 * sigma_y
                print("mean_x = %.3f, sigma_x = %.3f, thr_x = %.3f" % (weighted_x_mean, sigma_x, thr_x))
                print("mean_y = %.3f, sigma_y = %.3f, thr_y = %.3f" % (weighted_y_mean, sigma_y, thr_y))
                # test
                thr_x = 0.5
                thr_y = 0.8
                # end

                tri_centroid_filter_list = list(
                    filter(lambda x: abs(x[0] - weighted_x_mean) < thr_x, tri_centroid_list_temp))
                tri_centroid_filter_list = list(
                    filter(lambda y: abs(y[1] - weighted_y_mean) < thr_y, tri_centroid_filter_list))

                if len(tri_centroid_filter_list) <= 0:
                    thr_x = 0.5
                    thr_y = 1
                    # end

                    tri_centroid_filter_list = list(
                        filter(lambda x: abs(x[0] - weighted_x_mean) < thr_x, tri_centroid_list_temp))
                    tri_centroid_filter_list = list(
                        filter(lambda y: abs(y[1] - weighted_y_mean) < thr_y, tri_centroid_filter_list))

                if len(tri_centroid_filter_list) > 0:
                    print("11111111111111111111111111111")
                    if (len(tri_centroid_filter_list) < len(tri_centroid_list_temp)):
                        print("33333333333333333333333333333")
                    print(tri_centroid_filter_list)
                    tri_centroid = np.mean(tri_centroid_filter_list, axis=0)
                else:
                    print("22222222222222222222222222222")
                    print(tri_centroid_list_temp)
                    prev_temp_distance = 1000
                    for ii in range(len(tri_centroid_list_temp)):
                        temp_tri_centroid = tri_centroid_list_temp[ii]
                        # print("temp_tri_centroid =", temp_tri_centroid)
                        temp_distance = self.point_distance(temp_tri_centroid[0], weighted_x_mean, temp_tri_centroid[1],
                                                            weighted_y_mean)
                        if temp_distance < prev_temp_distance:
                            temp_x = temp_tri_centroid[0]
                            temp_y = temp_tri_centroid[1]
                            prev_temp_distance = temp_distance
                    tri_centroid = [temp_x, temp_y]

                    # tri_centroid = np.mean(tri_centroid_filter_list, axis=0)
                    # tri_centroid = np.mean(tri_centroid_list_temp, axis=0)
                print(
                    "Estimate position after remove outliers: X:%.4f  Y:%.4f" % (tri_centroid[0], tri_centroid[1]))
            else:
                # ￥￥￥￥￥￥￥￥￥￥￥￥￥
                weighted_x_mean = (np.sum(tri_centroid_list_temp, axis=0)[0]) / len(tri_centroid_list_temp)
                weighted_y_mean = (np.sum(tri_centroid_list_temp, axis=0)[1]) / len(tri_centroid_list_temp)
                thr_x = 0.5
                thr_y = 0.8
                # end

                tri_centroid_filter_list = list(
                    filter(lambda x: abs(x[0] - weighted_x_mean) < thr_x, tri_centroid_list_temp))
                tri_centroid_filter_list = list(
                    filter(lambda y: abs(y[1] - weighted_y_mean) < thr_y, tri_centroid_filter_list))
                tri_centroid = np.mean(tri_centroid_filter_list, axis=0)
                # if len(tri_centroid_filter_list) <= 0:
                #     print ("*&&&&&&&&&&&&&&&&&")
                #     thr_x = 1
                #     thr_y = 1
                #     # end
                #
                #     tri_centroid_filter_list = list(
                #         filter(lambda x: abs(x[0] - weighted_x_mean) < thr_x, tri_centroid_list_temp))
                #     tri_centroid_filter_list = list(
                #         filter(lambda y: abs(y[1] - weighted_y_mean) < thr_y, tri_centroid_filter_list))
                # tri_centroid = np.mean(tri_centroid_filter_list, axis=0)
                # ￥￥￥￥￥￥￥￥￥￥￥￥￥

                if len(tri_centroid_filter_list) <= 0:
                    tri_centroid = [np.mean(tri_centroid_list_temp, axis=0)[0],
                                    np.mean(tri_centroid_list_temp, axis=0)[1]]
                    x = sorted(tri_centroid_list_temp, key=lambda x:x[0])
                    print ("**todo1**", tri_centroid)
                    # tri_centroid[0] = np.mean(tri_centroid_list_temp, axis=0)[0]
                    # tri_centroid[1] = np.mean(tri_centroid_list_temp, axis=0)[1]
                    # print("**todo1**", tri_centroid)
                    # tri_centroid = count_xy(tri_centroid_list_temp)

            self.tri_centroid_list.append(tri_centroid)
        else:
            tri_centroid = None
        return tri_centroid

    def start_process(self,tag, data):
        gw_data, start, end, time_window, window_shift = self.dic_to_pd(data)
        print ("****start:%s, end:%s", start, end)
        with tf.Session() as sess:
            print ("*************tf.session")
            self.saver.restore(sess, self.model_path)
            iteration = 0
            while start + time_window <= end:
                print("****start+time_window:%s, end:%s", start + time_window, end)
                print ("iteration: ", iteration)
                print("Phase 1: Predict distance")
                observation_list = self.predict_distance(sess, gw_data, start, time_window)
                print (observation_list)

                print("Phase 2: Trilateral localization")
                sorted_observation = sorted(observation_list, key=operator.itemgetter(1), reverse=False)[0:4]
                print(sorted_observation)
                tri_centroid = self.trilateral_localization(sorted_observation, iteration)
                print ("centroid: ", tri_centroid)

                # Kalman Filter
                print("Phase 3: Kalman filter")
                self.movement_tracker = tracker()
                z = None  # Observation
                if tri_centroid is None:
                    last_state = self.movement_tracker.x
                    # TODO: check last_state
                    z = (last_state[0], last_state[2])
                    self.tri_centroid_list.append((last_state[0], last_state[1]))
                else:
                    if self.use_fitting:
                        # 用观测值拟合后输入到filter中
                        if len(self.curve_fit_sample_x_list) >= self.curve_fit_sample_num:
                            self.curve_fit_sample_x_list.pop(0)
                            self.curve_fit_sample_y_list.pop(0)
                        self.curve_fit_sample_x_list.append(tri_centroid[0])
                        self.curve_fit_sample_y_list.append(tri_centroid[1])
                        if len(self.curve_fit_sample_x_list) >= self.curve_fit_sample_num:
                            print("Tri-centroid: X: %.2f Y: %.2f" % (tri_centroid[0], tri_centroid[1]))
                            print("Curve fit sample: x[%s] y[%s]" % (self.curve_fit_sample_x_list, self.curve_fit_sample_y_list))
                            fitting_x, fitting_y = self.fit_point(self.curve_fit_sample_x_list, self.curve_fit_sample_y_list)
                            print("Fitting point: X: %.4f  Y:%.4f" % (fitting_x, fitting_y))
                            z = (fitting_x, fitting_y)
                        else:
                            z = (tri_centroid[0], tri_centroid[1])
                    else:
                        z = (tri_centroid[0], tri_centroid[1])

                z = np.array(z).reshape(2, 1)
                if iteration < self.initialization_step:
                    R, Q = gen_R_and_Q(0.5, 0.1)
                else:
                    # R, Q = gen_R_and_Q(0.3, 0.3)
                    R, Q = gen_R_and_Q(0.8, 0.08)
                    # R, Q = gen_R_and_Q(0.8, 0.05)

                self.movement_tracker.R = R
                self.movement_tracker.Q = Q

                self.movement_tracker.predict()
                self.movement_tracker.update(z)

                # Output
                output_x = self.movement_tracker.x[0]
                output_y = self.movement_tracker.x[2]
                output_timestamp = start + time_window
                print("Restuls:  timestamp: %s  X: %.2f  Y: %.2f" % (output_timestamp, output_x, output_y))
                print (datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

                db = ConnectDb()
                db.insert_table([(tag, "{0} {1}".format(output_x[0],output_y[0]), output_timestamp)])
                position = [output_x, output_y]
                self.position_list.append(position)
                start += window_shift
                iteration += 1
            print("***")
            print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))




