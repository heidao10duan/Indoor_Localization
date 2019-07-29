# -*- coding: utf-8 -*-
'''
Using LSTM to predict distance based on 'rssi' value

# @Time    : 2018/12/20 17:20
# @Author  : SJQ
# @File    : anchor_train.py
'''
import os
import math
import time
import pandas as pd
import numpy as np
import tensorflow as tf

# ======================
#   parameters setting
# ======================
# data_folder_path = r"E:\Code\Python\indoor_location\algorithm\ble_localization\localization\measure\training-data"
data_folder_path = r"E:\result"
# data_path = 'training_data_test'
feature_to_input = ['rssi']
feature_to_predict = ['distance']
if_do_fea_scaling = True
scaling_method = 'min-max'
feature_to_scale = ['rssi']

step_size = 10  # based on received rssi values in 2 seconds
n_features = len(feature_to_input)
n_step_ahead = 1
n_pred_class = 1
# batch_size = 200
lstm_layers = 1

if_do_test = False
tra_tst_split_ratio = 0.9

# training parameter
if_concat_anchor_id = True
anchor_id_scaling_factor = 10
distance_scaling_factor = 1
epochs = 10000
print_interval = 100
learning_rate = 0.03
# 0.08

if_use_decaying_lr = True
lr_decay_rate = 0.9999
n_l1_out_unit = 100
n_l2_out_unit = 30
n_l3_out_unit = 1
n_fc_layer = 3

save_model_rmse_threshold = 0.8
save_model_folder_name = r'E:\save_model_folder_name'


###############################################################################

def feature_scaling(input_df, scaling_meathod):
    rssi_min = -80
    rssi_max = -30
    rssi_mean = (rssi_min + rssi_max) / 2
    if scaling_meathod == 'z-score':
        # return (input_df - input_df.mean()) / input_df.std()
        return (input_df - rssi_mean) / input_df.std()
    elif scaling_meathod == 'min-max':
        # return (input_df - input_df.min()) / (input_df.max() - input_df.min())
        return (input_df - rssi_min) / (rssi_max - rssi_min)


def input_reshape(input_pd, n_step, n_features):
    start = 0
    end = input_pd.shape[0] - n_step - 1
    temp_pd = input_pd[start: end + n_step]
    output_pd = list(map(lambda y: temp_pd[y:y + n_step], range(0, end - start + 1, 1)))
    output_temp = list(map(lambda x: np.array(output_pd[x]).reshape([-1]), range(len(output_pd))))
    output = np.reshape(output_temp, [len(output_temp), n_step, n_features])
    return output


def read_and_concat_data():
    print("--- Start constructing train and test data ---")
    # read csv files
    file_names = os.listdir(data_folder_path)
    concat_data_array = None
    for file_name in file_names:
        if file_name.endswith('.csv'):
            print("read file: %s" % file_name)
            file_dir = os.path.join(data_folder_path, file_name)
            read_df = pd.read_csv(file_dir)
            anchor_id = read_df.drop_duplicates('anchor')['anchor'][0]
            distance = read_df.drop_duplicates('distance')['distance'][0]
            # TODO: feature scaling
            if if_do_fea_scaling:
                temp_df = feature_scaling(read_df[feature_to_scale], scaling_method)
                read_df[feature_to_scale] = temp_df
                anchor_id = anchor_id / anchor_id_scaling_factor
                distance = distance / distance_scaling_factor
            read_df = read_df[feature_to_input]
            reshaped_data_array = input_reshape(read_df, step_size, n_features)
            n_batch = len(reshaped_data_array)
            stack_anchor_id = np.expand_dims(np.array([[anchor_id] * n_features] * n_batch), axis=1)
            stack_distance = np.expand_dims(np.array([[distance] * n_features] * n_batch), axis=1)
            reshaped_data_array = np.concatenate((reshaped_data_array, stack_anchor_id), axis=1)
            reshaped_data_array = np.concatenate((reshaped_data_array, stack_distance), axis=1)
            if concat_data_array is None:
                concat_data_array = reshaped_data_array
            else:
                concat_data_array = np.concatenate((concat_data_array, reshaped_data_array), axis=0)

    return concat_data_array


def construct_train_test_set(concat_data):
    def construct_batch_data(input_data):
        rssi_batch_data = input_data[:, 0: step_size, :]
        anchor_id_batch_data = input_data[:, step_size, :]
        label_batch_data = input_data[:, step_size + 1, :]
        # TODO: need to check whether the input format is consistant with tensorflow
        return rssi_batch_data, anchor_id_batch_data, label_batch_data

    np.random.shuffle(concat_data)
    np.random.shuffle(concat_data)
    n_concat_batch = len(concat_data)
    # print("n_concat_batch =", n_concat_batch)
    if if_do_test is False:
        train_data = concat_data
        test_data = None
        train_rssi_bat_data, train_anchor_id_bat_data, train_label_bat_data \
            = construct_batch_data(train_data)
        if len(np.shape(train_anchor_id_bat_data)) == 2:
            train_anchor_id_bat_data = np.expand_dims(train_anchor_id_bat_data, axis=1)
        test_rssi_bat_data, test_anchor_id_bat_data, test_label_bat_data = None, None, None
    else:
        train_stop_index = int(n_concat_batch * tra_tst_split_ratio)
        test_start_index = train_stop_index + 1
        train_data = concat_data[0: train_stop_index]
        test_data = concat_data[test_start_index: n_concat_batch]
        train_rssi_bat_data, train_anchor_id_bat_data, train_label_bat_data \
            = construct_batch_data(train_data)
        if len(np.shape(train_anchor_id_bat_data)) == 2:
            train_anchor_id_bat_data = np.expand_dims(train_anchor_id_bat_data, axis=1)
        test_rssi_bat_data, test_anchor_id_bat_data, test_label_bat_data \
            = construct_batch_data(test_data)
        if len(np.shape(test_anchor_id_bat_data)) == 2:
            test_anchor_id_bat_data = np.expand_dims(test_anchor_id_bat_data, axis=1)
    return train_rssi_bat_data, train_anchor_id_bat_data, train_label_bat_data, \
           test_rssi_bat_data, test_anchor_id_bat_data, test_label_bat_data


def train_distance_estimator_model():
    def lstm(input, features, n_steps, n_lstm_layers, scope_name):
        input = tf.transpose(input, [1, 0, 2])
        input = tf.reshape(input, [-1, features])
        input = tf.split(input, n_steps, 0)
        with tf.variable_scope(scope_name):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=features)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_lstm_layers)
            output, state = tf.nn.static_rnn(cell, input, dtype=tf.float32)
            output_last = output[-1]
        return output, output_last

    # Get train and test data
    concat_batch_data = read_and_concat_data()
    train_rssi_bat_data, train_anchor_id_bat_data, train_label_bat_data, \
    test_rssi_bat_data, test_anchor_id_bat_data, test_label_bat_data \
        = construct_train_test_set(concat_batch_data)

    # Reset graph
    tf.reset_default_graph()

    # Build model
    lr_ = tf.placeholder(tf.float32, [])
    x_ = tf.placeholder(tf.float32, [None, step_size, n_features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    a_ = tf.placeholder(tf.float32, [None, 1, n_features])

    lstm_out, lstm_out_last = lstm(x_, n_features, step_size, lstm_layers, 'lstm_prediction')
    lstm_out_tensor = tf.convert_to_tensor(lstm_out)
    print("shape(lstm_out_tensor) =", lstm_out_tensor.get_shape())
    lstm_out_tensor = tf.transpose(lstm_out_tensor, [1, 0, 2])
    print("shape(lstm_out_tensor_transpose) =", lstm_out_tensor.get_shape())
    print("shape(a_) =", a_.get_shape())
    concat_lstm_anchor = tf.concat([lstm_out_tensor, a_], axis=1)
    print("shape(concat_lstm_anchor) =", concat_lstm_anchor.get_shape())

    if if_concat_anchor_id is True:
        layer1_in = concat_lstm_anchor
        layer1_in = tf.squeeze(layer1_in, axis=2)
    else:
        layer1_in = lstm_out_tensor
        layer1_in = tf.squeeze(layer1_in, axis=2)

    layer1_out = tf.layers.dense(inputs=layer1_in, units=n_l1_out_unit, activation=tf.nn.relu)
    layer2_out = tf.layers.dense(inputs=layer1_out, units=n_l2_out_unit, activation=tf.nn.relu)
    layer3_out = tf.layers.dense(inputs=layer2_out, units=n_l3_out_unit, activation=tf.nn.relu)
    y = layer3_out

    # y = tf.layers.dense(inputs=layer1_in, units=1, activation=tf.nn.relu)

    loss = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))
    train_op = tf.train.AdamOptimizer(lr_).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for ii in range(epochs):
            if if_use_decaying_lr:
                lr = learning_rate * math.pow(lr_decay_rate, ii)
            else:
                lr = learning_rate
            sess.run(train_op, feed_dict={x_: train_rssi_bat_data,
                                          a_: train_anchor_id_bat_data,
                                          y_: train_label_bat_data,
                                          lr_: lr})
            if (ii + 1) % print_interval == 0:
                cost = sess.run(loss, feed_dict={x_: train_rssi_bat_data,
                                                 a_: train_anchor_id_bat_data,
                                                 y_: train_label_bat_data,
                                                 lr_: lr})
                print("iteration = %d,  lr = %.8f,  loss = %.8f" % (ii + 1, lr, cost))

        if cost < save_model_rmse_threshold:
            print("---------- save model --------- \n")
            local_time = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
            folder_name = 'ble_distance_estimator_model_' + local_time
            folder_path = os.path.join(save_model_folder_name, folder_name)
            model_name = 'ble_distance_estimator_model_' + local_time + '.ckpt'
            if not os.path.exists(save_model_folder_name):
                os.makedirs(save_model_folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_model_path = os.path.join(folder_path, model_name)
            saver.save(sess, save_model_path)

        print("---------- validation ---------")
        valid_result = sess.run(y, feed_dict={x_: train_rssi_bat_data, a_: train_anchor_id_bat_data})
        train_anchor_id_list = train_anchor_id_bat_data
        # print("np.shape(train_anchor_id_list) =", np.shape(train_anchor_id_list))
        if if_do_fea_scaling:
            train_anchor_id_list = train_anchor_id_list * anchor_id_scaling_factor
            valid_result = valid_result * distance_scaling_factor
            train_label_bat_data = train_label_bat_data * distance_scaling_factor
        train_anchor_id_list = list(map(lambda x: int(np.around(x)), list(train_anchor_id_list)))
        validation_mse = np.mean(np.square(valid_result - train_label_bat_data))
        validation_rmse = np.sqrt(np.mean(np.square(valid_result - train_label_bat_data)))
        validation_mae = np.mean(np.abs(valid_result - train_label_bat_data))
        validation_mape = np.mean(np.abs((valid_result - train_label_bat_data) / train_label_bat_data))
        print('validation_mse = %.8f' % (validation_mse))
        print('validation_rmse = %.8f' % (validation_rmse))
        print('validation_mae = %.8f' % (validation_mae))
        print('validation_mape = %.8f' % (validation_mape))
        train_result_dict = {'anchor_id': train_anchor_id_list,
                             'train_labelled_distance': train_label_bat_data.flatten(),
                             'train_estimated_distance': valid_result.flatten()}
        train_result_df = pd.DataFrame(train_result_dict)
        train_result_df['train_estimation_error'] = abs(train_result_df['train_labelled_distance'] \
                                                        - train_result_df['train_estimated_distance'])
        train_result_sts = calc_statistics(train_result_df,
                                           col_name_label='train_labelled_distance',
                                           col_name_diff='train_estimation_error')

        test_result_df = None
        test_result_sts = None, None
        if if_do_test is True:
            print("-------- model testing --------")
            pred_result = sess.run(y, feed_dict={x_: test_rssi_bat_data, a_: test_anchor_id_bat_data})
            test_anchor_id_list = test_anchor_id_bat_data
            if if_do_fea_scaling:
                test_anchor_id_list = test_anchor_id_list * anchor_id_scaling_factor
                pred_result = pred_result * distance_scaling_factor
                test_label_bat_data = test_label_bat_data * distance_scaling_factor
            test_anchor_id_list = list(map(lambda x: int(np.around(x)), list(test_anchor_id_list)))
            prediction_mse = np.mean(np.square(pred_result - test_label_bat_data))
            prediction_rmse = np.sqrt(np.mean(np.square(pred_result - test_label_bat_data)))
            prediction_mae = np.mean(np.abs(pred_result - test_label_bat_data))
            prediction_mape = np.mean(np.abs((pred_result - test_label_bat_data) / test_label_bat_data))
            print('testing_mse = %.8f' % (prediction_mse))
            print('testing_rmse = %.8f' % (prediction_rmse))
            print('testing_mae = %.8f' % (prediction_mae))
            print('testing_mape = %.8f' % (prediction_mape))
            test_result_dict = {'anchor_id': test_anchor_id_list,
                                'test_labelled_distance': test_label_bat_data.flatten(),
                                'test_estimated_distance': pred_result.flatten()}
            test_result_df = pd.DataFrame(test_result_dict)
            test_result_df['test_estimation_error'] = abs(test_result_df['test_labelled_distance'] \
                                                          - test_result_df['test_estimated_distance'])
            test_result_sts = calc_statistics(test_result_df,
                                              col_name_label='test_labelled_distance',
                                              col_name_diff='test_estimation_error')

        return train_result_df, train_result_sts, test_result_df, test_result_sts


def calc_statistics(res_df, col_name_label, col_name_diff):
    tot_label_list = []
    tot_mean_diff_list = []
    measured_label_list = list(res_df.drop_duplicates(col_name_label)[col_name_label])
    anchor_id_list = list(res_df.drop_duplicates('anchor_id')['anchor_id'])
    for tot_label in measured_label_list:
        tot_mean_diff = np.mean(list(res_df[res_df[col_name_label] == tot_label][col_name_diff]))
        tot_label_list.append(tot_label)
        tot_mean_diff_list.append(tot_mean_diff)
    tot_res_sts_dict = {'labelled_distance': tot_label_list, 'mean_estimated_error': tot_mean_diff_list}
    tot_res_sts_df = pd.DataFrame(tot_res_sts_dict)

    sub_anchor_id_list = []
    sub_label_list = []
    sub_mean_diff_list = []
    for anchor_id in anchor_id_list:
        for sub_label in measured_label_list:
            temp_df = res_df[(res_df['anchor_id'] == anchor_id) & (res_df[col_name_label] == sub_label)][col_name_diff]
            if temp_df.empty is False:
                sub_mean_diff = np.mean(list(temp_df))
                sub_anchor_id_list.append(anchor_id)
                sub_label_list.append(sub_label)
                sub_mean_diff_list.append(sub_mean_diff)
    sub_res_sts_dict = {'anchor_id': sub_anchor_id_list, 'labelled_distance': sub_label_list,
                        'mean_estimated_error': sub_mean_diff_list}
    sub_res_sts_df = pd.DataFrame(sub_res_sts_dict)
    res_sts_dict = {'tot_res_sts': tot_res_sts_df, 'sub_anchor_res_sts': sub_res_sts_df}
    return res_sts_dict


def distance_model():
    tf.reset_default_graph()

    def lstm(input, features, n_steps, n_lstm_layers, scope_name):
        input = tf.transpose(input, [1, 0, 2])
        input = tf.reshape(input, [-1, features])
        input = tf.split(input, n_steps, 0)
        with tf.variable_scope(scope_name):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=features)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_lstm_layers)
            output, state = tf.nn.static_rnn(cell, input, dtype=tf.float32)
            output_last = output[-1]
        return output, output_last

    # Build model
    x_ = tf.placeholder(tf.float32, [None, step_size, n_features])
    a_ = tf.placeholder(tf.float32, [None, 1, n_features])

    lstm_out, lstm_out_last = lstm(x_, n_features, step_size, lstm_layers, 'lstm_prediction')
    lstm_out_tensor = tf.convert_to_tensor(lstm_out)
    lstm_out_tensor = tf.transpose(lstm_out_tensor, [1, 0, 2])
    concat_lstm_anchor = tf.concat([lstm_out_tensor, a_], axis=1)

    if if_concat_anchor_id is True:
        layer1_in = concat_lstm_anchor
        layer1_in = tf.squeeze(layer1_in, axis=2)
    else:
        layer1_in = lstm_out_tensor
        layer1_in = tf.squeeze(layer1_in, axis=2)

    layer1_out = tf.layers.dense(inputs=layer1_in, units=n_l1_out_unit, activation=tf.nn.relu)
    layer2_out = tf.layers.dense(inputs=layer1_out, units=n_l2_out_unit, activation=tf.nn.relu)
    layer3_out = tf.layers.dense(inputs=layer2_out, units=n_l3_out_unit, activation=tf.nn.relu)
    y = layer3_out

    return y, x_, a_


if __name__ == '__main__':
    import datetime
    print("start:", datetime.datetime.now().strftime("%H:%M:%S"))
    train_distance_estimator_model()
    print("end:", datetime.datetime.now().strftime("%H:%M:%S"))