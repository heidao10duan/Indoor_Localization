"""
1、从文本中读取数据
2、构造
3、模型训练
"""
import tensorflow as tf


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

